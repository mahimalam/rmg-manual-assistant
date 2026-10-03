import json
from pathlib import Path

import httpx
import pymupdf
import pytest

from app.structured import _assemble, _local_page, _searchable


@pytest.fixture(scope='module')
def control_units():
    path = Path(__file__).resolve().parents[1]/'S-7200A Service Manuel.pdf'
    if not path.is_file():
        pytest.skip("External S-7200A service manual is not distributed with this repository")
    with pymupdf.open(path)as pdf:
        pages=[{**_local_page(pdf[n-1]),'page':n,'visual_units':[],
                'visual_review':'unreviewed'}for n in (11,13,15,16)]
    return _assemble({'id':'m','sha256':'a'*64},pages)


def test_headerless_dip_table_keeps_first_row_and_merged_owner(control_units):
    rows=[u for u in control_units if u['page']==11 and u.get('setting_id')=='dip:1']
    assert {u.get('setting_value')for u in rows}=={'ON','OFF'}
    assert all(u['table_rows'][0][0]=='1'for u in rows)
    off=next(u for u in rows if u['setting_value']=='OFF')
    assert off['note_references']==['1']
    assert 'kept raised' in off['search_text']
    assert not off['blocking_issues']
    assert all('Presser foot position'not in ' '.join(u['table_headers'])
               for u in control_units if u['page']==11 and u['kind']=='table')


def test_new_headerless_switch_table_uses_its_own_numbers(tmp_path):
    pdf=pymupdf.open();p=pdf.new_page()
    p.insert_text((40,40),'4-3. Setting the DIP switches')
    for x in (40,90,250,300,500):p.draw_line((x,80),(x,140))
    for y in (80,110,140):p.draw_line((40,y),(500,y))
    for x,y,t in [(50,100,'7'),(100,100,'Cooling control'),(260,100,'ON'),(310,100,'Fan enabled'),
                   (50,130,'7'),(100,130,'Cooling control'),(260,130,'OFF'),(310,130,'Fan disabled')]:
        p.insert_text((x,y),t)
    from app.ingest import add_pdf,delete_pdf
    from app.db import Database
    from app.structured import process_manual
    content=pdf.tobytes();pdf.close()
    db=Database(tmp_path/'study.sqlite3');manual,added=add_pdf(db,tmp_path,'new-controls.pdf',content)
    assert added
    first=process_manual(db,tmp_path,manual['id'])
    reopened=Database(tmp_path/'study.sqlite3')
    second=process_manual(reopened,tmp_path,manual['id'])
    assert first['release_id']==second['release_id']and reopened.ingestion_spend()['attempts']==0
    units=[json.loads(u['payload_json'])for u in reopened.units(manual['id'])]
    rows=[u for u in units if u.get('setting_id')=='dip:7']
    assert {u.get('setting_value')for u in rows}=={'ON','OFF'}
    assert all(_searchable(u)for u in rows)
    delete_pdf(reopened,tmp_path,manual['id'])
    assert not reopened.units(manual['id'])


def test_prerequisite_uses_only_matching_dip_branch(control_units):
    lookup={u['id']:u for u in control_units}
    row=next(u for u in control_units if u.get('setting_id')=='function:40')
    targets=[lookup[d['id']]for d in row['dependencies']if d['relation']=='prerequisite']
    dip=next(u for u in targets if u.get('setting_id')=='dip:1')
    assert dip['setting_value']=='OFF'
    assert any(lookup[d['id']].get('note_ids')==['1']for d in dip['dependencies'])
    on=next(u for u in control_units if u.get('setting_id')=='function:55')
    assert any(lookup[d['id']].get('setting_value')=='ON'for d in on['dependencies'])
    trim=next(u for u in control_units if u.get('setting_id')=='function:12')
    assert {'when_value':'0','setting_id':'dip:1','value':'OFF'}in trim['setting_exclusions']


def test_unresolved_table_relationship_is_quarantined():
    unit={'kind':'table','parent_id':'table','table_rows':[['1',None,'OFF']],
          'blocking_issues':[],'flags':['Merged/blank cells or headers require visual source review']}
    assert not _searchable(unit)


def test_unresolved_native_row_cannot_return_through_a_dependency(tmp_path):
    from app.knowledge import UnitDraft
    from app.db import Database
    from app.retrieval import Retriever
    table=UnitDraft(kind='table',title='Unresolved table',text='Missing relationship.',
                    table_headers=['No.','Operation'],table_rows=[['1',None]],
                    uncertainties=['Merged/blank cells or headers require visual source review'])
    page={'page':1,'printed_label':'','method':'text','text':'Missing relationship.',
          'units':[table.model_dump()],'visual_units':[],'visual_review':'unreviewed'}
    unit=_assemble({'id':'m','sha256':'a'*64},[page])[0]
    assert unit['blocking_issues']
    db=Database(tmp_path/'study.sqlite3')
    with db.connect()as c:
        c.execute("INSERT INTO manuals VALUES ('m','Fixture','source.pdf',?,1,'ready','test',NULL,'baseline')",('a'*64,))
        c.execute('INSERT INTO knowledge_units VALUES (?,?,?,?,?,?,?)',(unit['id'],'m',1,'table',unit['search_text'],json.dumps(unit),'fixture'))
    with pytest.raises(ValueError,match='unresolved'):
        Retriever(db,tmp_path,'synthetic')._expand([{'id':unit['id'],'manual_id':'m','title':'Fixture','page':1,'text':unit['search_text']}],{})


def test_compound_question_preserves_each_symptom():
    from app.questions import question_parts
    q=('Brother S-7200A: the presser foot drops when the treadle returns to neutral after thread trimming. '
       'Additionally, after the knee switch lowers the foot while stopped, the treadle cannot raise it. '
       'Which DIP switch and function explain both issues?')
    parts=question_parts(q)
    assert len(parts)==2
    assert 'thread trimming' in parts[0]and 'cannot raise' in parts[1]
    assert question_parts('What prerequisite enables Function No. 39?')==['What prerequisite enables Function No. 39?']


@pytest.mark.parametrize('question', [
    'S-7200A thread keeps breaking and machine produces unusual noise. What should I check?',
    'On S-7200A machine, thread is repeatedly breaking and making loud noise. What to test?',
    'On S-7200A machine, thread is repeatedly breaking and making a lot of noise. What should be checked?',
    'S-7200A সুতা বারবার ছিঁড়ে যাচ্ছে এবং অনেক সাউন্ড করছে এর কারণ কী?',
])
def test_thread_breakage_and_audible_noise_are_separate_retrieval_intents(question):
    from app.questions import question_parts
    assert len(question_parts(question)) == 2


def test_audible_noise_cannot_be_explained_by_an_electrical_interference_only_excerpt():
    from app.questions import audible_noise_question, electrical_noise_only
    assert audible_noise_question('The machine is making a lot of noise.')
    assert electrical_noise_only('Use the machine away from sources of strong electrical noise such as welders.')
    assert not electrical_noise_only('Electrical noise is accompanied by an audible buzzing sound.')
    assert not audible_noise_question('Electrical noise from a welder interferes with the machine. What should I check?')
    assert not audible_noise_question('How do I reduce noise in ADC measurements?')


def test_generic_noise_does_not_request_lifter_response_settings():
    from app.questions import unrelated_noise_setting
    note = 'Only enabled when Function No. 35 is set to 0. Response is fastest and operating noise greatest.'
    assert unrelated_noise_setting('The machine is making a lot of noise.', note)
    assert not unrelated_noise_setting('The presser foot lifter is making loud noise. Which setting controls it?', note)
    assert not unrelated_noise_setting('The machine is making noise.', 'If abnormal noises occur, turn off power.')


@pytest.mark.parametrize('question',[
    'The foot drops at neutral and after knee use the treadle cannot raise the foot.',
    'ফুট নিচে পড়ে যাচ্ছে। এছাড়া হাঁটু সুইচ ব্যবহার করলে প্যাডেল দিয়ে ফুট ওঠে না।',
])
def test_linked_independent_symptoms_are_split(question):
    from app.questions import question_parts
    assert len(question_parts(question))==2


def test_compound_retrieval_collects_both_intents(monkeypatch):
    from app.retrieval import Retriever
    q='The thread snaps. Additionally, the motor will not start. Which settings cause both?'
    r=object.__new__(Retriever)
    def search(text,manual_ids=None,limit=4):
        key='motor'if 'motor' in text and 'thread'not in text else 'thread'
        return [{'id':key,'text':text,'page':1,'manual_id':'m','score':.8}]
    monkeypatch.setattr(r,'_search_one',search)
    ev=r.search(q,['m'])
    assert {e['id']for e in ev}=={'thread','motor'}
    assert {i for e in ev for i in e['retrieval_intents']}=={0,1}


def answer_provider(body):
    client=httpx.Client(transport=httpx.MockTransport(lambda req:httpx.Response(200,json={
        'choices':[{'message':{'content':json.dumps(body)}}]})))
    from app.provider import ManualProvider
    return ManualProvider('https://example.test','example','unchanged-model',client),client


@pytest.mark.parametrize('settings', [[], [{'setting_id': 'model_class', 'value': '-40[]'}]])
def test_subclass_qualifiers_are_not_function_or_dip_settings(settings):
    from app.provider import ProviderError
    source = {'id': 'needle', 'title': 'Fixture', 'page': 1,
              'text': 'Check whether the needle is bent.', 'retrieval_queries': ['Threads are breaking.'],
              'retrieval_intents': [0], 'primary_intents': [0], 'require_claims': True}
    p, client = answer_provider({'answer': 'সুই বাঁকা কি না পরীক্ষা করুন।', 'supported': True,
        'citations': [1], 'diagnosis': 'sufficient' if not settings else 'insufficient', 'claims': [
            {'text': 'সুই বাঁকা কি না পরীক্ষা করুন।', 'citations': [1], 'intent_ids': [0], 'settings': settings}]})
    with client:
        if settings:
            with pytest.raises(ProviderError, match='setting value lacks'):
                p.answer('Threads are breaking. What should I check?', [source])
        else:
            answer, citations = p.answer('Threads are breaking. What should I check?', [source])
            assert 'সুই' in answer and citations == [1]


def test_mechanical_clarification_keeps_both_symptoms_without_unrelated_function_prompts():
    from app.grounding import source_clarification
    queries = ['Threads are breaking.', 'Making a lot of noise.']
    sources = [
        {'id': 'needle', 'title': 'Fixture', 'page': 1, 'text': 'Check whether the needle is bent.',
         'primary_intents': [0], 'retrieval_queries': queries},
        {'id': 'function', 'title': 'Fixture', 'page': 2, 'text': 'Function 39 controls lifter noise.',
         'setting_id': 'function:39', 'primary_intents': [1], 'retrieval_queries': queries},
        {'id': 'noise', 'title': 'Fixture', 'page': 3, 'text': 'Stop if abnormal noises occur.',
         'primary_intents': [1], 'retrieval_queries': queries},
    ]
    answer, citations = source_clarification(sources)
    assert 'needle' in answer and 'abnormal noises' in answer and citations == [1, 3]
    assert 'Function' not in answer


@pytest.mark.parametrize('claim_id', [2, 3])
def test_checked_claim_citations_supply_the_top_level_union(claim_id):
    from app.provider import ProviderError
    sources = [{'id': str(i), 'title': 'Fixture', 'page': i, 'text': 'Check the needle.',
                'retrieval_queries': ['Threads break.'], 'primary_intents': [0], 'require_claims': True}
               for i in (1, 2)]
    p, client = answer_provider({'answer': 'সুই পরীক্ষা করুন।', 'supported': True, 'citations': [1],
        'diagnosis': 'insufficient', 'claims': [{'text': 'সুই পরীক্ষা করুন।', 'citations': [claim_id],
        'intent_ids': [0], 'settings': []}]})
    with client:
        if claim_id == 3:
            with pytest.raises(ProviderError, match='citation'):
                p.answer('Threads break.', sources)
        else:
            answer, citations = p.answer('Threads break.', sources)
            assert 'উপসর্গ 1' in answer and citations == [1, 2]


def diagnostic_sources():
    return [
        {'id':'a','title':'Fixture','page':1,'text':'0: Foot drops at neutral.',
         'setting_id':'function:72','setting_requirements':[{'setting_id':'dip:7','value':'ON'}],
         'retrieval_intents':[0],'retrieval_queries':['Foot drops.','After knee use, treadle will not lift.']},
        {'id':'b','title':'Fixture','page':2,'text':'0: Treadle cannot lift after knee use.',
         'setting_id':'function:83','setting_requirements':[{'setting_id':'dip:7','value':'OFF'}],
         'retrieval_intents':[1],'retrieval_queries':['Foot drops.','After knee use, treadle will not lift.']},
    ]


def test_diagnostic_answer_cannot_omit_second_intent():
    from app.provider import ProviderError
    p,c=answer_provider({'answer':'ফাংশন ৭২ দিন।','supported':True,'citations':[1]})
    with c,pytest.raises(ProviderError,match='diagnostic|intent|claim'):
        p.answer('দুটি সমস্যা কেন?',diagnostic_sources())


def test_contradictory_diagnosis_cannot_be_presented_as_one_configuration():
    from app.provider import ProviderError
    sources=diagnostic_sources()
    claims=[{'text':'প্রথম অবস্থা।','citations':[1],'intent_ids':[0],'settings':[{'setting_id':'function:72','value':'0'}]},
            {'text':'দ্বিতীয় অবস্থা।','citations':[2],'intent_ids':[1],'settings':[{'setting_id':'function:83','value':'0'}]}]
    p,c=answer_provider({'answer':'এই দুই সেটিং একসঙ্গে কারণ।','supported':True,'citations':[1,2],
                        'claims':claims,'diagnosis':'consistent'})
    with c,pytest.raises(ProviderError,match='conflict|incompatible'):
        p.answer('দুটি সমস্যা কেন?',sources)


def test_diagnostic_setting_cannot_use_another_intents_evidence():
    from app.provider import ProviderError
    claims=[{'text':'প্রথম অবস্থা।','citations':[1],'intent_ids':[0],'settings':[{'setting_id':'function:72','value':'0'}]},
            {'text':'দ্বিতীয় অবস্থা।','citations':[1],'intent_ids':[1],'settings':[{'setting_id':'function:72','value':'0'}]}]
    p,c=answer_provider({'answer':'সেটিং বদলান।','supported':True,'citations':[1,2],
                        'claims':claims,'diagnosis':'consistent'})
    with c,pytest.raises(ProviderError,match='intent'):
        p.answer('দুটি সমস্যা কেন?',diagnostic_sources())


def test_declared_incompatible_answer_is_rendered_as_separate_hypotheses():
    claims=[{'text':'প্রথম সম্ভাবনা।','citations':[1],'intent_ids':[0],'settings':[{'setting_id':'function:72','value':'0'}]},
            {'text':'দ্বিতীয় সম্ভাবনা।','citations':[2],'intent_ids':[1],'settings':[{'setting_id':'function:83','value':'0'}]}]
    p,c=answer_provider({'answer':'UNVALIDATED PARALLEL ANSWER','supported':True,'citations':[1,2],
                        'claims':claims,'diagnosis':'incompatible'})
    with c:answer,ids=p.answer('দুটি সমস্যা কেন?',diagnostic_sources())
    assert 'UNVALIDATED'not in answer
    assert 'আলাদা সম্ভাব্য' in answer and 'Function No. 72 = 0'in answer and 'Function No. 83 = 0'in answer
    assert p.last_answer_status=='clarify'


def test_claim_cannot_hide_an_unreported_function_number():
    from app.provider import ProviderError
    claims=[{'text':'Function 99 = 0 দিন।','citations':[1],'intent_ids':[0],'settings':[]},
            {'text':'দ্বিতীয় সম্ভাবনা।','citations':[2],'intent_ids':[1],'settings':[]}]
    p,c=answer_provider({'answer':'সেটিং দিন।','supported':True,'citations':[1,2],
                        'claims':claims,'diagnosis':'consistent'})
    with c,pytest.raises(ProviderError,match='setting|claim'):
        p.answer('দুটি সমস্যা কেন?',diagnostic_sources())


def test_cross_intent_setting_cannot_borrow_an_unrelated_primary_citation():
    from app.provider import ProviderError
    claims=[{'text':'প্রথম অবস্থা।','citations':[1],'intent_ids':[0],'settings':[]},
            {'text':'দ্বিতীয় অবস্থা।','citations':[1,2],'intent_ids':[1],'settings':[{'setting_id':'function:72','value':'0'}]}]
    p,c=answer_provider({'answer':'সেটিং দিন।','supported':True,'citations':[1,2],
                        'claims':claims,'diagnosis':'consistent'})
    with c,pytest.raises(ProviderError,match='intent'):
        p.answer('দুটি সমস্যা কেন?',diagnostic_sources())


def test_two_mode_requirements_inside_one_claim_are_rejected():
    from app.provider import ProviderError
    claim={'text':'একসঙ্গে দিন।','citations':[1,2],'intent_ids':[0,1],
           'settings':[{'setting_id':'function:72','value':'0'},{'setting_id':'function:83','value':'0'}]}
    p,c=answer_provider({'answer':'দিন।','supported':True,'citations':[1,2],
                        'claims':[claim],'diagnosis':'incompatible'})
    with c,pytest.raises(ProviderError,match='incompatible'):
        p.answer('দুটি সমস্যা কেন?',diagnostic_sources())


def test_unsupported_value_is_rejected_even_with_valid_citation():
    from app.provider import ProviderError
    claims=[{'text':'প্রথম সম্ভাবনা।','citations':[1],'intent_ids':[0],'settings':[{'setting_id':'function:72','value':'9'}]},
            {'text':'দ্বিতীয় সম্ভাবনা।','citations':[2],'intent_ids':[1],'settings':[]}]
    p,c=answer_provider({'answer':'দিন।','supported':True,'citations':[1,2],
                        'claims':claims,'diagnosis':'consistent'})
    with c,pytest.raises(ProviderError,match='value'):
        p.answer('দুটি সমস্যা কেন?',diagnostic_sources())


def test_claim_text_cannot_disagree_with_its_checked_setting_value():
    from app.provider import ProviderError
    claims=[{'text':'Function 72 = 9 দিন।','citations':[1],'intent_ids':[0],'settings':[{'setting_id':'function:72','value':'0'}]},
            {'text':'দ্বিতীয় সম্ভাবনা।','citations':[2],'intent_ids':[1],'settings':[]}]
    p,c=answer_provider({'answer':'দিন।','supported':True,'citations':[1,2],
                        'claims':claims,'diagnosis':'consistent'})
    with c,pytest.raises(ProviderError,match='value'):
        p.answer('দুটি সমস্যা কেন?',diagnostic_sources())


def test_cited_warning_is_preserved_even_if_generated_text_omits_it():
    sources=[{'id':'row','title':'Fixture','page':1,'text':'Function setting.',
              'required_evidence_ids':['warning']},
             {'id':'warning','title':'Fixture','page':2,'text':'DANGER: Unplug and wait five minutes before opening the control box.',
              'dependency':'warning'}]
    p,c=answer_provider({'answer':'সেটিং পরীক্ষা করুন।','supported':True,'citations':[1]})
    with c:answer,ids=p.answer('কোন সেটিং?',sources)
    assert 'wait five minutes' in answer and ids==[1,2]


def test_explicit_english_compound_answer_stays_english():
    claims=[{'text':'First possible mode.','citations':[1],'intent_ids':[0],'settings':[{'setting_id':'function:72','value':'0'}]},
            {'text':'Second possible mode.','citations':[2],'intent_ids':[1],'settings':[{'setting_id':'function:83','value':'0'}]}]
    p,c=answer_provider({'answer':'Modes.','supported':True,'citations':[1,2],
                        'claims':claims,'diagnosis':'incompatible'})
    with c:answer,ids=p.answer('Answer in English: explain both symptoms.',diagnostic_sources())
    assert not any('\u0980'<=c<='\u09ff'for c in answer)


def test_one_claim_cannot_combine_a_value_with_its_excluded_dip_state():
    from app.provider import ProviderError
    sources=diagnostic_sources()
    sources[0]['setting_requirements']=[]
    sources[0]['setting_exclusions']=[{'when_value':'0','setting_id':'dip:7','value':'OFF'}]
    sources.append({'id':'dip','title':'Fixture','page':1,'text':'OFF branch.',
                    'setting_id':'dip:7','setting_value':'OFF','retrieval_intents':[0]})
    claims=[{'text':'প্রথম অবস্থা।','citations':[1,3],'intent_ids':[0],
             'settings':[{'setting_id':'function:72','value':'0'},{'setting_id':'dip:7','value':'OFF'}]},
            {'text':'দ্বিতীয় অবস্থা।','citations':[2],'intent_ids':[1],'settings':[]}]
    p,c=answer_provider({'answer':'দিন।','supported':True,'citations':[1,2,3],
                        'claims':claims,'diagnosis':'incompatible'})
    with c,pytest.raises(ProviderError,match='incompatible|excluded'):
        p.answer('দুটি সমস্যা কেন?',sources)


def test_consistent_modes_are_accepted_instead_of_always_refusing():
    sources=diagnostic_sources();sources[1]['setting_requirements'][0]['value']='ON'
    claims=[{'text':'প্রথম সম্ভাবনা।','citations':[1],'intent_ids':[0],'settings':[{'setting_id':'function:72','value':'0'}]},
            {'text':'দ্বিতীয় সম্ভাবনা।','citations':[2],'intent_ids':[1],'settings':[{'setting_id':'function:83','value':'0'}]}]
    p,c=answer_provider({'answer':'ব্যাখ্যা।','supported':True,'citations':[1,2],
                        'claims':claims,'diagnosis':'consistent'})
    with c:answer,ids=p.answer('দুটি সমস্যা কেন?',sources)
    assert p.last_answer_status=='answered'and 'Function No. 72 = 0'in answer and ids==[1,2]


def test_one_uncertain_row_does_not_quarantine_a_complete_sibling(monkeypatch):
    from types import SimpleNamespace
    grid=[['No.','Setting range','Setting details'],['71','0-1','Fan setting'],['72',None,'Motor setting']]
    boxes=[[(40.,80.+i*30,90.,110.+i*30),(90.,80.+i*30,220.,110.+i*30),(220.,80.+i*30,500.,110.+i*30)]for i in range(3)]
    boxes[2][1]=None
    table=SimpleNamespace(extract=lambda:[list(row)for row in grid],col_count=3,bbox=(40,80,500,170),
                          header=SimpleNamespace(names=grid[0],external=False),rows=[SimpleNamespace(cells=b)for b in boxes])
    monkeypatch.setattr(pymupdf.Page,'find_tables',lambda self:SimpleNamespace(tables=[table]))
    pdf=pymupdf.open();p=pdf.new_page();p.insert_text((40,40),'4. FUNCTION SETTINGS')
    page={**_local_page(p),'page':1,'visual_units':[],'visual_review':'unreviewed'}
    units=_assemble({'id':'m','sha256':'a'*64},[page]);pdf.close()
    complete=next(u for u in units if u.get('setting_id')=='function:71')
    uncertain=next(u for u in units if u.get('setting_id')=='function:72')
    assert _searchable(complete)and not complete['blocking_issues']
    assert not _searchable(uncertain)and uncertain['blocking_issues']


def test_translation_cannot_drop_the_event_that_triggers_the_symptom():
    body='S-7200A: The foot drops at neutral and the knee switch blocks the treadle while stopped.'
    client=httpx.Client(transport=httpx.MockTransport(lambda req:httpx.Response(200,json={
        'choices':[{'message':{'content':json.dumps({'search_question':body})}}]})))
    from app.provider import ManualProvider
    question='S-7200A: সুতা কাটার পর নিউট্রালে ফুট নামে। হাঁটু সুইচ ব্যবহার করলে থামানো মেশিনে প্যাডেল কাজ করে না।'
    with client:
        p=ManualProvider('https://example.test','example','unchanged',client)
        assert p.search_query(question)==question
        assert 'event' in p.translation_issue


def test_question_history_and_restatement_stay_with_their_symptom():
    from app.questions import question_parts
    q='সুতা কাটার পর নিউট্রালে প্রেসার ফুট নিচে নামে। আগে সুতা কাটার পর ফুট উপরে থাকতো। হাঁটু সুইচের পরে প্যাডেলে ফুট ওঠে না। প্যাডেল তখন কাজ করে না।'
    parts=question_parts(q)
    assert len(parts)==2 and 'আগে' in parts[0]and 'তখন' in parts[1]


def test_requested_function_is_not_omitted_when_its_definition_is_available():
    from app.provider import ProviderError
    sources=diagnostic_sources()
    claims=[{'text':'প্রথম সম্ভাবনা।','citations':[1],'intent_ids':[0],'settings':[{'setting_id':'function:72','value':'0'}]},
            {'text':'দ্বিতীয় সম্ভাবনা।','citations':[2],'intent_ids':[1],'settings':[]}]
    p,c=answer_provider({'answer':'ব্যাখ্যা।','supported':True,'citations':[1,2],
                        'claims':claims,'diagnosis':'incompatible'})
    with c,pytest.raises(ProviderError,match='function'):
        p.answer('Which function numbers explain both symptoms?',sources)


def test_rejected_diagnosis_becomes_source_backed_clarification_and_is_logged(tmp_path):
    from app.db import Database
    from app.provider import ProviderError
    from app.service import ask
    db=Database(tmp_path/'study.sqlite3')
    with db.connect()as c:c.execute("INSERT INTO manuals VALUES ('m','Fixture','source.pdf',?,2,'ready','test',NULL,'baseline')",('a'*64,))
    sources=diagnostic_sources()
    for s in sources:s.update(manual_id='m',score=.8,primary_intents=s['retrieval_intents'])
    class Retriever:
        def search(self,q,manual_ids=None):return sources
    class Provider:
        validation_issue='Diagnostic claim contradicted its source'
        raw_responses=[{'content':'rejected model answer'}]
        def search_query(self,q):return q
        def answer(self,q,s):raise ProviderError('Invalid grounded-answer format')
    result=ask(db,Retriever(),Provider(),'Which function numbers cause both problems?',['m'])
    assert result['status']=='clarify'
    assert 'rejected model answer'not in result['answer']
    assert 'Foot drops' in result['answer']and 'Treadle cannot lift' in result['answer']
    assert result['error']==Provider.validation_issue
    with db.connect()as c:
        row=c.execute('SELECT status,error FROM queries WHERE id=?',(result['id'],)).fetchone()
    assert tuple(row)==('clarify',Provider.validation_issue)
