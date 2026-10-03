import json

import httpx
import pytest

from app.grounding import validate_diagnostic


def foot_source():
    return {'id':'foot','manual_id':'m','title':'Fixture','page':1,'score':.8,
            'setting_id':'function:89','text':
            'Setting details: Operation after knee switch use\n'
            '0: Presser foot cannot be raised and lowered by depressing the treadle backward\n'
            '1: Above operation is possible\nFootnote: Enabled when DIP switch 7 is OFF.',
            'setting_requirements':[{'setting_id':'dip:7','value':'OFF'}],
            'retrieval_queries':['After knee use, the treadle cannot raise the foot.'],
            'retrieval_intents':[0],'primary_intents':[0],'require_claims':True}


def body(value='0', text='সম্ভাব্য ব্যাখ্যা।', assessment='possible'):
    return {'answer':'উত্তর।','supported':True,'citations':[1],'diagnosis':'consistent',
            'claims':[{'text':text,'citations':[1],'intent_ids':[0],'assessment':assessment,
                       'settings':[{'setting_id':'function:89','value':value}]}]}


def test_branch_rules_resolve_positive_reference_without_changing_quote():
    from app.rules import setting_rules
    rules=setting_rules(foot_source())
    assert [(r['value'],r['effect'],r['enabled'])for r in rules]==[
        ('0','treadle_operation',False),('1','treadle_operation',True)]
    assert rules[1]['source_quote']=='Above operation is possible'
    assert 'cannot'not in rules[1]['description']


def test_unrelated_deictic_branch_remains_unresolved():
    from app.rules import setting_rules
    rules=setting_rules({'setting_id':'function:92','text':'0: Special operation\n1: Above operation is possible'})
    assert all(r['effect']is None for r in rules)


@pytest.mark.parametrize('detail',[
    '0: Sewing pauses when presser foot lifter pedal is on',
    '0: Presser foot can be raised and lowered using the knee switch',
    '0: Needle drops',
])
def test_nearby_foot_or_treadle_words_do_not_invent_an_operator_relationship(detail):
    from app.rules import setting_rules
    rules=setting_rules({'setting_id':'function:91','text':
        'Setting details: Foot controls and treadle status at neutral\n'+detail})
    assert rules[0]['effect']is None


def test_neutral_drop_branches_are_separate():
    from app.rules import setting_rules
    rules=setting_rules({'setting_id':'function:72','text':
        'Setting details: Presser foot status at neutral after thread trimming\n'
        '0: Presser foot drops (See NOTE 2)\n1: Presser foot does not drop'})
    assert [r['enabled']for r in rules]==[True,False]


def test_real_upload_persists_source_rules_without_visual_calls(tmp_path):
    import pymupdf
    from app.db import Database
    from app.ingest import add_pdf,delete_pdf
    from app.structured import process_manual
    pdf=pymupdf.open();page=pdf.new_page()
    page.insert_text((40,60),'Function settings')
    for x in (40,90,170,245,575):page.draw_line((x,90),(x,220))
    for y in (90,120,220):page.draw_line((40,y),(575,y))
    for x,label in [(45,'No.'),(95,'Initial value'),(175,'Setting range'),(250,'Setting details')]:
        page.insert_text((x,110),label,fontsize=9)
    for x,value in [(50,'89'),(100,'0'),(180,'0-1')]:page.insert_text((x,145),value,fontsize=9)
    page.insert_textbox((250,130,570,215),'Operation after knee switch use\n'
        '0: Presser foot cannot be raised and lowered by\ndepressing the treadle backward\n'
        '1: Above operation is possible',fontsize=9)
    content=pdf.tobytes();pdf.close();db=Database(tmp_path/'study.sqlite3')
    manual,added=add_pdf(db,tmp_path,'different-machine-controls.pdf',content);assert added
    first=process_manual(db,tmp_path,manual['id']);reopened=Database(tmp_path/'study.sqlite3')
    rules=next(json.loads(row['payload_json'])['setting_rules']for row in reopened.units(manual['id'])
               if json.loads(row['payload_json']).get('setting_id')=='function:89')
    assert [r['enabled']for r in rules]==[False,True]
    assert first['release_id']==process_manual(reopened,tmp_path,manual['id'])['release_id']
    assert reopened.ingestion_spend()['attempts']==0
    delete_pdf(reopened,tmp_path,manual['id']);assert not reopened.units(manual['id'])


def test_synthetic_effect_text_does_not_contaminate_quote_reconstruction():
    from app.rules import setting_rules
    source=foot_source();source['text']=source['text'].split('\nFootnote:',1)[0]+(
        '\nResolved setting branch 0: The treadle cannot raise/lower the presser foot.')
    assert setting_rules(source)[1]['enabled']is True


@pytest.mark.parametrize('query',[
    'হাঁটু সুইচের পর প্যাডেল দিয়ে ফুট উপরে তোলা যাচ্ছে না।',
    'After knee use the treadle can no longer raise the foot.',
    "After knee use the treadle can't raise the foot.",
    'After knee use the treadle can raise the foot.',
])
def test_bengali_and_positive_english_branch_effects(query):
    source=foot_source();source['retrieval_queries']=[query]
    positive=' can raise 'in query
    answer,_=validate_diagnostic(body('1'if positive else'0'),[source])
    assert ('ওঠানো/নামানো যায়।'if positive else'ওঠানো/নামানো যায় না।')in answer


def test_intermittent_observation_is_not_treated_as_a_boolean_fault():
    from app.rules import observed_effect
    assert observed_effect('After knee use the treadle sometimes cannot raise the foot.','treadle_operation')is None


@pytest.mark.parametrize('phrase',[
    'no longer raises', 'no longer lifts', 'does not raise', 'will not lift',
    'fails to raise', 'stops raising', 'is not able to raise', "isn't able to raise",
    'can no longer raise', 'cannot raise',
])
def test_negative_movement_paraphrases_never_select_a_positive_branch(phrase):
    from app.rules import observed_effect
    from app.diagnosis import source_rule_answer
    query=f'After knee switch use, the treadle {phrase} the presser foot.'
    assert observed_effect(query,'treadle_operation')is False
    source=foot_source();source['retrieval_queries']=[query]
    answer,ids,status=source_rule_answer([source])
    assert status=='answered'and 'Function No. 89 = 0'in answer
    assert 'Function No. 89 = 1'not in answer


def test_unasserted_movement_is_not_assumed_to_be_a_positive_observation():
    from app.rules import observed_effect
    assert observed_effect('After knee switch use, how do I raise the presser foot with the treadle?','treadle_operation')is None


def test_negation_of_a_different_action_does_not_invert_a_positive_movement():
    from app.rules import observed_effect
    assert observed_effect('After knee switch use, I do not mind noise and the treadle raises the foot.','treadle_operation')is True


def test_prerequisite_is_separate_from_symptom_and_source_pages_are_visible():
    source=foot_source();source['required_evidence_ids']=['dip']
    dip={'id':'dip','title':'Fixture','page':2,'text':'Operation: Foot is kept raised.',
         'setting_id':'dip:7','setting_value':'OFF','retrieval_intents':[0],'primary_intents':[]}
    parsed=body();parsed['claims'][0]['settings'].append({'setting_id':'dip:7','value':'OFF'})
    answer,_=validate_diagnostic(parsed,[source,dip])
    assert '**'in answer and '\n- Function No. 89 = 0'in answer
    assert 'DIP switch 7 = OFF:'not in answer
    assert 'কার্যকর হওয়ার শর্ত: DIP switch 7 = OFF'in answer
    assert 'PDF'in answer


def test_safety_output_keeps_instructions_without_duplicate_headers():
    from app.grounding import safety_notes
    sources=[{'title':'Fixture','page':1,'dependency':'warning','text':'Manual section\nNotes on safety\nDANGER',
              'warnings':['Notes on safety DANGER']},
             {'title':'Fixture','page':2,'dependency':'warning','text':'unused duplicated text',
              'warnings':['DANGER\nWait at least 5 minutes after disconnecting power before opening the control box.']},
             {'title':'Fixture','page':2,'dependency':'warning','text':'other duplicate',
              'warnings':['Wait at least 5 minutes after disconnecting power before opening the control box.']}]
    answer=safety_notes(sources)
    assert 'Notes on safety'not in answer and answer.count('Wait at least 5 minutes')==1
    assert 'PDF p. 2'in answer


def test_legal_value_with_opposite_effect_is_rejected():
    with pytest.raises(ValueError,match='effect'):
        validate_diagnostic(body('1','Function 89 = 1 অবস্থায় ফুট তোলা যায় না।'),[foot_source()])


def test_checked_branch_renders_source_meaning_instead_of_unchecked_paraphrase():
    answer,status=validate_diagnostic(body('0','UNVERIFIED — এটি চালু করলে ফুট ওঠে।'),[foot_source()])
    assert 'UNVERIFIED'not in answer and 'ওঠানো/নামানো যায় না'in answer
    assert 'Function No. 89 = 0'in answer and 'DIP switch 7 = OFF'in answer


def test_known_settings_exclude_a_hypothesis_but_allow_explaining_its_rejection():
    source=foot_source();source['known_settings']={'function:89':'1','dip:7':'OFF'}
    with pytest.raises(ValueError,match='observed|current'):
        validate_diagnostic(body(),[source])
    answer,_=validate_diagnostic(body(assessment='ruled_out'),[source])
    assert 'বাদ'in answer


def test_a_function_citation_closes_its_explicit_prerequisite_definition():
    source=foot_source();source['required_evidence_ids']=['dip']
    dip={'id':'dip','title':'Fixture','page':2,'text':'Operation: Foot is kept raised.',
         'setting_id':'dip:7','setting_value':'OFF','retrieval_intents':[0],'primary_intents':[]}
    parsed=body();parsed['claims'][0]['settings'].append({'setting_id':'dip:7','value':'OFF'})
    answer,_=validate_diagnostic(parsed,[source,dip])
    assert parsed['citations']==[1,2]and 'DIP switch 7 = OFF'in answer


def test_refusal_prefixed_translation_falls_back_to_original_question():
    from app.provider import ManualProvider
    with httpx.Client(transport=httpx.MockTransport(lambda request:httpx.Response(200,json={
        'choices':[{'message':{'content':'I cannot answer troubleshooting questions. Here is the translation: Foot drops at neutral.'}}]})))as client:
        provider=ManualProvider('https://example.test','example','same-model',client)
        original='নিউট্রালে প্রেসার ফুট নিচে পড়ছে কেন?'
        assert provider.search_query(original)==original and provider.translation_issue


@pytest.mark.parametrize('translated',[
    'After knee switch use, the treadle raises the presser foot.',
    'After knee switch use, how do I raise the presser foot with the treadle?',
    "I'm ready to translate. Do you want me to translate? After knee switch use the treadle cannot raise the foot.",
])
def test_translation_does_not_turn_a_bengali_negative_into_a_positive_or_unknown(translated):
    from app.provider import ManualProvider
    with httpx.Client(transport=httpx.MockTransport(lambda request:httpx.Response(200,json={
        'choices':[{'message':{'content':json.dumps({'search_question':translated})}}]})))as client:
        provider=ManualProvider('https://example.test','example','same-model',client)
        original='হাঁটু সুইচ ব্যবহারের পরে প্যাডেল দিয়ে প্রেসার ফুট উপরে তোলা যাচ্ছে না।'
        assert provider.search_query(original)==original and provider.translation_issue


def test_actual_request_contains_complete_evidence_and_primary_intent_roles():
    from app.provider import ManualProvider
    source=foot_source();source['required_evidence_ids']=['note']
    note={'id':'note','manual_id':'m','title':'Fixture','page':2,
          'text':'NOTE 7: Only enabled with DIP switch 7 OFF.','dependency':'note',
          'retrieval_intents':[0],'primary_intents':[]}
    sent=[]
    def respond(req):
        sent.append(json.loads(req.content))
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(
            {'answer':'অপর্যাপ্ত তথ্য।','supported':False,'citations':[]})}}]})
    with httpx.Client(transport=httpx.MockTransport(respond))as client:
        ManualProvider('https://example.test','example','same-model',client).answer('সমস্যা কেন?',[source,note])
    packet=sent[0]['messages'][1]['content']
    assert source['text']in packet and note['text']in packet
    assert 'Primary evidence'in packet and 'Setting branch rules'in packet and 'treadle_operation'in packet


def test_new_experiment_request_cap_keeps_cumulative_cost_guard(tmp_path,monkeypatch):
    from app.config import Settings
    from app.provider import ProviderError
    from evaluation.budget import RunBudget
    monkeypatch.setattr(RunBudget,'usage',lambda self:{'balance':100,'actual_cost':0})
    budget=RunBudget(Settings(runtime=tmp_path),limit=.02,max_attempts=1)
    small={'messages':[{'content':'small'}],'max_tokens':1}
    # Cost, not a larger request cap, prevents an oversized second reservation.
    budget.max_attempts=3
    with pytest.raises(ProviderError,match='allowance'):
        budget.reserve({'messages':[{'content':'x'*10000}],'max_tokens':1000})
    budget.max_attempts=1;budget.reserve(small)
    with pytest.raises(ProviderError,match='request limit'):
        budget.reserve(small)


def test_diagnostic_followup_keeps_question_scope_and_rejects_changed_manual(tmp_path,monkeypatch):
    from app.db import Database
    from app.service import ask,continue_diagnosis
    monkeypatch.setattr('app.service.source_rule_answer',lambda *args:None)
    db=Database(tmp_path/'study.sqlite3')
    with db.connect()as c:
        c.execute("INSERT INTO manuals VALUES ('m','Fixture','source.pdf',?,2,'ready','test','r1','structured')",('a'*64,))
    class Retriever:
        release_id='index-r1'
        def search(self,q,manual_ids=None):
            self.scope=manual_ids;return [foot_source()]
    class Provider:
        last_answer_status='clarify'
        def search_query(self,q):self.question=q;return q
        def answer(self,q,s):self.sources=s;return 'বর্তমান মান জানান।',[1]
    retriever=Retriever();provider=Provider()
    first=ask(db,retriever,provider,'After knee use, the treadle cannot raise the foot.',['m'])
    second=continue_diagnosis(db,retriever,provider,first['id'],'Function 89 = 1; DIP switch 7 = OFF')
    assert first['diagnostic_context']['root_question']in provider.question
    assert retriever.scope==['m']and provider.sources[0]['known_settings']=={'function:89':'1','dip:7':'OFF'}
    third=continue_diagnosis(db,retriever,provider,second['id'],'The foot now raises before knee switch use.')
    assert 'The foot now raises before knee switch use.'in provider.question
    assert third['diagnostic_context']['free_observations']is True
    assert provider.sources[0]['unstructured_observations']is True
    with db.connect()as c:
        stored=json.loads(c.execute('SELECT diagnostic_context_json FROM queries WHERE id=?',(second['id'],)).fetchone()[0])
        assert stored['parent_query_id']==first['id']
        c.execute("UPDATE manuals SET knowledge_release='r2'WHERE id='m'")
    with pytest.raises(ValueError,match='changed|updated'):
        continue_diagnosis(db,retriever,provider,second['id'],'It is still stopped.')


def test_source_solver_explains_an_unrelated_function_number_and_excludes_known_mismatch():
    from app.diagnosis import source_rule_answer
    source=foot_source()
    answer,ids,status=source_rule_answer([source])
    assert status=='answered'and ids==[1]and 'Function No. 89 = 0'in answer
    source['known_settings']={'function:89':'1','dip:7':'OFF'}
    answer,ids,status=source_rule_answer([source])
    assert status=='clarify'and 'বাদ'in answer and 'আগে, পরে'in answer


def test_source_solver_declines_unresolved_operator_effects():
    from app.diagnosis import source_rule_answer
    source=foot_source();source['text']='0: A special operation is active\n1: Above operation is possible'
    assert source_rule_answer([source])is None
    source=foot_source();source['unstructured_observations']=True
    assert source_rule_answer([source])is None


def test_a_numbered_exception_supplies_the_effect_of_the_same_value_in_another_mode():
    from app.diagnosis import source_rule_answer
    source={'id':'row','title':'Fixture','page':1,'setting_id':'function:72',
        'text':'Setting details: Presser foot status at neutral after thread trimming\n'
               '0: Presser foot drops\n1: Presser foot does not drop\n'
               'Footnote: (NOTE 2) Presser foot will not drop if DIP switch 7 is OFF.',
        'setting_exclusions':[{'when_value':'0','setting_id':'dip:7','value':'OFF'}],
        'retrieval_queries':['After thread trimming the foot does not drop at neutral.'],
        'primary_intents':[0],'retrieval_intents':[0],'require_claims':True,
        'known_settings':{'function:72':'0','dip:7':'OFF'}}
    dip={'id':'dip','title':'Fixture','page':1,'text':'Operation: Presser foot is kept raised.',
         'setting_id':'dip:7','setting_value':'OFF','primary_intents':[0],'retrieval_intents':[0]}
    answer,ids,status=source_rule_answer([source,dip])
    assert status=='answered'and 'Function No. 72 = 0'in answer and 'নিচে নামে না'in answer
