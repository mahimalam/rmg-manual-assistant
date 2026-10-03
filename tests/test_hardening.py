import pytest


@pytest.mark.parametrize('text,expected', [
    ('মেশিনের সেটিং কী হবে।', False),
    ('বাংলা প্রশ্ন॥', False),
    ('S-7200A Function 35 = 0', False),
    ('প্রশ্নে ३५ লেখা আছে।', False),
    ('मशीन सेटिंग', True),
])
def test_script_guard_distinguishes_shared_punctuation_from_letters(text, expected):
    from app.language import contains_devanagari_letters
    assert contains_devanagari_letters(text) is expected


def test_engineering_identifiers_normalize_unicode_digits_and_hyphens():
    from app.language import normalize_question
    assert normalize_question('S‑৭২০০A–৪৫৩ Function ৩৫ = ০।') == 'S-7200A-453 Function 35 = 0।'


def test_note_continuation_keeps_a_second_sentence():
    from app.semantics import note_blocks
    lines = [(20, 100, 400, 110, '(NOTE 7) The response is fastest at 10.'),
             (95, 113, 400, 123, 'Only enabled when Function No. 11 is set to "2".'),
             (20, 140, 400, 150, '(NOTE 8) A different note.')]
    assert 'Function No. 11' in note_blocks(lines)[0][4]


def test_star_footnote_is_linked_even_below_another_section():
    from app.knowledge import UnitDraft
    from app.structured import _assemble
    drafts = [UnitDraft(kind='table', title='1-1 Head specifications', section_key='1-1',
                        text='Oil: Oil X *2', table_headers=['Item','Specification'],
                        table_rows=[['Oil', 'Oil X *2']]),
              UnitDraft(kind='information', title='1-2 Control box', section_key='1-2',
                        text='*2 Only for the dry model.')]
    page = {'page':1,'printed_label':'1','method':'text','text':'',
            'units':[u.model_dump()for u in drafts],'visual_units':[],'visual_review':'unreviewed'}
    units = _assemble({'id':'m','sha256':'a'*64}, [page])
    assert any(e['id']==units[1]['id'] and e['relation']=='note'for e in units[0]['dependencies'])


def test_continued_page_inherits_the_previous_section_for_notes():
    from app.knowledge import UnitDraft
    from app.structured import _assemble
    drafts = [UnitDraft(kind='information',title='4. Settings',section_key='4',text='See NOTE 7.'),
              UnitDraft(kind='information',title='Manual section',text='(NOTE 7) Use mode 2.')]
    pages = [{'page':n,'printed_label':str(n),'method':'text','text':'',
              'units':[u.model_dump()],'visual_units':[],'visual_review':'unreviewed'}
             for n,u in enumerate(drafts,1)]
    units = _assemble({'id':'m','sha256':'a'*64}, pages)
    assert not units[0]['blocking_issues']
    assert any(e['id']==units[1]['id']for e in units[0]['dependencies'])


def test_truncated_answer_envelope_is_rejected_even_when_json_parses():
    import httpx,json
    from app.provider import ManualProvider,ProviderError
    def response(request):
        return httpx.Response(200,json={'choices':[{'finish_reason':'length','message':{'content':
            json.dumps({'answer':'মান ২ দিন।','supported':True,'citations':[1]})}}]})
    with httpx.Client(transport=httpx.MockTransport(response))as client:
        with pytest.raises(ProviderError,match='Truncated'):
            ManualProvider('https://example.test','example','model',client).answer('?',[
                {'title':'Fixture','page':1,'text':'Mode 2 only after disabling power.'}])


def test_duplicate_response_keys_are_rejected():
    import json
    from app.provider import _answer_json
    with pytest.raises(ValueError,match='Duplicate'):
        _answer_json('{"answer":"না","supported":false,"supported":true,"citations":[1]}')


def test_bengali_request_about_an_english_manual_still_requires_bengali():
    import httpx,json
    from app.provider import ManualProvider
    def response(request):
        payload=json.loads(request.content)
        assert 'Answer in Bengali' in payload['messages'][1]['content']
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps({
            'answer':'মান ২ দিন।','supported':True,'citations':[1]})}}]})
    with httpx.Client(transport=httpx.MockTransport(response))as client:
        ManualProvider('https://example.test','example','model',client).answer(
            'ইংরেজিতে লেখা ম্যানুয়াল অনুযায়ী বাংলায় উত্তর দিন।',[
                {'title':'Fixture','page':1,'text':'Mode 2.'}])


def test_translation_drift_preserves_the_original_model_and_numbers():
    import json
    import httpx
    from app.provider import ManualProvider
    def response(request):
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps({'search_question':'S-7300A Function 35 = 0'})}}]})
    with httpx.Client(transport=httpx.MockTransport(response))as client:
        result=ManualProvider('https://example.test','example','model',client).search_query(
            'S‑৭২০০A মেশিনে ফাংশন ৩৫ কি ০ না ১?')
    assert 'S-7200A' in result and '1' in result


@pytest.mark.parametrize('question', ['What should I set on my S-9900A?', 'What presser foot setting should I use?'])
def test_unknown_or_unspecified_machine_cannot_silently_use_the_top_hit(tmp_path, question):
    from app.db import Database
    from app.service import ask
    db=Database(tmp_path/'study.sqlite3')
    with db.connect()as c:
        for n in (7200,7300):
            c.execute('INSERT INTO manuals (id,title,filename,sha256,pages,status,created_at) VALUES (?,?,?,?,?,?,?)',
                      (str(n),f'S-{n}A',f'm{n}.pdf',('a' if n==7200 else 'b')*64,1,'ready','test'))
    class Retriever:
        def search(self, question, manual_ids=None):
            return [{'id':'7200:1','manual_id':'7200','title':'S-7200A','page':1,'score':.8,'text':'Setting 2.'}]
    class Provider:
        def search_query(self, question):return question
        def answer(self, question, sources):raise AssertionError('Ambiguous/unknown machine reached generation')
    result=ask(db,Retriever(),Provider(),question)
    assert result['status']=='clarify'


def test_asterisk_before_a_dimension_is_not_a_numbered_footnote():
    from app.semantics import note_numbers
    assert note_numbers('Maximum 7 mm (* 5 mm at time of shipment)') == []


def test_visual_table_mislabeled_information_still_indexes_its_cells():
    import json
    from app.structured import _validated_visual, _assemble
    response = {'content': json.dumps({'units': [{'kind':'information', 'title':'Specifications',
        'text':'Specifications table', 'table_headers':['Model','Grease'],
        'table_rows':[['X','Part 123 *2'],['Y','Part 456']],
        'footnotes':['*2 Only for dry models.']}]}), 'finish_reason':'stop'}
    local = {'width':600,'height':800}
    visual = _validated_visual(response, local)
    page = {'page':1,'printed_label':'1','method':'text','text':'', 'units':[],
            'visual_units':visual,'visual_review':'unreviewed'}
    units = _assemble({'id':'m','sha256':'a'*64},[page])
    row = next(u for u in units if u['parent_id'] and 'Part 123' in u['text'])
    assert row['kind']=='table' and 'Part 123' in row['search_text']
    assert not row['blocking_issues']
    assert any(e['relation']=='note' for e in row['dependencies'])


def test_automatic_enrichment_reuses_success_and_does_not_rebill_failure(tmp_path):
    from tests.test_structured import setup_manual, FakeVLM
    from app.structured import process_manual, enrich_recommended, BudgetPolicy
    db, manual_id = setup_manual(tmp_path)
    process_manual(db,tmp_path,manual_id)
    policy=BudgetPolicy(10,1,5,3)
    provider=FakeVLM()
    first=enrich_recommended(db,tmp_path,manual_id,provider,policy)
    assert first['selected_pages']==[1] and provider.calls==1
    enrich_recommended(db,tmp_path,manual_id,provider,policy)
    assert provider.calls==1
    other=tmp_path/'other';db2,mid2=setup_manual(other)
    process_manual(db2,other,mid2)
    invalid=FakeVLM(invalid=True)
    failed=enrich_recommended(db2,other,mid2,invalid,policy)
    assert failed['error'] and invalid.calls==1
    enrich_recommended(db2,other,mid2,invalid,policy)
    assert invalid.calls==1
    assert db2.manuals()[0]['status']=='ready'


def test_failed_answer_is_durable_and_feedbackable(tmp_path):
    import json
    from tests.test_structured import setup_manual
    from app.service import ask,save_feedback,export_feedback
    from app.provider import ProviderError
    db,mid=setup_manual(tmp_path)
    class Retriever:
        release_id='fixture'
        def search(self,question,manual_ids=None):
            return [{'id':'u','manual_id':mid,'page':1,'title':'Fixture','text':'Source','score':.9}]
    class Provider:
        raw_responses=[]
        usage_log=[]
        def search_query(self,q):return q
        def answer(self,q,s):
            self.raw_responses.append({'content':'{bad'})
            raise ProviderError('Malformed grounded response')
    result=ask(db,Retriever(),Provider(),'What do the oil marks mean?',[mid])
    assert result['status']=='provider_error' and not result['sources']
    with db.connect()as c:
        row=c.execute('SELECT * FROM queries WHERE id=?',(result['id'],)).fetchone()
        assert 'Malformed' in row['error']
        assert json.loads(row['provider_responses_json'])[0]['content']=='{bad'
    save_feedback(db,result['id'],'incorrect','Failed response')
    assert b'Malformed' in export_feedback(db)


def test_short_wrapped_condition_remains_one_paragraph():
    from app.semantics import merge_text_blocks
    blocks=[(100,20,350,30,'Be sure to mount the machine head',0,0),
            (100,31,350,41,'support rod so height becomes 63 to 68 mm. For the sewing',1,0),
            (100,42,350,52,'machine with AK, use 33 to 38 mm.',2,0)]
    merged=merge_text_blocks(blocks)
    assert len(merged)==1 and '63 to 68' in merged[0][4] and '33 to 38' in merged[0][4]


def test_visual_table_without_summary_retains_cells_without_rebilling():
    import json
    from app.structured import _validated_visual
    raw={'content':json.dumps({'units':[{'kind':'table','title':'Gauge',
         'table_headers':['Model','Gauge'],'table_rows':[['X','2 mm']]}]})}
    units=_validated_visual(raw,{'width':600,'height':800})
    assert units[0]['table_rows']==[['X','2 mm']] and units[0]['text']


def test_related_subsection_context_keeps_a_second_requested_constraint():
    import json
    from app.knowledge import UnitDraft
    from app.structured import _assemble
    from app.retrieval import Retriever
    drafts=[UnitDraft(kind='procedure',title='3-2-1 Oil check',section_key='3-2-1',text='1) Check for five seconds.'),
            UnitDraft(kind='procedure',title='3-2-2 Appropriate oil',section_key='3-2-2',text='1) Repeat on three sheets of paper.')]
    page={'page':2,'printed_label':'2','method':'text','text':'', 'units':[u.model_dump()for u in drafts],
          'visual_units':[],'visual_review':'unreviewed'}
    units=_assemble({'id':'m','sha256':'a'*64},[page])
    class DB:
        def units(self):return [{'id':u['id'],'payload_json':json.dumps(u)}for u in units]
    r=Retriever.__new__(Retriever);r.db=DB();r.release_id='test'
    lookup={u['id']:{'id':u['id'],'manual_id':'m','page':2,'title':'Fixture','text':u['search_text']}for u in units}
    source={**lookup[units[0]['id']],'score':.9}
    result=r._expand([source],lookup)
    assert any('three sheets' in u['text']for u in result)


def test_empty_scope_does_not_search_or_charge_for_translation(tmp_path):
    from tests.test_structured import setup_manual
    from app.service import ask
    db,mid=setup_manual(tmp_path)
    class Provider:
        def search_query(self,q):raise AssertionError('Empty scope made a paid call')
    result=ask(db,None,Provider(),'বাংলা প্রশ্ন।',[])
    assert result['status']=='no_evidence'


def test_hardening_budget_reserves_failed_calls_and_caps_total(tmp_path,monkeypatch):
    import json
    from app.config import Settings
    from app.provider import ProviderError
    from evaluation.budget import RunBudget
    monkeypatch.setattr(RunBudget,'usage',lambda self:{'balance':100,'actual_cost':0})
    b=RunBudget(Settings(runtime=tmp_path),limit=.01)
    payload={'messages':[{'content':'a'}],'max_tokens':1}
    with pytest.raises(ProviderError,match='allowance'):b.reserve(payload)
    assert json.loads(b.path.read_text())['attempts']==[]
    b.path.unlink();b=RunBudget(Settings(runtime=tmp_path),limit=.03)
    b.reserve(payload)
    b.reserve(payload)
    with pytest.raises(ProviderError):b.reserve(payload)
    assert all(a['status']=='pending' for a in json.loads(b.path.read_text())['attempts'])


def test_visual_missing_summary_uses_only_returned_exact_quote():
    import json
    from app.structured import _validated_visual
    raw={'content':json.dumps({'units':[{'kind':'procedure','title':'Check',
         'source_quote':'Disconnect power first.','actions':['Disconnect power first.']} ]})}
    units=_validated_visual(raw,{'width':600,'height':800})
    assert units[0]['text']=='Disconnect power first.'


def test_visual_table_footnotes_attach_only_to_marked_rows_and_supersede_ambiguous_cells():
    from app.knowledge import UnitDraft
    from app.structured import _assemble,_searchable
    table=UnitDraft(kind='table',title='Specifications',text='Specs',table_headers=['Model X','Model Y'],
                    table_rows=[['Oil #18','Oil #18'],['Grease *2','Grease *2']],footnotes=['*2 Only for model Y.'])
    local=UnitDraft(kind='table',title='Native row',text='Oil value uncertain',table_headers=['Model X','Model Y'],
                    table_rows=[['Oil #18',None]],uncertainties=['Merged/blank cells or headers require visual source review'])
    page={'page':1,'printed_label':'1','method':'text','text':'','units':[local.model_dump()],
          'visual_units':[table.model_dump()],'visual_review':'unreviewed'}
    units=_assemble({'id':'m','sha256':'a'*64},[page])
    oil=next(u for u in units if u['parent_id'] and 'Oil #18' in u['text'])
    grease=next(u for u in units if u['parent_id'] and 'Grease' in u['text'])
    assert not oil['footnotes'] and grease['footnotes']==['*2 Only for model Y.']
    assert not _searchable(units[0]) and _searchable(oil)


@pytest.mark.parametrize('text', ['', 'मशीन', 'იიიი', '\ufffd', '1234'])
def test_failed_voice_recognition_cannot_feed_junk_to_the_answer_pipeline(monkeypatch,text):
    from types import SimpleNamespace
    import app.voice as voice
    class Whisper:
        def transcribe(self,audio,**kwargs):
            return [SimpleNamespace(text=text)],None
    monkeypatch.setattr(voice,'_whisper',lambda model:Whisper())
    with pytest.raises(ValueError):voice.transcribe(b'synthetic fixture')


def test_bengali_voice_recognition_preserves_latin_machine_names(monkeypatch):
    from types import SimpleNamespace
    import app.voice as voice
    class Whisper:
        def transcribe(self,audio,**kwargs):return [SimpleNamespace(text='S-7200A মেশিনের প্রশ্ন।')],None
    monkeypatch.setattr(voice,'_whisper',lambda model:Whisper())
    assert voice.transcribe(b'synthetic fixture')=='S-7200A মেশিনের প্রশ্ন।'


def test_table_subclass_headers_establish_scope():
    from app.knowledge import UnitDraft
    from app.structured import _assemble
    draft=UnitDraft(kind='table',title='Specification',text='Pitch over 4: 4000',
                    table_headers=['-333P,-433P, -303P, -403P','-305P -405P'],table_rows=[['4000','4000']])
    page={'page':1,'printed_label':'1','method':'text','text':'','units':[draft.model_dump()],
          'visual_units':[],'visual_review':'unreviewed'}
    unit=_assemble({'id':'m','sha256':'a'*64},[page])[0]
    assert '-405P' in unit['applicable_models'] and '-333P' in unit['applicable_models']


@pytest.mark.parametrize('value,low,high', [('4',True,False),('4.5',False,True),('3.9',True,False)])
def test_pitch_threshold_keeps_open_and_closed_boundaries(value,low,high):
    from app.conditions import pitch_condition
    q=f'What speed at pitch {value} mm?'
    assert pitch_condition(q,'Pitch 4 or less /5000 sti/min') is low
    assert pitch_condition(q,'More than pitch 4/4000 sti/min') is high


def test_pitch_condition_never_converts_other_units_or_filters_unknown_conditions():
    from app.conditions import pitch_condition
    assert pitch_condition('pitch 4 inches','More than pitch 4') is None
    assert pitch_condition('pitch 4 mm','Speed depends on material') is None
    assert pitch_condition('pitch 4 mm or 5 mm','More than pitch 4') is None
    assert pitch_condition('pitch 4 mm','More than pitch 4 cm') is None
    assert pitch_condition('pitch 4 mm','Pitch 4 or more') is True
    assert pitch_condition('pitch 4 mm','Less than pitch 4') is False


def test_cache_reuse_across_local_parser_upgrade_does_not_repeat_image(tmp_path):
    import json,pymupdf
    from tests.test_structured import setup_manual,FakeVLM,manual_pdf
    from app.ingest import add_pdf
    from app.structured import process_manual,BudgetPolicy,_reuse_replacement_cache
    db,mid=setup_manual(tmp_path);p=FakeVLM();budget=BudgetPolicy(10,1,5,3)
    process_manual(db,tmp_path,mid,p,budget,[1])
    raw_path=next((tmp_path/'knowledge'/mid/'visual').glob('*.json'))
    raw=json.loads(raw_path.read_text());raw['extraction_config']['parser']='older-local-parser'
    raw_path.write_text(json.dumps(raw))
    with pymupdf.open(tmp_path/'manuals'/f'{mid}.pdf')as doc:
        doc.set_metadata({'author':'New revision'});pdf=doc.tobytes()
    new,_=add_pdf(db,tmp_path,'synthetic.pdf',pdf,publish_baseline=False)
    old=next(dict(m)for m in db.manuals()if m['id']==mid)
    current=next(dict(m)for m in db.manuals()if m['id']==new['id'])
    _reuse_replacement_cache(tmp_path,old,current)
    process_manual(db,tmp_path,new['id'],p,budget,[1])
    assert p.calls==1


def test_native_corroboration_requires_header_alignment_and_exact_decimal_values():
    from app.knowledge import UnitDraft
    from app.structured import _native_row_support
    native=UnitDraft(kind='table',title='Specs',text='Specs',table_headers=['X model','Y model'],
                     table_rows=[['0.5 mm','1.5 mm']]).model_dump()
    row=UnitDraft(kind='table',title='Specs',text='Specs',table_headers=['X model','Y model'],
                  table_rows=[['0.5 mm','1.5 mm']])
    assert _native_row_support(row,[native])
    row.table_rows=[['1.5 mm','0.5 mm']]
    assert not _native_row_support(row,[native])
    row.table_rows=[['5.0 mm','1.5 mm']]
    assert not _native_row_support(row,[native])


def test_visual_duplicate_fields_cannot_override_extracted_evidence():
    from app.structured import _validated_visual
    raw={'content':'{"units":[{"kind":"information","title":"X","text":"Correct","text":"Changed"}]}'}
    with pytest.raises(ValueError,match='Duplicate'):_validated_visual(raw,{'width':600,'height':800})


def test_empty_manual_scope_cannot_be_bypassed_by_page_image(tmp_path):
    from tests.test_structured import setup_manual
    from app.service import ask
    db,mid=setup_manual(tmp_path)
    class Provider:
        def search_query(self,q):raise AssertionError('Invalid scope made a paid translation')
        def answer(self,q,s):raise AssertionError('Invalid scope sent an image')
    result=ask(db,None,Provider(),'What is on this page?',[],page_hint=(mid,1),runtime=tmp_path)
    assert result['status']=='pipeline_error' and not result['sources']


def test_page_question_retrieves_only_its_selected_manual(tmp_path):
    from tests.test_structured import setup_manual
    from app.service import ask
    db,mid=setup_manual(tmp_path)
    class Retriever:
        def search(self,q,manual_ids=None):
            assert manual_ids==[mid]
            return []
    class Provider:
        def search_query(self,q):return q
        def answer(self,q,s):return 'নির্বাচিত পৃষ্ঠা।',[1]
    result=ask(db,Retriever(),Provider(),'What is on this page?',page_hint=(mid,1),runtime=tmp_path)
    assert result['status']=='answered'


def test_first_model_download_publishes_the_resolved_revision(tmp_path,monkeypatch):
    from tests.test_structured import setup_manual
    import app.retrieval as retrieval
    db,mid=setup_manual(tmp_path)
    state={'downloaded':False}
    class Encoder:
        max_seq_length=4096
        def tokenizer(self,text,**kwargs):return {'input_ids':[1]}
        def encode(self,texts,**kwargs):return [[1.,0.]for text in texts]
    def load(name):
        state['downloaded']=True
        return Encoder()
    monkeypatch.setattr(retrieval,'encoder',load)
    monkeypatch.setattr(retrieval,'model_signature',lambda name:'model@revision' if state['downloaded'] else 'model@unresolved')
    first=retrieval.Retriever(db,tmp_path,'model');first.sync()
    reopened=retrieval.Retriever(db,tmp_path,'model')
    assert not reopened.model_mismatch


def test_numeric_parameter_phrases_are_not_machine_identifiers():
    from app.language import model_ids
    assert model_ids('On S-7200A at 4500RPM, speed 5000 sti/min and 220 to 240V?')==['S7200A']
    assert model_ids('Model AT-1000')==['AT1000']


def test_error_and_parameter_codes_do_not_override_the_machine_scope():
    from app.language import model_ids
    assert model_ids('On DDL-9000C-SMS, what does error E221 mean?')==['DDL9000CSMS']
    assert model_ids('On S-7200A, what is Function No 118?')==['S7200A']
    assert model_ids('Model E-221')==['E221']
    assert model_ids('DDL-9000C-SMS-এ E221 ত্রুটি কী বোঝায়?')==['DDL9000CSMS']
    assert model_ids('On DDL-9000C-SMS, what does K118 error resetting mean?')==['DDL9000CSMS']
