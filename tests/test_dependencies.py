import json
from pathlib import Path

import pymupdf
import pytest

from app.structured import _assemble, _local_page


@pytest.fixture(scope="module")
def brother_units():
    path = Path(__file__).resolve().parents[1] / "S-7200A Service Manuel.pdf"
    if not path.is_file():
        pytest.skip("External S-7200A service manual is not distributed with this repository")
    with pymupdf.open(path) as document:
        pages = [{**_local_page(document[number - 1]), "page": number,
                  "visual_units": [], "visual_review": "unreviewed"} for number in (15, 38)]
    return _assemble({"id": "brother", "sha256": "a" * 64}, pages)


def closure(unit, units):
    lookup = {item["id"]: item for item in units}
    seen, queue = {}, [unit]
    while queue:
        item = queue.pop()
        if item["id"] in seen:
            continue
        seen[item["id"]] = item
        queue.extend(lookup[dependency["id"]] for dependency in item["dependencies"])
    return list(seen.values())


def test_function_row_requires_note_and_prerequisite_setting(brother_units):
    row = next(unit for unit in brother_units if unit.get("setting_id") == "function:39")
    linked = closure(row, brother_units)
    assert any("response is fastest" in unit["text"] for unit in linked)
    assert any(unit.get("setting_id") == "function:35" for unit in linked)
    assert {("function:35", "0")} <= {(item["setting_id"], item["value"])
                                      for unit in linked for item in unit["setting_requirements"]}
    assert "operating noise is greatest" in row["search_text"]


def test_exclusion_is_complete_and_follows_procedure(brother_units):
    procedure = next(unit for unit in brother_units if "8 seconds" in unit["text"])
    linked = closure(procedure, brother_units)
    rule = next(unit for unit in linked if "-45?" in unit["excluded_models"])
    assert "fully dry-type" in rule["text"] and "amount is not necessary" in rule["text"]
    assert procedure["applicable_models"] == ["-40?", "-43?"]


@pytest.mark.parametrize("pattern,subclass,expected", [
    ("-45[]", "-453", True), ("-45□", "-455", True),
    ("-4[]S", "-43S", True), ("-45?", "-403", False),
    ("-45?", "-4530", False), ("-45?", "-45", False),
])
def test_subclass_wildcards_match_one_character(pattern, subclass, expected):
    from app.semantics import matches_model
    assert matches_model(pattern, subclass) is expected


def test_required_dependencies_survive_primary_candidate_limit(brother_units, tmp_path):
    from app.db import Database
    from app.retrieval import Retriever
    from app.structured import _searchable
    db = Database(tmp_path / "study.sqlite3")
    with db.connect() as connection:
        connection.execute("INSERT INTO manuals (id,title,filename,sha256,pages,status,created_at) "
                           "VALUES ('brother','S-7200A','source.pdf',?,67,'ready','test')", ("a" * 64,))
        for unit in brother_units:
            connection.execute("INSERT INTO knowledge_units VALUES (?,?,?,?,?,?,?)",
                               (unit["id"], "brother", unit["page"], unit["kind"], unit["search_text"],
                                json.dumps(unit), "fixture"))
    lookup = {unit["id"]: {"id": unit["id"], "manual_id": "brother", "page": unit["page"],
                           "title": "S-7200A", "text": unit["search_text"]}
              for unit in brother_units if _searchable(unit)}
    row = next(unit for unit in brother_units if unit.get("setting_id") == "function:39")
    engine = Retriever(db, tmp_path, "synthetic-model")
    evidence = engine._expand([{**lookup[row["id"]], "score": 0.5}], lookup)
    assert any("operating noise is greatest" in item["text"] for item in evidence)
    assert any(item.get("setting_id") == "function:35" for item in evidence)


def test_indented_note_continuation_is_retained(brother_units):
    from app.semantics import note_blocks
    lines = [(20, 100, 500, 110, '(NOTE 7) Output is enabled when the actuator'),
             (100, 113, 480, 123, 'is used. The output comes from connector CN2.'),
             (20, 140, 480, 150, '(NOTE 8) Only enabled when Function No. 11 is set to "2".')]
    notes = note_blocks(lines)
    assert len(notes) == 2
    assert 'connector CN2.' in notes[0][4]
    assert notes[0][3] == 123


def test_printed_footer_labels_are_retained(brother_units):
    assert {unit['printed_page_label'] for unit in brother_units if unit['page'] == 15} == {'9'}
    assert {unit['printed_page_label'] for unit in brother_units if unit['page'] == 38} == {'32'}


def test_setting_prerequisite_selects_only_the_matching_note_branch(tmp_path):
    from app.db import Database
    from app.retrieval import Retriever
    db = Database(tmp_path / 'study.sqlite3')
    units = [
        {'id': 'owner', 'manual_id': 'm', 'kind': 'information', 'page': 1,
         'search_text': 'Function 47 needs Function 11=2.', 'dependencies': [
             {'id': 'setting', 'relation': 'prerequisite', 'required_value': '2'}]},
        {'id': 'setting', 'manual_id': 'm', 'kind': 'information', 'page': 1,
         'search_text': 'Function 11 modes', 'dependencies': [
             {'id': 'matching', 'relation': 'note', 'when_value': '2'},
             {'id': 'opposite', 'relation': 'note', 'when_value': '1'}]},
        {'id': 'matching', 'manual_id': 'm', 'kind': 'information', 'page': 2,
         'search_text': 'NOTE for mode 2', 'dependencies': []},
        {'id': 'opposite', 'manual_id': 'm', 'kind': 'information', 'page': 2,
         'search_text': 'NOTE for mode 1', 'dependencies': []},
    ]
    with db.connect() as connection:
        connection.execute("INSERT INTO manuals (id,title,filename,sha256,pages,status,created_at) "
                           "VALUES ('m','Fixture','source.pdf',?,2,'ready','test')", ('a' * 64,))
        for unit in units:
            connection.execute('INSERT INTO knowledge_units VALUES (?,?,?,?,?,?,?)',
                               (unit['id'], 'm', unit['page'], unit['kind'], unit['search_text'],
                                json.dumps(unit), 'fixture'))
    lookup = {unit['id']: {'id': unit['id'], 'manual_id': 'm', 'title': 'Fixture',
                          'page': unit['page'], 'text': unit['search_text']} for unit in units}
    evidence = Retriever(db, tmp_path, 'synthetic-model')._expand([lookup['owner']], lookup)
    assert {source['id'] for source in evidence} == {'owner', 'setting', 'matching'}


def test_literal_subclasses_and_opposite_exceptions_are_distinct():
    from app.semantics import model_patterns, link_semantics
    from app.knowledge import KnowledgeUnit, UnitDraft
    assert model_patterns('If the machine is sub-class -453, adjustment is not necessary.') == ['-453']
    draft = UnitDraft(kind='information', title='Maintenance',
                      text='If the machine is sub-class -403, adjust the oil. If the machine is sub-class -453, adjustment is not necessary.')
    unit = KnowledgeUnit(**draft.model_dump(), id='m:1:one', manual_id='m', manual_sha256='a'*64,
                         page=1, printed_page_label='', parent_id=None,
                         source_refs=[{'manual_id':'m','pdf_page_index':0,'printed_page_label':'','bbox':None}],
                         search_text=draft.text, extraction_method='text',review_status='unreviewed',
                         flags=[],blocking_issues=[],dependencies=[]).model_dump()
    linked = link_semantics([unit])[0]
    assert linked['excluded_models'] == ['-453']


def test_generated_answer_cannot_drop_cited_setting_prerequisite():
    import httpx
    from app.provider import ManualProvider
    def response(request):
        return httpx.Response(200, json={'choices':[{'message':{'content':json.dumps({
            'answer':'সেটিং পরিবর্তন করুন।','supported':True,'citations':[1]})}}]})
    source = {'title':'Fixture','page':2,'text':'Function 47 only works when Function 11 is set to 2.',
              'setting_id':'function:47','setting_requirements':[{'setting_id':'function:11','value':'2'}]}
    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        answer, _ = ManualProvider('https://example.test','example','model',client).answer('?', [source])
    assert 'Function No. 11 = 2' in answer and 'Function No. 47' in answer


def test_applicability_guard_rejects_numeric_answer_for_excluded_subclass():
    import httpx
    from app.provider import ManualProvider, ProviderError
    def response(request):
        return httpx.Response(200, json={'choices':[{'message':{'content':json.dumps({
            'answer':'Use 7 mm.','supported':True,'citations':[1],'applicability':'applicable'})}}]})
    sources = [{'title':'Fixture','page':2,'text':'Oil procedure','section_key':'maintenance'},
               {'title':'Fixture','page':2,'text':'Subclass -453 needs no lubrication adjustment.',
                'section_key':'maintenance','dependency':'applicability','is_applicability_rule':True,
                'applicability':{'-453':'explicitly_excluded'}}]
    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        with pytest.raises(ProviderError):
            ManualProvider('https://example.test','example','model',client).answer('?', sources)


def test_cross_page_note_keeps_its_provenance_and_citation(tmp_path):
    import httpx
    from app.knowledge import UnitDraft
    from app.provider import ManualProvider
    row = UnitDraft(kind='table', title='4. FUNCTION SETTINGS', section_key='4',
                    text='No.: 47; Setting range: 0-10; See NOTE 7.',
                    table_headers=['No.', 'Setting range'], table_rows=[['47', '0-10']])
    note = UnitDraft(kind='information', title='4. FUNCTION SETTINGS', section_key='4',
                     text='(NOTE 7) Response is fastest at 10.')
    pages = [{'page': number, 'printed_label': str(number), 'method': 'text', 'text': draft.text,
              'units': [draft.model_dump()], 'visual_units': [], 'visual_review': 'unreviewed'}
             for number, draft in ((1, row), (2, note))]
    units = _assemble({'id': 'm', 'sha256': 'a' * 64}, pages)
    assert {ref['pdf_page_index'] for ref in units[0]['source_refs']} == {0, 1}
    sources = [{'id': units[0]['id'], 'title': 'Fixture', 'page': 1, 'text': units[0]['search_text'],
                'required_evidence_ids': [units[1]['id']]},
               {'id': units[1]['id'], 'title': 'Fixture', 'page': 2, 'text': units[1]['text']}]
    def response(request):
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps({
            'answer': 'সবচেয়ে দ্রুত প্রতিক্রিয়ার জন্য ১০ দিন।', 'supported': True, 'citations': [1]})}}]})
    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        _, citations = ManualProvider('https://example.test', 'example', 'model', client).answer('?', sources)
    assert citations == [1, 2]


def test_unresolved_numbered_note_blocks_the_owner():
    from app.knowledge import UnitDraft
    draft = UnitDraft(kind='information', title='4. SETTINGS', section_key='4', text='See NOTE 99.')
    page = {'page': 1, 'printed_label': '1', 'method': 'text', 'text': draft.text,
            'units': [draft.model_dump()], 'visual_units': [], 'visual_review': 'unreviewed'}
    units = _assemble({'id': 'm', 'sha256': 'a' * 64}, [page])
    assert 'NOTE 99' in units[0]['blocking_issues'][0]


def test_malformed_answer_quotes_are_escaped_without_another_api_call():
    import httpx
    from app.provider import ManualProvider
    malformed = '{"answer":"মান "1" দিন।","supported":true,"citations":[1]}'
    calls = []
    def response(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'message': {'content': malformed}}]})
    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        answer, citations = ManualProvider('https://example.test', 'example', 'model', client).answer(
            '?', [{'title': 'Fixture', 'page': 1, 'text': 'Select 1.'}])
    assert answer == 'মান "1" দিন।' and citations == [1]
    assert len(calls) == 1


def test_quote_repair_does_not_accept_malformed_control_fields():
    from app.provider import _answer_json
    with pytest.raises(json.JSONDecodeError):
        _answer_json('{"answer":"মান "1" দিন।","supported":yes,"citations":[1]}')


def test_english_question_still_requires_bengali_answer():
    import httpx
    from app.provider import ManualProvider, ProviderError
    def response(request):
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps({
            'answer': 'Select 1.', 'supported': True, 'citations': [1]})}}]})
    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        with pytest.raises(ProviderError, match='Bengali'):
            ManualProvider('https://example.test', 'example', 'model', client).answer(
                'Which setting?', [{'title': 'Fixture', 'page': 1, 'text': 'Select 1.'}])


@pytest.mark.parametrize('text', [
    'Operation is disabled when Function No. 62 is set to "1".',
    'Setting is enabled when Function No. 70 is set to "2" or "5".',
    'Setting is enabled when Function No. 118 is set from "1" to "3".',
    'Presser foot will not drop if DIP switch 1 is set to OFF.',
])
def test_complex_or_negative_conditions_are_not_reduced_to_one_required_value(text):
    from app.semantics import _requirements
    assert _requirements(text) == []
