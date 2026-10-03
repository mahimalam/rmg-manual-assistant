"""A merged PDF header is recoverable only with explicit cell geometry."""

from types import SimpleNamespace

from app.tables import AMBIGUOUS_TABLE, native_table


class Table:
    def __init__(self, merged=True):
        self.header = SimpleNamespace(names=['Problem', None, 'Possible cause'],
            cells=[(0, 0, 100 if merged else 20, 10), None, (100, 0, 200, 10)], external=False)
        self.rows = [SimpleNamespace(cells=self.header.cells), SimpleNamespace(cells=[
            (0, 10, 20, 40), (20, 10, 100, 40), (100, 10, 200, 40)])]

    def extract(self):
        return [['Problem', None, 'Possible cause'], ['10', 'Threads are breaking.', 'Is the needle bent?']]


def test_shared_header_is_propagated_only_when_its_rectangle_spans_the_column():
    headers, rows, _, flags = native_table(Table(), 'Troubleshooting')
    assert headers == ['Problem', 'Problem', 'Possible cause']
    assert rows == [['10', 'Threads are breaking.', 'Is the needle bent?']] and not flags
    headers, _, _, flags = native_table(Table(merged=False), 'Troubleshooting')
    assert 'header unresolved' in headers[1] and AMBIGUOUS_TABLE in flags


def test_new_pdf_with_shared_problem_header_is_ingested_without_manual_edits(tmp_path):
    import pymupdf
    from app.db import Database
    from app.ingest import add_pdf
    from app.structured import process_manual

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((30, 50), '1. TROUBLESHOOTING')
    page.draw_rect(pymupdf.Rect(30, 80, 520, 180))
    for y in (105, 140):
        page.draw_line((30, y), (520, y))
    page.draw_line((220, 80), (220, 180))
    page.draw_line((70, 105), (70, 180))
    for x, y, text in [(35, 96, 'Problem'), (225, 96, 'Possible cause'),
                       (35, 122, '1'), (75, 122, 'Threads are breaking.'), (225, 122, 'Is the needle bent?'),
                       (35, 157, '2'), (75, 157, 'Skipped stitches.'), (225, 157, 'Is the needle installed correctly?')]:
        page.insert_text((x, y), text, fontsize=10)
    db = Database(tmp_path / 'study.sqlite3')
    manual, _ = add_pdf(db, tmp_path, 'new-manual.pdf', document.tobytes(), publish_baseline=False)
    document.close()
    process_manual(db, tmp_path, manual['id'])
    rows = [dict(row) for row in db.chunks([manual['id']])]
    assert any('Threads are breaking.' in row['text'] and 'Is the needle bent?' in row['text'] for row in rows)
    assert not any('header unresolved' in row['text'] for row in rows)


def test_unreadable_optional_procedure_does_not_hide_complete_troubleshooting(tmp_path):
    import json
    import pytest
    from app.db import Database
    from app.knowledge import UnitDraft
    from app.retrieval import Retriever
    from app.structured import _assemble

    drafts = [UnitDraft(kind='troubleshooting', title='Threads break',
                        text='Check whether the needle is bent.', checks=['Check whether the needle is bent.'], references=['2']),
              UnitDraft(kind='table', title='Adjustment', text='Unresolved cells',
                        table_headers=['No.', 'Value'], table_rows=[['1', None]],
                        uncertainties=[AMBIGUOUS_TABLE])]
    pages = [dict(page=i + 1, printed_label=str(i + 1), method='text', text=d.text,
                  units=[d.model_dump()], visual_units=[], visual_review='unreviewed')
             for i, d in enumerate(drafts)]
    units = _assemble({'id': 'm', 'sha256': 'a' * 64}, pages)
    db = Database(tmp_path / 'study.sqlite3')
    with db.connect() as c:
        c.execute("INSERT INTO manuals VALUES ('m','Fixture','source.pdf',?,2,'ready','test',NULL,'baseline')", ('a' * 64,))
        for u in units:
            c.execute('INSERT INTO knowledge_units VALUES (?,?,?,?,?,?,?)',
                      (u['id'], 'm', u['page'], u['kind'], u['search_text'], json.dumps(u), 'fixture'))
    r = object.__new__(Retriever)
    r.db, r.release_id = db, 'fixture'
    primary = {'id': units[0]['id'], 'manual_id': 'm', 'page': 1, 'title': 'Fixture', 'text': units[0]['search_text']}
    sources = r._expand([primary], {primary['id']: primary})
    assert len(sources) == 1 and sources[0]['unresolved_dependencies'][0]['page'] == 2
    assert not sources[0]['required_evidence_ids']
    units[0]['dependencies'][0]['relation'] = 'prerequisite'
    with db.connect() as c:
        c.execute('UPDATE knowledge_units SET payload_json=? WHERE id=?', (json.dumps(units[0]), units[0]['id']))
    with pytest.raises(ValueError, match='unresolved extraction'):
        r._expand([primary], {primary['id']: primary})


def test_individual_lubrication_check_keeps_its_subclass_qualifier():
    from app.grounding import scope_notes
    notes = scope_notes([{'title': 'Fixture', 'page': 61, 'text':
        'Is the needle bent?\nIs the hook lubricated? (-40[], 43[] specifications)'}])
    assert '(-40[], 43[] specifications)' in notes and 'lubricated' in notes
    assert 'needle' not in notes
