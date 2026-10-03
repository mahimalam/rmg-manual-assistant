import json

import pymupdf
import pytest

from app.db import Database
from app.ingest import add_pdf, delete_pdf
from app.structured import BudgetPolicy, process_manual


def manual_pdf():
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Oil specifications")
    for x in (50, 180, 310):
        page.draw_line((x, 90), (x, 180))
    for y in (90, 120, 150, 180):
        page.draw_line((50, y), (310, y))
    for point, value in [((60, 110), "Mark"), ((190, 110), "Meaning"),
                         ((60, 140), "A"), ((190, 140), "HIGH"),
                         ((60, 170), "B"), ((190, 170), "LOW")]:
        page.insert_text(point, value)
    page.insert_text((50, 210), "WARNING: Disconnect power before servicing.")
    pdf = doc.tobytes()
    doc.close()
    return pdf


class FakeVLM:
    model = "synthetic-vlm"
    base_url = "https://example.test/v1"

    def __init__(self, invalid=False):
        self.calls = 0
        self.invalid = invalid

    def extract_page(self, image, text, context):
        assert image.startswith(b"\x89PNG")
        self.calls += 1
        return {"content": "{bad" if self.invalid else json.dumps({"units": [{
            "kind": "figure", "title": "Oil marks", "text": "A is HIGH; B is LOW.",
            "labels": {"A": "HIGH", "B": "LOW"}, "source_quote": "HIGH",
        }]}), "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "model": self.model, "finish_reason": "stop"}


def setup_manual(tmp_path):
    db = Database(tmp_path / "study.sqlite3")
    manual, _ = add_pdf(db, tmp_path, "synthetic.pdf", manual_pdf())
    return db, manual["id"]


def test_table_headers_rows_and_warnings_survive_restart(tmp_path):
    db, manual_id = setup_manual(tmp_path)
    result = process_manual(db, tmp_path, manual_id)
    assert result["pages_done"] == 1
    units = db.units(manual_id)
    rows = [json.loads(row["payload_json"]) for row in units if row["kind"] == "table"]
    assert rows and rows[0]["table_headers"] == ["Mark", "Meaning"]
    assert ["A", "HIGH"] in rows[0]["table_rows"]
    assert any("Disconnect power" in row["text"] for row in units)
    reopened = Database(tmp_path / "study.sqlite3")
    again = process_manual(reopened, tmp_path, manual_id)
    assert again["release_id"] == result["release_id"]
    assert len(reopened.units(manual_id)) == len(units)


def test_visual_extraction_cache_avoids_repeat_image_call(tmp_path):
    db, manual_id = setup_manual(tmp_path)
    provider = FakeVLM()
    budget = BudgetPolicy(1, 1, 5, 2)
    first = process_manual(db, tmp_path, manual_id, provider, budget, visual_pages=[1])
    second = process_manual(db, tmp_path, manual_id, provider, budget, visual_pages=[1])
    assert provider.calls == 1
    assert first["release_id"] == second["release_id"]
    assert any(row["kind"] == "figure" for row in db.units(manual_id))
    # The printed label is sent in the extraction request. Correcting that
    # context must not silently reuse a response to a different request.
    from app.structured import atomic_json
    local_path = next((tmp_path / "knowledge" / manual_id / "local").glob("*.json"))
    local = json.loads(local_path.read_text())
    local["printed_label"] = "corrected-label"
    atomic_json(local_path, local)
    third = process_manual(db, tmp_path, manual_id, provider, budget, visual_pages=[1])
    assert provider.calls == 2 and third["release_id"] != second["release_id"]


def test_missing_budget_makes_zero_visual_requests(tmp_path):
    db, manual_id = setup_manual(tmp_path)
    provider = FakeVLM()
    with pytest.raises(ValueError, match="budget"):
        process_manual(db, tmp_path, manual_id, provider, visual_pages=[1])
    assert provider.calls == 0


def test_failed_visual_output_is_not_automatically_rebilled(tmp_path):
    db, manual_id = setup_manual(tmp_path)
    local = process_manual(db, tmp_path, manual_id)
    original_chunks = [dict(row) for row in db.chunks()]
    provider = FakeVLM(invalid=True)
    for _ in range(2):
        with pytest.raises(ValueError, match="review|invalid|Invalid"):
            process_manual(db, tmp_path, manual_id, provider,
                           BudgetPolicy(1, 1, 5, 2), visual_pages=[1])
    assert provider.calls == 1
    assert db.job(manual_id)["status"] == "failed"
    assert [dict(row) for row in db.chunks()] == original_chunks
    assert local["release_id"]


def test_zero_remaining_allowance_prevents_dispatch(tmp_path):
    db, manual_id = setup_manual(tmp_path)
    provider = FakeVLM()
    with pytest.raises(ValueError, match="budget|allowance"):
        process_manual(db, tmp_path, manual_id, provider,
                       BudgetPolicy(0.001, 1, 5, 1), visual_pages=[1])
    assert provider.calls == 0


def test_deletion_removes_structured_assets_and_prevents_resume(tmp_path):
    db, manual_id = setup_manual(tmp_path)
    process_manual(db, tmp_path, manual_id)
    delete_pdf(db, tmp_path, manual_id)
    assert not db.units(manual_id)
    assert not (tmp_path / "knowledge" / manual_id).exists()
    with pytest.raises(ValueError, match="exist|removed"):
        process_manual(db, tmp_path, manual_id)


def test_independent_keyword_search_finds_code_without_dense_candidate():
    from app.retrieval import lexical_rank
    rows = [{"id": "needle", "text": "Check the needle."},
            {"id": "code", "text": "Error E950: disconnect power."}]
    assert lexical_rank("What is E950?", rows)[0][0] == "code"


def test_table_shape_rejects_misaligned_cells():
    from app.knowledge import UnitDraft
    with pytest.raises(ValueError):
        UnitDraft(kind="table", title="Oil", text="Oil table",
                  table_headers=["Mark", "Meaning"], table_rows=[["A"]])


def test_corrected_invalid_response_reuses_user_data_without_image_request(tmp_path):
    from app.structured import save_visual_correction
    db, manual_id = setup_manual(tmp_path)
    provider = FakeVLM(invalid=True)
    with pytest.raises(ValueError):
        process_manual(db, tmp_path, manual_id, provider, BudgetPolicy(1, 1, 5, 2), [1])
    corrected = json.dumps({"units": [{"kind": "figure", "title": "Oil marks",
                                      "text": "A is HIGH.", "source_quote": "HIGH"}]})
    save_visual_correction(db, tmp_path, manual_id, 1, corrected)
    process_manual(db, tmp_path, manual_id, provider, BudgetPolicy(1, 1, 5, 2), [1])
    assert provider.calls == 1
    figures = [json.loads(row["payload_json"]) for row in db.units(manual_id) if row["kind"] == "figure"]
    assert figures[0]["review_status"] == "user_checked"


def test_budget_unknown_usage_keeps_reservation(tmp_path):
    db, manual_id = setup_manual(tmp_path)
    class MissingUsage(FakeVLM):
        def extract_page(self, *args):
            response = super().extract_page(*args)
            response["usage"] = None
            return response
    budget = BudgetPolicy(1, 1, 5, 2)
    process_manual(db, tmp_path, manual_id, MissingUsage(), budget, [1])
    assert db.ingestion_spend()["accounted_usd"] == pytest.approx(budget.reservation)


def test_unpublished_upload_is_not_searchable(tmp_path):
    db = Database(tmp_path / "study.sqlite3")
    manual, _ = add_pdf(db, tmp_path, "pending.pdf", manual_pdf(), publish_baseline=False)
    assert not db.chunks()
    process_manual(db, tmp_path, manual["id"])
    assert db.chunks() and db.manuals()[0]["status"] == "ready"


def test_failed_replacement_keeps_old_manual_active(tmp_path):
    from app.structured import replace_manual
    db, manual_id = setup_manual(tmp_path)
    doc = pymupdf.open(stream=manual_pdf(), filetype="pdf")
    doc[0].insert_text((50, 250), "New revision")
    revision = doc.tobytes()
    doc.close()
    class FailedIndex:
        def prepare(self, *args, **kwargs):
            raise RuntimeError("simulated indexing failure")

        def sync(self):
            raise RuntimeError("simulated indexing failure")
    with pytest.raises(RuntimeError):
        replace_manual(db, tmp_path, manual_id, "revised.pdf", revision, FailedIndex())
    old = next(row for row in db.manuals() if row["id"] == manual_id)
    assert old["status"] == "ready"


def test_failed_vector_generation_does_not_switch_published_pointer(tmp_path, monkeypatch):
    from app.retrieval import Retriever
    monkeypatch.setattr('app.retrieval.model_signature',lambda name:name+'@synthetic-revision')
    db, manual_id = setup_manual(tmp_path)
    process_manual(db, tmp_path, manual_id)
    retriever = Retriever(db, tmp_path, "synthetic-embedding")
    batches = []
    def vectors(texts):
        batches.append(len(texts))
        return [[1.0, 0.0, 0.0] for _ in texts]
    monkeypatch.setattr(retriever, "_encode", vectors)
    retriever.sync()
    pointer = retriever.pointer.read_bytes()
    retriever.rebuild()
    assert len(batches) == 1
    with db.connect() as connection:
        connection.execute("UPDATE chunks SET text=text || ' changed' WHERE id=(SELECT id FROM chunks LIMIT 1)")
    def fail(texts):
        raise RuntimeError("simulated encoding failure")
    monkeypatch.setattr(retriever, "_encode", fail)
    with pytest.raises(RuntimeError):
        retriever.sync()
    assert retriever.pointer.read_bytes() == pointer


def test_required_warning_is_retrieved_with_table(tmp_path):
    from app.retrieval import Retriever
    db, manual_id = setup_manual(tmp_path)
    process_manual(db, tmp_path, manual_id)
    rows = {row["id"]: dict(row) for row in db.chunks()}
    table = next(row for row in db.units(manual_id) if row["kind"] == "table" and row["id"] in rows)
    retriever = Retriever(db, tmp_path, "synthetic-embedding")
    expanded = retriever._expand([{**rows[table["id"]], "score": 0.8}], rows)
    assert any("Disconnect power" in row["text"] for row in expanded)


def test_manual_feedback_is_durable_and_exported(tmp_path):
    from app.service import ask, export_feedback, save_feedback
    db, _ = setup_manual(tmp_path)
    class Search:
        def search(self, *args, **kwargs):
            return []
    class Provider:
        def search_query(self, text):
            return text
    result = ask(db, Search(), Provider(), "Unsupported question")
    save_feedback(db, result["id"], "incorrect", "Expected refusal on page 1")
    exported = export_feedback(Database(tmp_path / "study.sqlite3"))
    assert b"incorrect" in exported and b"Expected refusal" in exported


def test_explicit_visual_retry_retains_both_attempt_charges(tmp_path):
    db, manual_id = setup_manual(tmp_path)
    provider = FakeVLM(invalid=True)
    budget = BudgetPolicy(1, 1, 5, 2)
    with pytest.raises(ValueError):
        process_manual(db, tmp_path, manual_id, provider, budget, [1])
    provider.invalid = False
    process_manual(db, tmp_path, manual_id, provider, budget, [1], retry_failed=True)
    assert provider.calls == 2
    assert db.ingestion_spend()["attempts"] == 2
    assert db.ingestion_spend()["accounted_usd"] == pytest.approx(0.0007)


def test_index_failure_cannot_replace_published_knowledge(tmp_path):
    from app.structured import save_visual_correction
    db, manual_id = setup_manual(tmp_path)
    process_manual(db, tmp_path, manual_id)
    published = [dict(row) for row in db.chunks()]
    class FailedIndex:
        def prepare(self, *args, **kwargs):
            raise RuntimeError("simulated indexing failure")
    with pytest.raises(RuntimeError):
        save_visual_correction(db, tmp_path, manual_id, 1, json.dumps({"units": [{
            "kind": "figure", "title": "Oil marks", "text": "A is HIGH.", "source_quote": "HIGH",
        }]}), retriever=FailedIndex())
    assert [dict(row) for row in db.chunks()] == published


def test_unchanged_replacement_page_reuses_visual_data_and_resets_review(tmp_path):
    from app.structured import replace_manual, save_visual_correction
    db, manual_id = setup_manual(tmp_path)
    provider = FakeVLM()
    process_manual(db, tmp_path, manual_id, provider, BudgetPolicy(1, 1, 5, 2), [1])
    correction = json.dumps({"units": [{"kind": "figure", "title": "Oil marks",
                                       "text": "A is HIGH.", "source_quote": "HIGH"}]})
    save_visual_correction(db, tmp_path, manual_id, 1, correction)
    doc = pymupdf.open(stream=(tmp_path / "manuals" / f"{manual_id}.pdf").read_bytes(), filetype="pdf")
    doc.new_page().insert_text((50, 50), "Additional revision page")
    revision = doc.tobytes()
    doc.close()
    class Index:
        def prepare(self, *args):
            return None

        def sync(self):
            return 0
    replacement = replace_manual(db, tmp_path, manual_id, "synthetic.pdf", revision, Index())
    figures = [json.loads(row["payload_json"]) for row in db.units(replacement["id"])
               if row["kind"] == "figure"]
    assert provider.calls == 1
    assert figures[0]["text"] == "A is HIGH."
    assert figures[0]["review_status"] == "unreviewed"
    assert figures[0]["source_refs"][0]["manual_id"] == replacement["id"]
    assert next(row for row in db.manuals() if row["id"] == manual_id)["status"] == "retired"


def test_table_parent_is_canonical_but_rows_are_searchable(tmp_path):
    db, manual_id = setup_manual(tmp_path)
    process_manual(db, tmp_path, manual_id)
    tables = [json.loads(row["payload_json"]) for row in db.units(manual_id) if row["kind"] == "table"]
    parent = next(unit for unit in tables if len(unit["table_rows"]) > 1)
    searchable = {row["id"] for row in db.chunks()}
    assert parent["id"] not in searchable
    children = [unit for unit in tables if unit["parent_id"] == parent["id"]]
    assert len(children) == 2 and all(unit["id"] in searchable for unit in children)
    assert children[0]["table_headers"] == parent["table_headers"]


def test_ambiguous_machine_question_requests_manual_selection(tmp_path):
    from app.service import ask
    db, _ = setup_manual(tmp_path)
    class Search:
        def search(self, *args, **kwargs):
            return [{"id": "one:1", "manual_id": "one"}, {"id": "two:1", "manual_id": "two"}]
    class Provider:
        def search_query(self, text):
            return "How do I adjust the needle?"

        def answer(self, *args):
            pytest.fail("Ambiguous machine guidance must not reach generation")
    result = ask(db, Search(), Provider(), "সুঁই কীভাবে ঠিক করব?")
    assert result["status"] == "clarify" and not result["sources"]


def test_removal_purges_uncited_retrieval_evidence_and_feedback(tmp_path):
    from app.service import ask, save_feedback
    db, manual_id = setup_manual(tmp_path)
    source = dict(db.chunks()[0])
    source["score"] = 0.5
    class Search:
        def search(self, *args, **kwargs):
            return [source]
    class Provider:
        def search_query(self, text): return text
        def answer(self, *args): return "Unsupported", []
    result = ask(db, Search(), Provider(), "Unsupported")
    save_feedback(db, result["id"], "correct", "Refusal")
    delete_pdf(db, tmp_path, manual_id)
    with db.connect() as connection:
        assert not connection.execute("SELECT * FROM queries WHERE id=?", (result["id"],)).fetchone()
        assert not connection.execute("SELECT * FROM feedback WHERE query_id=?", (result["id"],)).fetchone()


def test_invalid_visual_cell_coordinates_are_rejected():
    from app.knowledge import UnitDraft
    with pytest.raises(ValueError):
        UnitDraft(kind="figure", title="Invalid", text="Bad rectangle", bbox=[0, 0, float('nan'), 10])
    with pytest.raises(ValueError, match="shape"):
        UnitDraft(kind="table", title="Bad grid", text="Bad grid", table_headers=["A", "B"],
                  table_rows=[["1", "2"]], cell_boxes=[[(0, 0, 1, 1)]])
