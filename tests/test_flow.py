import io
import json

import pymupdf
import httpx
import pytest

from app.db import Database
from app.ingest import add_pdf, delete_pdf, extract
from app.provider import ManualProvider, ProviderError
from app.service import (ask, export_queries, export_study, finish_trial, interrupt_trial,
                         start_trial)


def sample_pdf():
    document = pymupdf.open()
    document.set_metadata({"title": "Synthetic Sewing Machine Manual"})
    first = document.new_page()
    first.insert_text((72, 72), "Sewing machine test manual. If thread breaks, check needle installation.")
    second = document.new_page()
    second.insert_text((72, 72), "Before adjustment, switch off power. Technician only: inspect timing.")
    output = document.tobytes()
    document.close()
    return output


def test_pdf_upload_duplicate_and_delete(tmp_path):
    db = Database(tmp_path / "study.sqlite3")
    pdf = sample_pdf()
    manual, added = add_pdf(db, tmp_path, "manual.pdf", pdf)
    assert added and manual["pages"] == 2
    assert manual["title"] == "Synthetic Sewing Machine Manual"
    assert [(row["page"], row["extraction"]) for row in db.chunks()] == [
        (1, "text"), (2, "text")]
    duplicate, added = add_pdf(db, tmp_path, "same.pdf", pdf)
    assert not added and duplicate["id"] == manual["id"]
    with db.connect() as connection:
        connection.execute(
            "INSERT INTO queries (id, created_at, question, answer, status, sources_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("q1", "2026-01-01T00:00:00+00:00", "test", "source answer", "answered",
             json.dumps([{"id": f"{manual['id']}:1:0", "text": "source answer"}])),
        )
    delete_pdf(db, tmp_path, manual["id"])
    assert not db.chunks() and not db.manuals()
    with db.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM queries").fetchone()[0] == 0
    assert not (tmp_path / "manuals" / f"{manual['id']}.pdf").exists()


def test_invalid_pdf_is_not_stored(tmp_path):
    db = Database(tmp_path / "study.sqlite3")
    with pytest.raises(ValueError, match="not a PDF"):
        add_pdf(db, tmp_path, "bad.pdf", b"not a pdf")
    assert not db.manuals()


def test_image_only_page_can_be_asked_with_explicit_page_hint(tmp_path):
    db = Database(tmp_path / "study.sqlite3")
    document = pymupdf.open()
    document.new_page()
    pdf = document.tobytes()
    document.close()
    manual, added = add_pdf(db, tmp_path, "diagram.pdf", pdf, ocr=False)
    assert added and not db.chunks()

    class Retriever:
        def search(self, question, manual_ids=None):
            return []

    class Provider:
        def search_query(self, question):
            return question

        def answer(self, question, sources):
            assert sources[0]["page_image"].startswith(b"\x89PNG")
            return "এটি নির্বাচিত পৃষ্ঠা।", [1]

    result = ask(db, Retriever(), Provider(), "What is on this page?", [manual["id"]],
                 page_hint=(manual["id"], 1), runtime=tmp_path)
    assert result["status"] == "answered" and result["sources"][0]["page"] == 1


def test_provider_citation_validation():
    source = [{"title": "Manual", "page": 2, "text": "Switch off power first."}]

    def mock(request):
        assert request.headers["authorization"] == "Bearer example"
        payload = json.loads(request.content)
        assert "Switch off power" in payload["messages"][1]["content"]
        return httpx.Response(200, json={"choices": [{"message": {"content":
            json.dumps({"answer": "বিদ্যুৎ বন্ধ করুন।", "supported": True, "citations": [1]})}}]})

    with httpx.Client(transport=httpx.MockTransport(mock)) as client:
        provider = ManualProvider("https://example.test/v1", "example", "test-model", client)
        answer, citations = provider.answer("প্রথমে কী করব?", source)
    assert answer == "বিদ্যুৎ বন্ধ করুন।" and citations == [1]

    def invalid(request):
        return httpx.Response(200, json={"choices": [{"message": {"content":
            '{"answer":"wrong","supported":true,"citations":[2]}'}}]})

    with httpx.Client(transport=httpx.MockTransport(invalid)) as client:
        provider = ManualProvider("https://example.test/v1", "example", "test-model", client)
        with pytest.raises(ProviderError, match="invalid grounded-answer"):
            provider.answer("?", source)


def test_bangla_question_is_translated_for_english_manual_search():
    def mock(request):
        payload = json.loads(request.content)
        assert "Translate" in payload["messages"][0]["content"]
        return httpx.Response(200, json={"choices": [{"message": {"content":
            json.dumps({"search_question": "What if thread breaks on S-7200A?"})}}]})

    with httpx.Client(transport=httpx.MockTransport(mock)) as client:
        provider = ManualProvider("https://example.test/v1", "example", "test-model", client)
        assert provider.search_query("S-7200A তে সুতা ছিঁড়লে কী করব?") == \
            "What if thread breaks on S-7200A?"
        assert provider.search_query("What if thread breaks?") == "What if thread breaks?"


def test_page_image_is_sent_only_when_supplied():
    def mock(request):
        payload = json.loads(request.content)
        content = payload["messages"][1]["content"]
        assert any(part["type"] == "image_url" and
                   part["image_url"]["url"].startswith("data:image/png;base64,")
                   for part in content)
        return httpx.Response(200, json={"choices": [{"message": {"content":
            '{"answer":"Figure shows power switch","supported":true,"citations":[1]}'}}]})

    source = [{"title": "Synthetic", "page": 1, "text": "[Selected page image]",
               "page_image": b"synthetic-png"}]
    with httpx.Client(transport=httpx.MockTransport(mock)) as client:
        provider = ManualProvider("https://example.test/v1", "example", "test-model", client)
        answer, citations = provider.answer("Answer in English: what does this figure show?", source)
    assert citations == [1] and "power switch" in answer


def test_answer_citation_numbers_are_resolved_to_pdf_pages():
    def mock(request):
        return httpx.Response(200, json={"choices": [{"message": {"content":
            '{"answer":"Switch off power [1].","supported":true,"citations":[1]}'}}]})

    with httpx.Client(transport=httpx.MockTransport(mock)) as client:
        provider = ManualProvider("https://example.test/v1", "example", "test-model", client)
        answer, ids = provider.answer("Answer in English: what first?", [
            {"title": "Synthetic", "page": 3, "text": "Switch off power."}])
    assert answer == "Switch off power (Synthetic, PDF p. 3)." and ids == [1]


def test_provider_auth_error_does_not_expose_credential():
    with httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(401, text="invalid token"))) as client:
        provider = ManualProvider("https://example.test/v1", "private-example-token",
                                  "test-model", client)
        with pytest.raises(ProviderError, match="HTTP 401") as error:
            provider.answer("What first?", [
                {"title": "Synthetic", "page": 1, "text": "Switch off power."}])
    assert "private-example-token" not in str(error.value)


def test_trial_final_answer_and_query_are_durable(tmp_path):
    db = Database(tmp_path / "study.sqlite3")
    with db.connect() as connection:
        connection.execute("INSERT INTO manuals (id,title,filename,sha256,pages,status,created_at) "
                           "VALUES ('m','Synthetic','synthetic.pdf',?,1,'ready','test')", ('a' * 64,))
    trial_id = start_trial(db, "assistant", "Synthetic thread-break scenario")

    class Retriever:
        def search(self, question, manual_ids=None):
            return [{"id": "m:1:0", "manual_id": "m", "title": "Synthetic", "page": 1,
                     "score": 0.8, "text": "Check needle installation."}]

    class Provider:
        def search_query(self, question):
            return question

        def answer(self, question, matches):
            return "Check needle installation.", [1]

    result = ask(db, Retriever(), Provider(), "Why does thread break?", trial_id=trial_id)
    assert result["sources"][0]["page"] == 1
    first = finish_trial(db, trial_id, "Needle may be installed incorrectly")
    second = finish_trial(db, trial_id, "This must not overwrite")
    assert first["final_answer"] == second["final_answer"]
    assert b"Needle may be installed incorrectly" in export_study(db)
    assert b",1\r\n" in export_study(db)
    assert b"Check needle installation" in export_queries(db)


def test_only_one_active_trial_and_interruption_is_recorded(tmp_path):
    db = Database(tmp_path / "study.sqlite3")
    first = start_trial(db, "pdf", "Synthetic case")
    with pytest.raises(ValueError, match="active trial"):
        start_trial(db, "assistant", "Next case")
    interrupt_trial(db, first)
    with db.connect() as connection:
        assert connection.execute("SELECT status FROM trials WHERE id=?", (first,)).fetchone()[0] == "interrupted"
    assert start_trial(db, "assistant", "Next case") != first


def test_no_evidence_skips_provider_and_logs_refusal(tmp_path):
    db = Database(tmp_path / "study.sqlite3")

    class Retriever:
        def search(self, question, manual_ids=None):
            return []

    class Provider:
        def search_query(self, question):
            return question

        def answer(self, question, matches):
            raise AssertionError("Provider should not be called")

    result = ask(db, Retriever(), Provider(), "Unknown model")
    assert result["status"] == "no_evidence" and result["sources"] == []
