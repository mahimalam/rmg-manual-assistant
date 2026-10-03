"""The cloud recognizer is preferred; failures return to local speech."""

import httpx
import pymupdf

from app import voice
from app.db import Database
from app.ingest import add_pdf, delete_pdf
from app.structured import process_manual


def test_deepgram_is_primary_and_receives_only_the_recording(monkeypatch):
    seen = {}

    def post(url, **kwargs):
        seen.update(url=url, **kwargs)
        return httpx.Response(200, request=httpx.Request("POST", url), json={"results": {"channels": [
            {"alternatives": [{"transcript": "S-7200A মেশিনের প্রশ্ন।"}]}]}})

    monkeypatch.setattr(voice.httpx, "post", post)
    monkeypatch.setattr(voice, "transcribe", lambda *_: (_ for _ in ()).throw(AssertionError("local used")))
    assert voice.transcribe_preferred(b"recording", "local-model", "test-key") == (
        "S-7200A মেশিনের প্রশ্ন।", "Deepgram")
    assert seen["url"] == "https://api.deepgram.com/v1/listen"
    assert seen["params"]["language"] == "bn" and seen["params"]["model"] == "nova-3"
    assert seen["headers"]["Authorization"] == "Token test-key"
    assert seen["content"] == b"recording"


def test_deepgram_auth_failure_uses_local_recognizer(monkeypatch):
    request = httpx.Request("POST", "https://api.deepgram.com/v1/listen")
    monkeypatch.setattr(voice.httpx, "post", lambda *_args, **_kwargs: httpx.Response(401, request=request))
    monkeypatch.setattr(voice, "transcribe", lambda audio, model, language: "লোকাল প্রশ্ন")
    assert voice.transcribe_preferred(b"recording", "local-model", "invalid-key") == (
        "লোকাল প্রশ্ন", "Local fallback")


def test_deepgram_bad_transcript_and_network_failure_use_local(monkeypatch):
    monkeypatch.setattr(voice, "transcribe", lambda audio, model, language: "লোকাল প্রশ্ন")
    monkeypatch.setattr(voice.httpx, "post", lambda *_args, **_kwargs: httpx.Response(
        200, request=httpx.Request("POST", "https://api.deepgram.com/v1/listen"),
        json={"results": {"channels": [{"alternatives": [{"transcript": "मशीन"}]}]}}))
    assert voice.transcribe_preferred(b"recording", "local-model", "test-key")[1] == "Local fallback"

    def offline(*_args, **_kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(voice.httpx, "post", offline)
    assert voice.transcribe_preferred(b"recording", "local-model", "test-key")[1] == "Local fallback"


def test_no_key_stays_local_without_network(monkeypatch):
    monkeypatch.setattr(voice.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("network used")))
    monkeypatch.setattr(voice, "transcribe", lambda audio, model, language: "লোকাল প্রশ্ন")
    assert voice.transcribe_preferred(b"recording", "local-model", "") == ("লোকাল প্রশ্ন", "Local")


def test_domain_hints_are_separate_parameters_and_confidence_does_not_rewrite_speech(monkeypatch):
    seen = {}
    def post(url, **kwargs):
        seen.update(kwargs)
        return httpx.Response(200, request=httpx.Request('POST', url), json={
            'metadata': {'request_id': 'test-request'}, 'results': {'channels': [{'alternatives': [{
                'transcript': 'Function 39 অন নয়।', 'confidence': .9,
                'words': [{'word': '39', 'confidence': .4}, {'word': 'নয়', 'confidence': .5}]}]}]}})
    monkeypatch.setattr(voice.httpx, 'post', post)
    details = {}
    transcript, source = voice.transcribe_preferred(b'recording', 'local', 'test-key', 'bn',
                                                   ['S-7200A', 'rotary hook', 'প্রেসার ফুট'], details)
    assert source == 'Deepgram' and transcript == 'Function 39 অন নয়।'
    assert seen['params'].get_list('keyterm') == ['S-7200A', 'rotary hook', 'প্রেসার ফুট']
    assert seen['content'] == b'recording' and seen['params']['language'] == 'bn'
    assert details['request_id'] == 'test-request'
    assert '39' in voice.recognition_warnings(details)[0] and 'নয়' in voice.recognition_warnings(details)[0]
    assert voice.recognition_warnings({'words': [{'word': '39', 'confidence': .99}]}) == []


def test_invalid_key_fallback_has_safe_diagnostics(monkeypatch):
    monkeypatch.setattr(voice.httpx, 'post', lambda url, **kwargs: httpx.Response(401, request=httpx.Request('POST', url)))
    monkeypatch.setattr(voice, 'transcribe', lambda *args: 'ফুট ওঠে না')
    details = {'confidence': .99}
    assert voice.transcribe_preferred(b'recording', 'local', 'secret-test-key', details=details)[1] == 'Local fallback'
    assert details == {'fallback_reason': 'Deepgram rejected the credential.'}


def test_keyterms_are_deduplicated_and_bounded_including_bengali():
    terms = voice.bounded_keyterms(['rotary hook', 'Rotary hook', 'bad,term', 'bad:0.5', 'প্রেসার ফুট'] +
                                 [f'pressure regulator {i}' for i in range(100)])
    assert terms[:2] == ['rotary hook', 'প্রেসার ফুট']
    assert len(terms) <= 32 and sum(len(t.encode()) + 1 for t in terms) <= 480


def test_new_upload_and_deletion_update_glossary_without_manual_edit(tmp_path):
    db = Database(tmp_path / 'study.sqlite3')
    pdf = pymupdf.open(); page = pdf.new_page()
    pdf.set_metadata({'title': 'Acme PX-900 Service Manual'})
    page.insert_text((40, 45), '1. ADJUSTING THE PRESSURE REGULATOR')
    page.insert_text((40, 90), 'Check the inlet pressure before adjusting the regulator.')
    manual, _ = add_pdf(db, tmp_path, 'pressure.pdf', pdf.tobytes(), ocr=False)
    pdf.close()
    process_manual(db, tmp_path, manual['id'])
    terms = voice.manual_keyterms(db, [manual['id']])
    assert 'PX-900' in terms and 'pressure regulator' in terms
    assert 'S-7200A' not in terms and 'প্রেসার ফুট' not in terms
    assert voice.manual_keyterms(db, []) == []
    delete_pdf(db, tmp_path, manual['id'])
    assert voice.manual_keyterms(db, [manual['id']]) == []
