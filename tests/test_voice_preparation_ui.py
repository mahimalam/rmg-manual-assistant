"""Exercise prepared voice forms against an isolated library, with no paid calls."""

import json
import pytest
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

from app.config import Settings
from app.db import Database, utc_now
from app.provider import ManualProvider
import httpx


def test_english_recording_direct_ask_and_user_correction_reach_the_real_question_flow(tmp_path, monkeypatch):
    settings = Settings(runtime=tmp_path, api_key="test-key")
    db = Database(tmp_path / "study.sqlite3")
    with db.connect() as connection:
        connection.execute("INSERT INTO manuals (id,title,filename,sha256,pages,status,created_at) "
                           "VALUES (?,?,?,?,?,?,?)", ("test", "S-7200A test manual", "manual.pdf",
                           "a" * 64, 1, "ready", utc_now()))
    monkeypatch.setattr(Settings, "load", lambda: settings)
    monkeypatch.setattr("streamlit.audio_input", lambda *a, **k: SimpleNamespace(getvalue=lambda: b"recording"))
    monkeypatch.setattr("app.interface.detect_recording_language", lambda _: ("en", 0.9))
    monkeypatch.setattr("app.interface.speech_samples", lambda _: None)
    original = "S-7200A: After knee switch use, the pedal cannot raise the presser foot."
    draft = "S-7200A: হাঁটু সুইচ ব্যবহারের পরে প্যাডেল দিয়ে প্রেসার ফুট উপরে তোলা যায় না।"
    recognized = []
    monkeypatch.setattr("app.interface.transcribe_preferred", lambda *a:
                        (recognized.append(a), (original, "Local"))[1])
    monkeypatch.setattr(ManualProvider, "prepare_voice_question", lambda *a, **k: (draft, []))
    class Search:
        def __init__(self, *a): pass
        def sync(self): pass
        def search(self, *a, **k): return []
    monkeypatch.setattr("app.retrieval.Retriever", Search)
    app = AppTest.from_file(str(settings.root / "main.py"), default_timeout=20).run()
    app.session_state['recording_language'] = 'Auto detect'
    app.radio(key="input_mode").set_value("Voice").run()
    assert not app.exception and app.text_area(key="voice_question").value == draft
    ask_button = lambda: next(b for b in app.button if b.label == "Ask")
    ask_button().click().run()
    assert not any(c.label.startswith("I checked the transcript") for c in app.checkbox)
    assert not app.exception and not app.error and len(recognized) == 1
    result = app.session_state["result"]
    with db.connect() as connection:
        saved = connection.execute("SELECT * FROM queries WHERE id=?", (result['id'],)).fetchone()
    context = json.loads(saved['input_context_json'])
    assert saved['question'] == original == saved['search_question']
    assert context['bengali_draft'] == draft and not context['edited']
    corrected = "S-7200A: হাঁটু সুইচ ব্যবহারের আগেই প্যাডেল কাজ করে না।"
    monkeypatch.setattr(ManualProvider, "search_query", lambda *a: "S-7200A pedal fails before knee switch use")
    app.text_area(key="voice_question").set_value(corrected)
    ask_button().click().run()
    assert not app.exception and not app.error
    context = app.session_state["result"]["input_context"]
    assert context['submitted_question'] == corrected and context['edited']
    assert context['transcript'] == original and len(recognized) == 1
    monkeypatch.setattr(ManualProvider, "prepare_voice_question", lambda *a, **k:
                        (_ for _ in ()).throw(ValueError("Translation offline")))
    app.selectbox(key="recording_language").set_value("English").run()
    assert not app.exception
    assert any(b.label == "Retry question preparation" for b in app.button)
    assert app.text_area(key="voice_question").value == original
    with pytest.raises(KeyError):
        app.session_state["result"]
    monkeypatch.setattr(ManualProvider, "prepare_voice_question", lambda *a, **k: (draft, []))
    next(b for b in app.button if b.label == "Retry question preparation").click().run()
    assert not app.exception and not app.error and len(recognized) == 2
    assert app.text_area(key="voice_question").value == draft


def test_bangla_default_cloud_hints_optional_warning_and_direct_ask(tmp_path, monkeypatch):
    settings = Settings(runtime=tmp_path, api_key='test-key', deepgram_api_key='mock-cloud-key')
    db = Database(tmp_path / 'study.sqlite3')
    with db.connect() as c:
        c.execute("INSERT INTO manuals (id,title,filename,sha256,pages,status,created_at) VALUES (?,?,?,?,?,?,?)",
                  ('m', 'Brother S-7200A Manual', 'manual.pdf', 'b' * 64, 1, 'ready', utc_now()))
        c.execute("INSERT INTO chunks VALUES ('c','m',1,0,'presser foot rotary hook','text')")
    monkeypatch.setattr(Settings, 'load', lambda: settings)
    monkeypatch.setattr('streamlit.audio_input', lambda *a, **k: SimpleNamespace(getvalue=lambda: b'recording'))
    monkeypatch.setattr('app.interface.speech_samples', lambda a: None)
    monkeypatch.setattr('app.interface.detect_recording_language', lambda a: pytest.fail('Bangla ran detection'))
    seen = []
    original = 'S-7200A প্রেসার ফুট ওঠে না।'
    def post(url, **kwargs):
        seen.append(kwargs['params'])
        return httpx.Response(200, request=httpx.Request('POST', url), json={'results': {'channels': [
            {'alternatives': [{'transcript': original, 'words': [{'word': 'না', 'confidence': .4}]}]}]}})
    monkeypatch.setattr('app.voice.httpx.post', post)
    monkeypatch.setattr(ManualProvider, 'search_query', lambda self, q: q)
    class Search:
        def __init__(self, *a): pass
        def sync(self): pass
        def search(self, *a, **k): return []
    monkeypatch.setattr('app.retrieval.Retriever', Search)
    app = AppTest.from_file(str(settings.root / 'main.py'), default_timeout=20).run()
    app.radio(key='input_mode').set_value('Voice').run()
    assert not app.exception and app.selectbox(key='recording_language').value == 'Bangla'
    assert app.session_state['voice_source'] == 'Deepgram'
    assert seen[0]['language'] == 'bn' and 'presser foot' in seen[0].get_list('keyterm')
    assert any('negatives' in w.value for w in app.warning)
    next(b for b in app.button if b.label == 'Ask').click().run()
    assert not app.exception and not app.error and len(seen) == 1
    context = app.session_state['result']['input_context']
    assert context['submitted_question'] == original and not context['reviewed']
    assert context['recognition']['words'][0]['word'] == 'না'
