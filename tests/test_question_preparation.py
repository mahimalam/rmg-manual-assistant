"""Regression cases at the speech/translation/RAG boundary, without paid calls."""

import json
from types import SimpleNamespace

import httpx
import pytest

from app import interface, voice
from app.provider import ManualProvider, ProviderError, validate_question_translation


@pytest.mark.parametrize("original, translated", [
    ("S-7200A Function No. 40 = 0", "S-7300A Function No. 40 = 0 কেন?"),
    ("S-7200A Function No. 40 = 0", "S-7200A Function No. 40 = 1 কেন?"),
    ("Function No. 40 = 0; Function No. 12 = 1", "Function No. 40 = 1; Function No. 12 = 0 কেন?"),
    ("DIP switch 1 = OFF", "DIP switch 1 = ON কেন?"),
    ("Run it for 8 seconds", "এটি 8 মিনিট চালান"),
    ("8 seconds and 0.5 mm", "8 মিমি এবং 0.5 সেকেন্ড"),
    ("How wide in mm?", "কত সেমি চওড়া?"),
    ("Oil band is 0.5 mm", "তেলের দাগ 0.5 ভোল্ট"),
    ("The motor does not start", "মোটর চালু হয়"),
    ("The motor starts", "মোটর চালু হয় না"),
    ("After knee switch use, the pedal cannot raise the foot.", "হাঁটু সুইচ ব্যবহারের পরে প্যাডেল দিয়ে প্রেসার ফুট ওঠে।"),
    ("After thread trimming the foot drops at neutral", "নিউট্রালে ফুট নামে"),
    ("The stopped machine's pedal cannot raise the foot", "প্যাডেল দিয়ে ফুট ওঠে না"),
    ("Sometimes the motor stops", "মোটর থামে"),
    ("Maybe the motor is faulty", "মোটর খারাপ"),
    ("Turn it counterclockwise", "ঘড়ির কাঁটার দিকে ঘুরান"),
    ("Function No. 40 = 0 and Function No. 12 = 0", "Function No. 40 এবং Function No. 12 = 0 কেন?"),
])
def test_translation_drift_is_rejected(original, translated):
    with pytest.raises(ValueError):
        validate_question_translation(original, translated)


@pytest.mark.parametrize("original, translated", [
    ("S-7200A-453: run for 8 seconds; band 0.5 mm", "S-7200A-453: 8 সেকেন্ড চালান; দাগ 0.5 মিমি"),
    ("Function No. 40 = 0; DIP switch 1 = OFF", "Function No. 40 = 0; DIP switch 1 = OFF কেন?"),
    ("The motor does not start", "মোটর চালু হয় না"),
    ("Do not adjust the motor", "মোটর সমন্বয় করবেন না"),
    ("Sometimes the motor stops", "মাঝে মাঝে মোটর থামে"),
    ("मशीन शुरू नहीं हो रही है", "মেশিন চালু হচ্ছে না"),
])
def test_valid_translation_preserves_constraints(original, translated):
    validate_question_translation(original, translated)


def test_english_is_transcribed_with_multilingual_model_not_bengali_checkpoint(monkeypatch):
    models, options = [], []
    class Recognizer:
        def transcribe(self, audio, **kwargs):
            options.append(kwargs)
            return [SimpleNamespace(text="S-7200A motor does not start")], None
    monkeypatch.setattr(voice, "_whisper", lambda name: (models.append(name), Recognizer())[1])
    assert voice.transcribe(b"fixture", "bengali-checkpoint", "en") == "S-7200A motor does not start"
    assert models == ["medium"] and options[0]["language"] == "en" and options[0]["task"] == "transcribe"
    assert options[0]['without_timestamps'] is True


@pytest.mark.parametrize('content', [
    'I need the Bengali text to translate. Please provide the original text.',
    '{"search_question":"Please provide the original Bengali text."}',
    '{"search_question":"Why does the motor not start?","answer":"Replace it"}',
])
def test_search_translation_commentary_falls_back_to_original(content):
    original = 'মোটর চালু হয় না কেন?'
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
            'choices': [{'message': {'content': content}}]}))) as client:
        provider = ManualProvider('https://example.test', 'test', 'model', client)
        assert provider.search_query(original) == original
        assert provider.translation_issue


def test_cloud_receives_detected_english_and_returns_original_english(monkeypatch):
    def post(url, **kwargs):
        assert kwargs["params"]["language"] == "en"
        assert "detect_language" not in kwargs["params"]
        return httpx.Response(200, request=httpx.Request("POST", url), json={"results": {
            "channels": [{"alternatives": [{"transcript": "The motor does not start."}]}]}})
    monkeypatch.setattr(voice.httpx, "post", post)
    assert voice.transcribe_preferred(b"fixture", "bn-model", "test-key", "en")[0] == "The motor does not start."


def test_genuine_non_bengali_script_is_not_mistaken_for_failed_bengali():
    assert voice._validated_transcript("मशीन शुरू नहीं हो रही है", "hi")
    with pytest.raises(ValueError):
        voice._validated_transcript("मशीन", "bn")


def test_bengali_and_technical_terms_do_not_make_translation_call():
    provider = ManualProvider("https://example.test", "", "model")
    text = "S-7200A মেশিনে Function No. 40 = 0 কেন?"
    assert provider.prepare_voice_question(text) == (text, []) and provider.attempt_count == 0


@pytest.mark.parametrize("fenced", [False, True])
def test_live_gateway_json_fence_is_accepted_without_accepting_commentary(fenced):
    content = json.dumps({"bengali_question": "মোটর চালু হয় না", "uncertainties": []})
    if fenced:
        content = "```json\n" + content + "\n```"
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
            "choices": [{"message": {"content": content}}]}))) as client:
        provider = ManualProvider("https://example.test", "test", "model", client)
        assert provider.prepare_voice_question("The motor does not start", "en")[0] == "মোটর চালু হয় না"


@pytest.mark.parametrize("response", [
    "not json", {"bengali_question": "English only", "uncertainties": []},
    {"bengali_question": "মোটর চালু হয়", "uncertainties": []},
    {"bengali_question": "মোটর চালু হয় না", "uncertainties": "none"},
    {"bengali_question": "মোটর চালু হয় না", "uncertainties": [], "answer": "invented"},
])
def test_bad_preparation_response_is_not_submitted(response):
    content = response if isinstance(response, str) else json.dumps(response)
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
            "choices": [{"message": {"content": content}}]}))) as client:
        provider = ManualProvider("https://example.test", "test", "model", client)
        with pytest.raises(ProviderError):
            provider.prepare_voice_question("The motor does not start")


def test_preparation_retry_does_not_resend_audio_and_edit_becomes_authoritative(monkeypatch):
    calls = []
    monkeypatch.setattr(interface, "detect_recording_language", lambda _: ("en", 0.9))
    monkeypatch.setattr(interface, "transcribe_preferred", lambda *args: (calls.append(args),
        ("The motor does not start", "Local"))[1])
    class Translator:
        attempts = 0
        def prepare_voice_question(self, original, language=None):
            self.attempts += 1
            if self.attempts == 1:
                raise ProviderError("offline")
            return "মোটর চালু হয় না", []
    translator, state = Translator(), {}
    interface.update_recording(b"audio", state, "model", "", provider=translator)
    assert state["voice_transcript"] == "The motor does not start" and state["voice_preparation_error"] == "offline"
    original, metadata = interface.voice_submission(state["voice_question"], state)
    assert original == "The motor does not start" and not metadata["reviewed"]
    corrected, context = interface.voice_submission("S-7200A মোটর চালু হয় না", state)
    assert corrected == "S-7200A মোটর চালু হয় না" and context["edited"]
    interface.update_recording(b"audio", state, "model", "", retry=True, provider=translator)
    assert len(calls) == 1 and translator.attempts == 2
    original, context = interface.voice_submission(state["voice_question"], state)
    assert original == "The motor does not start" and not context["edited"]
    assert "voice_preparation_error" not in state
    assert not interface.update_recording(b"audio", state, "model", "", provider=translator)
    assert translator.attempts == 2


def test_language_override_reprocesses_same_audio(monkeypatch):
    calls = []
    monkeypatch.setattr(interface, "detect_recording_language", lambda _: ("en", 0.4))
    monkeypatch.setattr(interface, "speech_samples", lambda _: None)
    monkeypatch.setattr(interface, "transcribe_preferred", lambda a, m, k, language, *extra:
                        (calls.append(language), ("মোটরের প্রশ্ন", "Local"))[1])
    provider = ManualProvider("https://example.test", "", "model")
    state = {}
    interface.update_recording(b"audio", state, "model", "", provider=provider)
    first_hash = state["recording_hash"]
    interface.update_recording(b"audio", state, "model", "", language="bn", provider=provider)
    assert calls == ["en", "bn"] and first_hash != state["recording_hash"]


@pytest.mark.parametrize("language", ["auto", "bn", "en"])
def test_silent_audio_stops_before_cloud_or_translation(monkeypatch, language):
    import io
    import wave
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1); output.setsampwidth(2); output.setframerate(16000)
        output.writeframes(b"\0\0" * 16000)
    monkeypatch.setattr(interface, "transcribe_preferred", lambda *args: pytest.fail("Silent audio reached recognizer"))
    state = {"voice_question": "আগের প্রশ্ন"}
    interface.update_recording(buffer.getvalue(), state, "model", "", language=language)
    assert state["voice_question"] == "" and "No speech" in state["voice_error"]


def test_voice_provenance_is_saved_and_exported_even_when_no_evidence(tmp_path):
    from app.db import Database
    from app.service import ask, export_feedback
    db = Database(tmp_path / "test.sqlite3")
    provider = ManualProvider("https://example.test", "", "model")
    context = {"mode": "voice", "transcript": "The motor does not start", "reviewed": True}
    result = ask(db, None, provider, "The motor does not start", input_context=context)
    with db.connect() as connection:
        stored = json.loads(connection.execute("SELECT input_context_json FROM queries WHERE id=?",
                                              (result["id"],)).fetchone()[0])
    assert stored == context and result["input_context"] == context
    assert "input_context_json" in export_feedback(db).decode("utf-8-sig")


def test_incomplete_opening_word_warns_without_blocking_or_silent_repair(monkeypatch):
    monkeypatch.setattr(interface, "detect_recording_language", lambda _: ("bn", 0.9))
    monkeypatch.setattr(interface, "transcribe_preferred", lambda *args: ("্রেসার ফুট ওঠে না", "Local"))
    state = {}
    interface.update_recording(b"audio", state, "model", "", provider=ManualProvider("https://example.test", "", "model"))
    assert state["voice_transcript"].startswith("্") and state["voice_warnings"]
    original, metadata = interface.voice_submission(state["voice_question"], state)
    assert original.startswith("্") and metadata["warnings"] and not metadata["reviewed"]
    question, metadata = interface.voice_submission("প্রেসার ফুট ওঠে না", state)
    assert question == "প্রেসার ফুট ওঠে না" and metadata["edited"]


def test_foreign_language_draft_is_used_for_search_and_original_is_kept(monkeypatch):
    monkeypatch.setattr(interface, "detect_recording_language", lambda _: ("hi", 0.9))
    monkeypatch.setattr(interface, "transcribe_preferred", lambda *args: ("मशीन शुरू नहीं हो रही है", "Local"))
    provider = SimpleNamespace(prepare_voice_question=lambda *args, **kwargs: ("মেশিন চালু হচ্ছে না", []))
    state = {}
    interface.update_recording(b"audio", state, "model", "", provider=provider)
    submitted, metadata = interface.voice_submission(state["voice_question"], state)
    assert submitted == "মেশিন চালু হচ্ছে না" and metadata["transcript"] == "मशीन शुरू नहीं हो रही है"
    assert not metadata["edited"] and metadata["submission_basis"] == "question_draft"
