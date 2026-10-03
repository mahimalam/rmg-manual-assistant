from app import interface
from app.provider import ManualProvider


def test_recording_is_processed_once_and_keeps_user_edits(monkeypatch):
    calls = []
    def recognize(audio, model, key, language, *extra):
        assert language == "bn"
        calls.append(audio)
        return "মেশিনের প্রশ্ন", "Local"
    monkeypatch.setattr(interface, "transcribe_preferred", recognize)
    monkeypatch.setattr(interface, "detect_recording_language", lambda _: ("bn", 0.9))
    provider = ManualProvider("https://example.test", "", "model")
    state = {}
    assert interface.update_recording(b"first", state, "model", "key", provider=provider)
    state["voice_question"] = "সংশোধিত প্রশ্ন"
    assert not interface.update_recording(b"first", state, "model", "key")
    assert state["voice_question"] == "সংশোধিত প্রশ্ন" and calls == [b"first"]
    assert interface.update_recording(b"second", state, "model", "key", provider=provider)
    assert calls == [b"first", b"second"]


def test_failed_recording_clears_stale_draft_and_requires_explicit_retry(monkeypatch):
    calls = []
    def recognize(audio, model, key, language, *extra):
        calls.append(audio)
        if len(calls) == 1:
            raise ValueError("No speech")
        return "নতুন প্রশ্ন", "Deepgram"
    monkeypatch.setattr(interface, "transcribe_preferred", recognize)
    monkeypatch.setattr(interface, "detect_recording_language", lambda _: ("bn", 0.9))
    provider = ManualProvider("https://example.test", "", "model")
    state = {"voice_question": "আগের প্রশ্ন", "voice_source": "Local"}
    interface.update_recording(b"audio", state, "model", "key")
    assert state["voice_question"] == "" and state["voice_error"] == "No speech"
    assert "voice_source" not in state
    assert not interface.update_recording(b"audio", state, "model", "key")
    interface.update_recording(b"audio", state, "model", "key", retry=True, provider=provider)
    assert state["voice_question"] == "নতুন প্রশ্ন" and "voice_error" not in state
    assert len(calls) == 2
