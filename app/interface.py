"""State for the recorded-question interaction."""

from hashlib import sha256
import unicodedata

from app.language import normalize_question
from app.voice import transcribe_preferred, detect_recording_language, speech_samples, recognition_warnings


def _prepare_recording(state, provider):
    state.pop("voice_preparation_error", None)
    state["voice_question"] = ""
    state["voice_prepared_question"] = ""
    state["voice_warnings"] = recognition_warnings(state.get('voice_recognition', {}))
    usage_start = len(getattr(provider, "usage_log", []))
    raw_start = len(getattr(provider, "raw_responses", []))
    try:
        original = state["voice_transcript"]
        if provider is None:
            raise ValueError("Question preparation is not configured.")
        draft, warnings = provider.prepare_voice_question(original, language=state["voice_language"])
        state["voice_question"] = draft
        state["voice_prepared_question"] = draft
        state["voice_warnings"].extend(warnings)
        if unicodedata.category(original[0]).startswith('M'):
            state["voice_warnings"].append("The recording starts with an incomplete word. Correct the opening words or record again.")
    except Exception as exc:
        state["voice_preparation_error"] = str(exc)
        state["voice_question"] = state["voice_transcript"]
        state["voice_prepared_question"] = state["voice_transcript"]
    state["voice_preparation_usage"] = state.get("voice_preparation_usage", []) + list(
        getattr(provider, "usage_log", [])[usage_start:])
    state["voice_preparation_responses"] = state.get("voice_preparation_responses", []) + list(
        getattr(provider, "raw_responses", [])[raw_start:])


def update_recording(audio, state, model, api_key, retry=False, language="auto", provider=None, keyterms=()):
    """Transcribe once per completed recording, preserving subsequent user edits."""
    digest = sha256(language.encode() + b"\0" + audio).hexdigest()
    if state.get("recording_hash") == digest:
        if not retry:
            return False
        if state.get("voice_preparation_error") and state.get("voice_transcript"):
            _prepare_recording(state, provider)
            return True
    state["recording_hash"] = digest
    state["voice_language_mode"] = language
    state["voice_question"] = ""
    state.pop("result", None)
    state.pop("answer_audio", None)
    for key in ("voice_error", "voice_source", "voice_transcript", "voice_preparation_error",
                "voice_prepared_question", "voice_language", "voice_language_confidence",
                "voice_warnings", "voice_preparation_usage", "voice_preparation_responses", "voice_recognition"):
        state.pop(key, None)
    try:
        if language != "auto":
            speech_samples(audio)
        detected, probability = detect_recording_language(audio) if language == "auto" else (language, None)
        state["voice_language"] = detected
        state["voice_language_confidence"] = probability
        state['voice_recognition'] = {}
        transcript, source = transcribe_preferred(audio, model, api_key, detected, keyterms, state['voice_recognition'])
        state["voice_transcript"] = transcript
        state["voice_source"] = source
        _prepare_recording(state, provider)
    except Exception as exc:
        state["voice_error"] = str(exc)
    return True


def voice_submission(question, state):
    """Use the source when unchanged; the user's edited draft takes precedence."""
    if state.get("voice_error") or not state.get("voice_transcript"):
        raise ValueError("Record and prepare a usable question before asking.")
    if not question.strip() or len(question.strip()) > 2000:
        raise ValueError("Enter a question of 1–2,000 characters.")
    edited = normalize_question(question.strip()) != normalize_question(state["voice_prepared_question"].strip())
    if state.get("voice_preparation_error") and not edited and state["voice_language"] not in ('bn', 'en'):
        raise ValueError("Preparation failed. Correct the question manually or retry before asking.")
    use_original = not edited and state["voice_language"] in ('bn', 'en')
    submitted = state["voice_transcript"] if use_original else question.strip()
    return submitted, {"mode": "voice", "transcript": state["voice_transcript"],
        "bengali_draft": state["voice_prepared_question"], "question_draft": question.strip(),
        "submitted_question": submitted, "edited": edited, "reviewed": False,
        "submission_basis": "source_transcript" if use_original else "question_draft",
        "language": state["voice_language"], "language_confidence": state["voice_language_confidence"],
        "language_mode": state["voice_language_mode"],
        "recognizer": state["voice_source"], "warnings": state.get("voice_warnings", []),
        "recognition": state.get('voice_recognition', {}),
        "preparation_error": state.get("voice_preparation_error"),
        "preparation_responses": state.get("voice_preparation_responses", []),
        "preparation_usage": state.get("voice_preparation_usage", [])}
