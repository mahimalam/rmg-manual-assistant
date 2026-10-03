"""Local UI smoke check with real retrieval and mocked generation; no API calls."""

from unittest.mock import patch
import json
import io
import wave
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

from app.config import Settings
from app.db import Database, utc_now
from app.provider import ManualProvider, ProviderError
from app.structured import atomic_json


def run():
    settings = Settings.load()
    db = Database(settings.runtime / "study.sqlite3")
    query_id = None
    rejected_query_id = None
    continued_query_id = None
    voice_query_id = None

    def answer(_provider, _question, sources):
        row = next(index for index, source in enumerate(sources, 1)
                   if source.get("setting_id") == "function:39")
        note = next(index for index, source in enumerate(sources, 1)
                    if source.get("note_ids") == ["13"])
        return "প্রথমে Function 35 = 0, তারপর Function 39 = 10 দিন।", [row, note]

    audio = io.BytesIO()
    with wave.open(audio, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * 160)

    try:
        with patch.object(ManualProvider, "search_query", return_value=(
                "On the Brother S-7200A, set the presser foot lifter solenoid for "
                "fastest response and greatest operating noise; what prerequisite is required?")), \
                patch.object(ManualProvider, "answer", autospec=True, side_effect=answer) as generation, \
                patch("app.voice.speak", return_value=audio.getvalue()):
            app = AppTest.from_file(str(settings.root / "main.py"), default_timeout=90).run()
            assert not app.exception
            assert [tab.label for tab in app.tabs] == ["Ask the manuals", "Manual library", "Study workspace"]
            next(button for button in app.button if button.label == "Ask").click().run()
            assert any("Enter a question" in warning.value for warning in app.warning)
            assert generation.call_count == 0
            exact_question=json.loads((settings.root/'evaluation/adversarial_cases.json').read_text())[0]['question']
            app.text_area(key="question").set_value(exact_question).run()
            assert not next(button for button in app.button if button.label=='Ask').disabled
            next(button for button in app.button if button.label == "Ask").click().run()
            result = app.session_state["result"]
            query_id = result["id"]
            assert not app.exception and not app.error, [error.value for error in app.error]
            assert result["status"] == "answered"
            assert any(source.get("note_ids") == ["13"] for source in result["retrieved_sources"])
            assert any(item.label == "Sources and answer details" for item in app.expander)
            assert any(button.label == "Read answer aloud in Bangla" for button in app.button)
            next(button for button in app.button if button.label == "Read answer aloud in Bangla").click().run()
            assert app.session_state["answer_audio"][:4] == b"RIFF"
            app.radio(key=f"verdict-{query_id}").set_value("correct")
            app.text_area(key=f"feedback-{query_id}").set_value("Synthetic UI smoke check, not a human verdict.")
            app.button(key=f"save-feedback-{query_id}").click().run()
            app.run()
            assert not app.exception and not app.error and generation.call_count == 1
            with db.connect() as connection:
                assert connection.execute("SELECT verdict FROM feedback WHERE query_id=?", (query_id,)).fetchone()[0] == "correct"
            saved=json.loads((settings.runtime/'evaluations/knee_switch_root_cause.json').read_text())['original_record']
            def reject(provider,_question,_sources):
                provider.validation_issue='Synthetic incompatible diagnosis rejection'
                raise ProviderError('Diagnostic source check failed')
            generation.side_effect=reject
            with patch.object(ManualProvider,'search_query',return_value=saved['search_question']):
                app.text_area(key='question').set_value(saved['question']).run()
                next(button for button in app.button if button.label=='Ask').click().run()
                rejected=app.session_state['result'];rejected_query_id=rejected['id']
                assert not app.exception and not app.error
                assert rejected['status']=='clarify'and not rejected['error']and rejected['reasoning_path']=='source_rules'
                assert 'Function No. 40' in rejected['answer']and 'Function No. 12' in rejected['answer']
                assert any('derived from the retrieved manual setting rules'in caption.value for caption in app.caption)
                app.text_area(key=f"diagnostic-reply-{rejected_query_id}").set_value(
                    'DIP switch 1 = OFF; Function 12 = 0; Function 40 = 0').run()
                app.button(key=f"continue-{rejected_query_id}").click().run()
                continued=app.session_state['result'];continued_query_id=continued['id']
                assert not app.exception and not app.error and continued_query_id!=rejected_query_id
                assert continued['diagnostic_context']['parent_query_id']==rejected_query_id
                assert continued['diagnostic_context']['known_settings']=={
                    'dip:1':'OFF','function:12':'0','function:40':'0'}
                assert saved['question']in continued['diagnostic_context']['root_question']
                assert continued['reasoning_path']=='source_rules'and generation.call_count==1
            generation.side_effect = answer
            with patch("streamlit.audio_input", return_value=SimpleNamespace(getvalue=lambda: b"recorded audio")), \
                    patch("app.interface.speech_samples", return_value=None), \
                    patch("app.interface.detect_recording_language", return_value=("bn", 0.9)), \
                    patch("app.interface.transcribe_preferred", return_value=(exact_question, "Local")) as recognition, \
                    patch.object(ManualProvider, "search_query", return_value=(
                        "On the Brother S-7200A, set the presser foot lifter solenoid for "
                        "fastest response and greatest operating noise; what prerequisite is required?")):
                app.radio(key="input_mode").set_value("Voice").run()
                assert app.text_area(key="voice_question").value == exact_question
                assert not any(button.label == "Transcribe recording" for button in app.button)
                app.run()
                assert recognition.call_count == 1
                next(button for button in app.button if button.label == "Ask").click().run()
                voice_query_id = app.session_state["result"]["id"]
                assert not app.exception and not app.error and recognition.call_count == 1
                assert app.session_state["result"]["input_context"]["transcript"] == exact_question
                app.radio(key="input_mode").set_value("Type").run()
                assert app.text_area(key="question").value == saved["question"]
                app.radio(key="input_mode").set_value("Voice").run()
                assert app.text_area(key="voice_question").value == exact_question
            report = {"created_at": utc_now(), "passed": True, "new_api_calls": 0,
                      "checks": ["real library render", "Bengali input", "real reranked retrieval",
                                 "exact reported Bengali question with danda punctuation", "linked note display", "source preview", "mocked WAV playback control", "feedback save", "rerun without resubmission",
                                 "source-rule compound diagnosis displays conditional clarification",
                                 "follow-up preserves case/scope and records current setting values",
                                 "empty form rejected before provider call", "automatic voice transcription once",
                                 "direct voice submission without review checkbox", "voice input provenance saved",
                                 "voice form submission", "independent drafts survive mode switches"],
                      "generation": "ordinary answer mocked; recognised diagnostic cases solved locally", "engineering_validation": False}
            atomic_json(settings.runtime / "evaluations/dependency_ui_smoke.json", report)
            print(report)
    finally:
        if query_id or rejected_query_id:
            with db.connect() as connection:
                for identifier in (query_id,rejected_query_id,continued_query_id,voice_query_id):
                    if identifier:connection.execute("DELETE FROM queries WHERE id=?", (identifier,))


if __name__ == "__main__":
    run()
