"""Run bounded live text preparation and labeled public speech checks.

Usage: python -m evaluation.voice_preparation [--live]
No answer generation or image extraction. Live mode sends six text questions.
"""

import argparse
import json
import re
import time

import httpx

from app.config import Settings
from app.db import utc_now
from app.language import normalize_question
from app.provider import ManualProvider, ProviderError
from app.structured import atomic_json
from app.voice import detect_recording_language, transcribe_preferred


CASES = [
    ("english_negation", "en", "S-7200A: After using the knee switch, the pedal cannot raise the presser foot while the machine is stopped."),
    ("numeric_subclass", "en", "For S-7200A-453, after running for 8 seconds, how wide in mm should the oil band be?"),
    ("binding_and_polarity", "en", "S-7200A: Function No. 40 = 0 and DIP switch 1 = OFF. After knee switch use the pedal cannot raise the presser foot. Why?"),
    ("uncertain_intermittent", "en", "S-7200A: Sometimes the motor does not start. Maybe the setting changed, but I am uncertain. What should I check?"),
    ("genuine_hindi", "hi", "S-7200A मशीन शुरू नहीं हो रही है। क्या जांचना चाहिए?"),
    ("compound_sequence", "en", "S-7200A: After thread trimming the foot drops when the pedal returns to neutral. After using the knee switch the pedal can no longer raise the foot while the machine is stopped. Previously it worked. Which settings explain these symptoms?"),
]


def words(text):
    return re.sub(r'[^\w\s]', '', normalize_question(text).casefold()).split()


def word_error_rate(reference, hypothesis):
    expected, actual = words(reference), words(hypothesis)
    row = list(range(len(actual) + 1))
    for i, word in enumerate(expected, 1):
        following = [i]
        for j, heard in enumerate(actual, 1):
            following.append(min(following[-1] + 1, row[j] + 1, row[j-1] + (word != heard)))
        row = following
    return round(row[-1] / max(1, len(expected)), 4)


def run(live=False, text_only=False):
    settings = Settings.load()
    report_path = settings.runtime / "evaluations" / ("voice_preparation_text.json" if text_only else "voice_preparation.json")
    report = {"created_at": utc_now(), "live_text_provider": live,
              "live_deepgram": False, "text_cases": [], "audio_cases": [],
              "limits": ["Small development set, not independent engineering validation",
                         "Language confidence is not transcript correctness",
                         "Mocked contracts separately covered by pytest",
                         "No user microphone or accented engineering recording measured"]}
    if live:
        for name, language, transcript in CASES:
            provider = ManualProvider(settings.base_url, settings.api_key, settings.model)
            case = {"name": name, "original": transcript, "language": language}
            started = time.monotonic()
            try:
                draft, warnings = provider.prepare_voice_question(transcript, language)
                case.update(status="prepared", draft=draft, warnings=warnings)
            except ProviderError as exc:
                case.update(status="blocked_for_review", error=str(exc))
            case.update(elapsed_seconds=round(time.monotonic()-started, 3), usage=provider.usage_log,
                        raw_responses=provider.raw_responses)
            report["text_cases"].append(case)
            atomic_json(report_path, report)
            print(name, case['status'], flush=True)

    report['prepared_text_cases'] = sum(c['status'] == 'prepared' for c in report['text_cases'])
    if text_only:
        atomic_json(report_path, report)
        return report

    # Public labeled clips; do not save user microphone recordings.
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        recordings = []
        for offset in (0, 2):
            response = client.get('https://datasets-server.huggingface.co/rows', params={
                'dataset':'imonghose/bengali-asr-data', 'config':'default', 'split':'test',
                'offset':offset, 'length':1})
            response.raise_for_status()
            row = response.json()['rows'][0]['row']
            response = client.get(row['audio'][0]['src']); response.raise_for_status()
            reference = row.get('sentence') or row.get('text') or row.get('transcription')
            if not reference:
                raise ValueError("Labeled Bengali clip has no reference transcript")
            recordings.append((f'bengali_{offset}', 'bn', reference, response.content,
                               'https://huggingface.co/datasets/imonghose/bengali-asr-data'))
        url = 'https://raw.githubusercontent.com/SYSTRAN/faster-whisper/master/tests/data/jfk.flac'
        response = client.get(url); response.raise_for_status()
        recordings.append(('english_jfk', 'en',
            'And so my fellow Americans ask not what your country can do for you ask what you can do for your country',
            response.content, 'https://github.com/SYSTRAN/faster-whisper/tree/master/tests/data'))
    for name, expected, reference, audio, source in recordings:
        started = time.monotonic()
        language, probability = detect_recording_language(audio)
        transcript, recognizer = transcribe_preferred(audio, settings.whisper_model, '', language)
        case = {'name':name, 'reference':reference, 'transcript':transcript,
                'expected_language':expected, 'detected_language':language,
                'language_confidence':probability, 'language_matched':language == expected,
                'word_error_rate':word_error_rate(reference, transcript), 'recognizer':recognizer,
                'source':source, 'elapsed_seconds':round(time.monotonic()-started, 3)}
        report['audio_cases'].append(case)
        atomic_json(report_path, report)
        print(name, 'language:', language, 'WER:', case['word_error_rate'], flush=True)
    report['passed_language_checks'] = all(c['language_matched'] for c in report['audio_cases'])
    report['prepared_text_cases'] = sum(c['status'] == 'prepared' for c in report['text_cases'])
    atomic_json(report_path, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--text-only', action='store_true')
    arguments = parser.parse_args()
    report = run(arguments.live, arguments.text_only)
    print('Report saved; prepared text cases:', report['prepared_text_cases'])
