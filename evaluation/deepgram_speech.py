"""Plan or explicitly run a bounded, cloud-only speech evaluation."""

import argparse
import json
import re
import time
import wave
from pathlib import Path

from app.config import Settings
from app.db import Database, utc_now
from app.language import normalize_question
from app.structured import atomic_json, manual_lock
from app.voice import manual_keyterms, speech_samples, transcribe_deepgram
from evaluation.voice_preparation import word_error_rate


def reserve(settings, estimate, max_usd):
    path = settings.runtime / 'evaluations/deepgram_speech_budget.json'
    with manual_lock(settings.runtime, 'adversarial-api-budget'):
        ledger = json.loads(path.read_text()) if path.exists() else {'reserved_usd': 0, 'requests': 0}
        old_path = settings.runtime / 'evaluations/adversarial_api_ledger.json'
        old = json.loads(old_path.read_text()) if old_path.exists() else {'limit_usd': 20, 'attempts': []}
        accounted = sum(a.get('estimated_usd', a['reserved_usd']) for a in old['attempts'])
        if ledger['reserved_usd'] + estimate > max_usd or accounted + ledger['reserved_usd'] + estimate > old['limit_usd']:
            raise ValueError('Speech evaluation allowance would be exceeded; no request sent.')
        ledger.update(reserved_usd=ledger['reserved_usd'] + estimate, requests=ledger['requests'] + 1)
        atomic_json(path, ledger)


def run(manifest=None, live=False, max_usd=.75):
    settings = Settings.load()
    db = Database(settings.runtime / 'study.sqlite3')
    if manifest:
        manifest = Path(manifest).resolve()
        entries = json.loads(manifest.read_text())
        folder = manifest.parent
    else:
        folder = settings.runtime / 'evaluations/speech-repair'
        entries = [{'id': p.stem, 'audio': p.name, **json.loads(p.with_suffix('.json').read_text())}
                   for p in sorted(folder.glob('bn-*.wav'))]
    if not isinstance(entries, list) or not 1 <= len(entries) <= 100:
        raise ValueError('Supply a JSON list of 1–100 labeled WAV recordings.')
    planned = []
    for entry in entries:
        path = (folder / entry['audio']).resolve()
        with wave.open(str(path)) as audio:
            seconds = audio.getnframes() / audio.getframerate()
        if not 0 < seconds <= 120 or not isinstance(entry['reference'], str) or not entry['reference'].strip():
            raise ValueError('Each recording needs nonempty reference text and 0–120 seconds of WAV audio.')
        spans = entry.get('critical_spans', [])
        if not isinstance(spans, list) or any(not isinstance(span, str) or not span.strip() for span in spans):
            raise ValueError('critical_spans must be a list of nonempty reference phrases.')
        planned.append({**entry, 'path': path, 'seconds': seconds,
                        'reserved_usd': max(seconds, 1) / 60 * .01})
    report = {'created_at': utc_now(), 'live': live, 'deepgram_configured': bool(settings.deepgram_api_key),
              'planned_recordings': len(planned), 'planned_usd': sum(e['reserved_usd'] for e in planned),
              'cases': [], 'limits': ['Development evaluation unless references form an untouched held-out set',
              'Confidence and exact text spans do not prove semantic correctness',
              'USD 0.01/min reservations are estimates, not invoices; verify current provider pricing']}
    output = settings.runtime / 'evaluations/deepgram_speech.json'
    atomic_json(output, report)
    if not live:
        return report
    if not settings.deepgram_api_key:
        raise ValueError('A valid DEEPGRAM_API_KEY is required; no request sent.')
    if report['planned_usd'] > max_usd:
        raise ValueError('Planned recordings exceed the evaluation allowance; no request sent.')
    for entry in planned:
        audio = entry['path'].read_bytes()
        speech_samples(audio)
        scope = entry.get('manual_ids', [m['id'] for m in db.manuals()])
        terms = manual_keyterms(db, scope)
        reserve(settings, entry['reserved_usd'], max_usd)
        started = time.monotonic(); details = {}
        # No fallback: the report must measure Deepgram rather than local speech.
        try:
            transcript = transcribe_deepgram(audio, settings.deepgram_api_key, entry.get('language', 'bn'), terms, details)
        except Exception as exc:
            report['cases'].append({'id': entry.get('id', entry['path'].stem), 'status': 'failed',
                                    'error_type': type(exc).__name__,
                                    'http_status': getattr(getattr(exc, 'response', None), 'status_code', None),
                                    'new_request_reserved': True})
            atomic_json(output, report)
            raise ValueError('Cloud evaluation failed; inspect the saved report. No local fallback was measured.') from None
        spans = entry.get('critical_spans', [])
        report['cases'].append({'id': entry.get('id', entry['path'].stem), 'reference': entry['reference'],
            'transcript': transcript, 'wer': word_error_rate(entry['reference'], transcript),
            'critical_spans_found': {span: bool(re.search(r'(?<!\w)' + re.escape(normalize_question(span).casefold()) + r'(?!\w)', normalize_question(transcript).casefold())) for span in spans},
            'seconds': round(time.monotonic() - started, 3), 'recognition': details})
        report['macro_wer'] = sum(e['wer'] for e in report['cases']) / len(report['cases'])
        atomic_json(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--live', action='store_true', help='Send labeled recordings to Deepgram; no local fallback.')
    args = parser.parse_args()
    report = run(args.manifest, args.live)
    print({k: report[k] for k in ('live', 'deepgram_configured', 'planned_recordings', 'planned_usd')})
