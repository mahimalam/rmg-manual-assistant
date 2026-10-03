"""Reproduce the local decoder comparison using the cached labeled human clips."""

import json
import statistics
import time

from app.config import Settings
from app.db import utc_now
from app.structured import atomic_json
from app.voice import _whisper
from evaluation.voice_preparation import word_error_rate


def run():
    settings = Settings.load()
    folder = settings.runtime / 'evaluations' / 'speech-repair'
    clips = sorted(folder.glob('bn-*.wav'))
    if not clips:
        raise ValueError('Labeled audio cache is missing; see SPEECH_RAG_REPAIR_REPORT.md')
    modes = [('baseline', settings.whisper_model, False), ('fixed', settings.whisper_model, True)]
    candidate = settings.runtime / 'models' / 'mozilla-bengali-ct2'
    if (candidate / 'tokenizer.json').exists():
        modes.append(('candidate', str(candidate), True))
    report = {'created_at': utc_now(), 'cases': [], 'summaries': {},
              'dataset': 'https://huggingface.co/datasets/imonghose/bengali-asr-data',
              'split': 'test', 'offsets': [0, 2, 3, 4, 5, 6], 'independent_test': False,
              'limits': ['Six development clips, not engineering speech or user microphone tests',
                         'Dataset/model training overlap has not been established',
                         'WER depends on the documented normalization in voice_preparation.words']}
    for name, model_name, without_timestamps in modes:
        model = _whisper(model_name)
        records = []
        for clip in clips:
            reference = json.loads(clip.with_suffix('.json').read_text())['reference']
            started = time.monotonic()
            segments, _ = model.transcribe(str(clip), language='bn', task='transcribe',
                beam_size=5, vad_filter=True, without_timestamps=without_timestamps)
            transcript = ' '.join(segment.text.strip() for segment in segments)
            record = {'configuration': name, 'clip': clip.stem, 'reference': reference,
                      'transcript': transcript, 'wer': word_error_rate(reference, transcript),
                      'seconds': round(time.monotonic() - started, 3)}
            records.append(record)
            report['cases'].append(record)
            print(name, clip.stem, record['wer'], flush=True)
        report['summaries'][name] = {'macro_wer': statistics.mean(r['wer'] for r in records),
            'median_seconds': statistics.median(r['seconds'] for r in records)}
        atomic_json(folder / 'reproducible-comparison.json', report)
    return report


if __name__ == '__main__':
    run()
