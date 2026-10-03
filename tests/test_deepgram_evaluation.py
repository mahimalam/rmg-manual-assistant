"""Speech evaluation reservations share the existing paid-test allowance."""

import json
import wave

import pytest

from app.config import Settings
from app.provider import ProviderError
from app.structured import atomic_json
from evaluation.budget import RunBudget
from evaluation.deepgram_speech import reserve
from evaluation import deepgram_speech


def test_speech_cannot_exceed_existing_ai_allowance(tmp_path):
    s = Settings(runtime=tmp_path)
    atomic_json(tmp_path / 'evaluations/adversarial_api_ledger.json',
                {'limit_usd': 20, 'attempts': [{'reserved_usd': 19.95}]})
    reserve(s, .03, .75)
    with pytest.raises(ValueError, match='allowance'):
        reserve(s, .03, .75)
    assert json.loads((tmp_path / 'evaluations/deepgram_speech_budget.json').read_text())['requests'] == 1


def test_ai_requests_count_speech_reservations(tmp_path, monkeypatch):
    monkeypatch.setattr(RunBudget, 'usage', lambda self: {'balance': 100, 'actual_cost': 0})
    s = Settings(runtime=tmp_path)
    budget = RunBudget(s, limit=.03)
    reserve(s, .025, .75)
    with pytest.raises(ProviderError, match='allowance'):
        budget.reserve({'messages': [], 'max_tokens': 1})


def test_plan_missing_key_and_cloud_failure_never_measure_local_fallback(tmp_path, monkeypatch):
    manifest = tmp_path / 'speech.json'
    with wave.open(str(tmp_path / 'recording.wav'), 'wb') as audio:
        audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(16000)
        audio.writeframes(b'\0\0' * 16000)
    manifest.write_text(json.dumps([{'audio': 'recording.wav', 'reference': 'ফুট ওঠে না'}]))
    monkeypatch.setattr(Settings, 'load', lambda: Settings(runtime=tmp_path))
    assert deepgram_speech.run(manifest)['cases'] == []
    with pytest.raises(ValueError, match='DEEPGRAM_API_KEY'):
        deepgram_speech.run(manifest, live=True)
    monkeypatch.setattr(Settings, 'load', lambda: Settings(runtime=tmp_path, deepgram_api_key='mock-key'))
    monkeypatch.setattr(deepgram_speech, 'speech_samples', lambda audio: None)
    def unavailable(*args):
        raise ValueError('Unusable cloud transcript')
    monkeypatch.setattr(deepgram_speech, 'transcribe_deepgram', unavailable)
    with pytest.raises(ValueError, match='No local fallback'):
        deepgram_speech.run(manifest, live=True)
    report = json.loads((tmp_path / 'evaluations/deepgram_speech.json').read_text())
    assert report['cases'][0]['status'] == 'failed' and 'macro_wer' not in report
