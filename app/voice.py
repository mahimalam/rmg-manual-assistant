"""Source-language speech recognition with a Bengali-specialist local fallback."""

import io
import wave
import unicodedata
import json
import re
from functools import lru_cache

import numpy as np
import httpx
from app.model_loading import serialized_model_load


@lru_cache(maxsize=2)
def _whisper(model_name):
    from faster_whisper import WhisperModel

    return WhisperModel(model_name, device="cpu", compute_type="int8")


def _check_audio(audio: bytes) -> None:
    if not audio:
        raise ValueError('No recording was received')
    if len(audio) > 20 * 1024 * 1024:
        raise ValueError("Recording exceeds 20 MB")


def _validated_transcript(text: str, language="bn") -> str:
    if not isinstance(text, str):
        raise ValueError('Speech response did not contain a transcript.')
    text = text.strip()
    if not text or '\ufffd' in text or not any(character.isalpha() for character in text):
        raise ValueError('No usable speech was recognized. Record again or type the question.')
    if language in ("bn", "en") and any(character.isalpha() and not ('\u0980' <= character <= '\u09ff' or
           'LATIN' in unicodedata.name(character, '')) for character in text):
        raise ValueError('Recognition returned letters outside Bengali or English. Record again or type the question.')
    return text


def speech_samples(audio: bytes):
    """Reject invalid/silent recordings before any paid recognition or translation."""
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import get_speech_timestamps

    _check_audio(audio)
    samples = decode_audio(io.BytesIO(audio), sampling_rate=16000)
    if not get_speech_timestamps(samples):
        raise ValueError("No speech was detected. Record again or type the question.")
    return samples


def detect_recording_language(audio: bytes):
    """Identify the source, without using Deepgram's Bengali-excluding detection list."""
    samples = speech_samples(audio)
    language, probability, _ = _whisper("medium").detect_language(samples, vad_filter=True)
    return language, float(probability)


def transcribe(audio: bytes, model_name="SayedShaun/bengali-whisper-medium-ct2", language="bn") -> str:
    _check_audio(audio)
    segments, _ = _whisper(model_name if language == "bn" else "medium").transcribe(
        io.BytesIO(audio), language=language, task="transcribe", beam_size=5, vad_filter=True,
        without_timestamps=True
    )
    return _validated_transcript(" ".join(segment.text.strip() for segment in segments), language)


def bounded_keyterms(terms):
    # ponytail: a conservative UTF-8 byte cap avoids depending on a proprietary
    # tokenizer; use its tokenizer if larger vocabularies become necessary.
    result, size = [], 0
    for term in terms:
        if not isinstance(term, str):
            continue
        term = ' '.join(term.split())
        if not term or len(term) > 80 or re.search(r'[,;:]', term) or term.casefold() in {t.casefold() for t in result}:
            continue
        cost = len(term.encode('utf-8')) + 1
        if size + cost > 480:
            continue
        result.append(term); size += cost
        if len(result) == 32:
            break
    return result


def manual_keyterms(db, manual_ids):
    """Use the current ready library, never function values or expected answers."""
    manuals = [dict(m) for m in db.manuals() if m['id'] in manual_ids and m['status'] == 'ready']
    if not manuals:
        return []
    ids = [m['id'] for m in manuals]
    text = ' '.join(' '.join(c['text'].casefold().split()) for c in db.chunks(ids))
    terms = []
    for manual in manuals:
        title = manual['title']
        terms.extend(re.findall(r'\b[A-Z]{1,12}-?\d{3,}[A-Z0-9]*(?:-[A-Z0-9]+)*\b', title))
        terms.extend(w for w in re.findall(r'\b[A-Za-z]{2,}\b', title)
                     if w.casefold() not in {'service', 'manual', 'manuals', 'instruction', 'instructions', 'english'}
                     and not re.search(r'\b' + re.escape(w) + r'-?\d', title))
    aliases = {'Brother': 'ব্রাদার', 'JUKI': 'জুকি', 'presser foot': 'প্রেসার ফুট', 'rotary hook': 'রোটারি হুক',
               'knee switch': 'হাঁটু সুইচ', 'solenoid': 'সোলেনয়েড',
               'DIP switch': 'ডিপ সুইচ', 'thread take-up spring': 'থ্রেড টেক-আপ স্প্রিং',
               'thread tension': 'সুতার টান', 'feed dog': 'ফিড ডগ',
               'needle bar': 'নিডল বার', 'thread trimming': 'সুতা কাটা', 'treadle': 'প্যাডেল'}
    for english, bengali in aliases.items():
        if english.casefold() in text:
            terms.extend((english, bengali))
    headings = []
    for manual_id in ids:
        for row in db.units(manual_id):
            unit = json.loads(row['payload_json'])
            if unit.get('blocking_issues'):
                continue
            heading = re.sub(r'^\d+(?:-\d+)*\.?\s*', '', unit.get('title', '')).lower()
            action = re.match(r'^(?:adjust(?:ing|ment of)?|check(?:ing)?|set(?:ting)?|replac(?:ing|ement of)|install(?:ing|ation of)?)\s+(?:the\s+)?', heading)
            if not action:
                continue
            heading = heading[action.end():]
            if re.fullmatch(r'[a-z][a-z -]+', heading) and 2 <= len(heading.split()) <= 5 and heading != 'manual section':
                headings.append(heading)
    terms.extend(dict.fromkeys(headings))
    return bounded_keyterms(terms)


def recognition_warnings(details):
    uncertain = []
    for word in details.get('words', []):
        if not isinstance(word, dict):
            continue
        text, confidence = word.get('word', ''), word.get('confidence')
        if (isinstance(text, str) and type(confidence) in (int, float) and 0 <= confidence < 0.7 and
                (any(c.isdecimal() for c in text) or re.fullmatch(r"(?i)(?:on|off|not|no|cannot|can't|don't|doesn't|won't|না|নয়|নয়|নাই|অন|অফ)[.!?।,]*", text))):
            uncertain.append(text)
    if not uncertain:
        return []
    return ['Recognition is uncertain about these numbers or negatives: ' + ', '.join(dict.fromkeys(uncertain)) +
            '. Correct them if needed before Ask. Confidence does not prove transcript correctness.']


def transcribe_deepgram(audio: bytes, api_key: str, language="bn", keyterms=(), details=None) -> str:
    _check_audio(audio)
    params = httpx.QueryParams({'model': 'nova-3', 'language': language, 'punctuate': 'true'})
    for term in bounded_keyterms(keyterms):
        params = params.add('keyterm', term)
    response = httpx.post(
        "https://api.deepgram.com/v1/listen",
        params=params,
        headers={"Authorization": f"Token {api_key}", "Content-Type": "audio/wav"},
        content=audio, timeout=httpx.Timeout(60.0, connect=5.0),
    )
    response.raise_for_status()
    body = response.json()
    alternative = body['results']['channels'][0]['alternatives'][0]
    transcript = _validated_transcript(alternative['transcript'], language)
    if details is not None:
        details.update(request_id=body.get('metadata', {}).get('request_id'),
                       confidence=alternative.get('confidence'), words=alternative.get('words') if isinstance(alternative.get('words'), list) else [],
                       keyterms=bounded_keyterms(keyterms))
    return transcript


def transcribe_preferred(audio: bytes, model_name: str, deepgram_api_key: str, language="bn", keyterms=(), details=None) -> tuple[str, str]:
    _check_audio(audio)
    if deepgram_api_key:
        try:
            return transcribe_deepgram(audio, deepgram_api_key, language, keyterms, details), "Deepgram"
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            if details is not None:
                details.clear()
                details['fallback_reason'] = ('Deepgram rejected the credential.' if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (401, 403)
                                              else 'Deepgram was unavailable or returned unusable speech.')
            return transcribe(audio, model_name, language), "Local fallback"
    return transcribe(audio, model_name, language), "Local"


@serialized_model_load
@lru_cache(maxsize=1)
def _tts():
    from transformers import AutoTokenizer, VitsModel

    name = "facebook/mms-tts-ben"
    return AutoTokenizer.from_pretrained(name), VitsModel.from_pretrained(name)


def speak(text: str) -> bytes:
    import torch

    tokenizer, model = _tts()
    with torch.no_grad():
        waveform = model(**tokenizer(text, return_tensors="pt")).waveform[0]
    samples = np.clip(waveform.cpu().numpy(), -1, 1)
    samples = (samples * 32767).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(model.config.sampling_rate)
        wav.writeframes(samples.tobytes())
    return output.getvalue()
