"""Unicode handling shared by typed input, search and model identifiers."""

import re
import unicodedata


def contains_devanagari_letters(text):
    # Danda/double-danda are shared Indic punctuation, not script evidence.
    return any('\u0900' <= character <= '\u097f' and
               unicodedata.category(character).startswith('L') for character in text)


def normalize_question(text):
    text = unicodedata.normalize('NFC', text)
    text = ''.join(str(unicodedata.decimal(character)) if character.isdecimal() else character
                   for character in text)
    text = re.sub(r'(?<=[A-Za-z0-9])[‐‑‒–—−](?=[A-Za-z0-9])', '-', text)
    return re.sub(r'(?<=[A-Za-z0-9])[\u200b\u200c\u200d\ufeff](?=[A-Za-z0-9])', '', text)


def requested_english(text):
    if re.search(r'বাংলায়\s+উত্তর|উত্তর(?:টি)?\s+বাংলা', text):
        return False
    if re.search(r'ইংরেজিতে\s+(?:উত্তর|বলুন|ব্যাখ্যা)|উত্তর(?:টি)?\s+ইংরেজিতে', text):
        return True
    for match in re.finditer(r'\b(?:answer|respond|reply|explain)\s+(?:in|using)\s+English\b', text, re.I):
        if not re.search(r"(?:do not|don't|never|not)\s*$", text[:match.start()], re.I):
            return True
    return False


def model_ids(text):
    text = normalize_question(text).upper()
    identifiers = []
    pattern = r'(?<![A-Z0-9])([A-Z]{1,6})[- ]?(\d{3,5})([A-Z]{0,3})(?:-([A-Z0-9]{1,5}))?(?![A-Z0-9])'
    for match in re.finditer(pattern, text):
        if match[1] in ('ISO', 'IEC', 'DIN', 'MSW', 'CN'):
            continue
        before=text[max(0,match.start()-40):match.start()]
        after=text[match.end():match.end()+32]
        if match[1] in ('E','ERR','K') and (re.search(
                r'(?:ERROR(?:\s+CODE)?|FAULT(?:\s+CODE)?|ALARM|ত্রুটি|কোড|অ্যালার্ম)\s*[:=-]?\s*$',before) or
                re.match(r'^\s*(?:ERROR|FAULT|ALARM|ত্রুটি|কোড|অ্যালার্ম)(?:\s|[:!?।]|$)',after)):
            continue
        if match[1]=='NO' and re.search(r'(?:FUNCTION|PARAMETER|SETTING|ফাংশন|প্যারামিটার|সেটিং)\s*$',before):
            continue
        if re.match(r'^[A-Z]+\s',match[0]) and match[1] in (
                'AT','TO','FROM','FOR','ABOUT','WITH','OF','IN','IS','MAX','MIN','AND','OR','ON',
                'SPEED','RANGE','MODE','VALUE','THE','THIS','THAT','NEED'):
            continue  # Ordinary quantity phrases, e.g. "at 4500RPM".
        identifier = ''.join(value or '' for value in match.groups())
        identifiers.append(identifier)
        optional = re.match(r'\(([A-Z])\)', text[match.end():])
        if optional:
            identifiers.append(identifier[:-1] + optional[1] if match[3] else identifier + optional[1])
    return list(dict.fromkeys(identifiers))
