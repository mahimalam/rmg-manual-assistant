"""Split explicit independent event clauses without paid planning calls."""

import re


def audible_noise_question(question):
    if re.search(r'electrical|electromagnetic|\bEMI\b|welder|interference|বৈদ্যুতিক|হস্তক্ষেপ', question, re.I):
        return False
    return bool(re.search(r'(?:mak\w*|produc\w*|hear\w*|loud|audible).{0,60}\b(?:noise|sounds?)\b|আওয়াজ|আওয়াজ|সাউন্ড', question, re.I))


def electrical_noise_only(text):
    return bool(re.search(r'electrical\s+(?:noise|interference)', text, re.I)) and not re.search(
        r'abnormal\s+noises?|audible|mechanical\s+noise|sounds?|buzz\w*|humming|rattl\w*', text, re.I)


def unrelated_noise_setting(question, text):
    """A generic sound complaint does not request lifter timing configuration."""
    return (audible_noise_question(question) and
            not re.search(r'function|setting|DIP|solenoid|presser|foot|lifter|response|ফাংশন|সেটিং|ফুট|সোলেন', question, re.I) and
            bool(re.search(r'function\s+settings|function\s+(?:No\.?|number)\s*\d+', text, re.I)))


def question_parts(question):
    parts=re.split(r'(?<=[.!?;।])\s+|\n\s*\n|(?=\b(?:Additionally|Furthermore|Secondly)\b)|\s+and\s+(?=(?:after|once|the motor|the treadle|(?:the )?(?:sewing )?machine)\b|(?:mak\w*|produc\w*)\b[^.!?;]{0,50}\b(?:noise|sounds?)\b)|\s+(?:এবং|আর)\s+(?=(?:মেশিন|[^।.!?]{0,35}(?:আওয়াজ|আওয়াজ|সাউন্ড|শব্দ)))|\s*(?=এছাড়া|এছাড়াও|অন্যদিকে)',question,flags=re.I)
    event=re.compile(r'\b(?:foot|treadle|pedal|needle|thread|motor|oil|nozzle|temperature|voltage|error|alarm|noise)\b|ফুট|প্যাডেল|সুতা|হাঁটু|মোটর|ত্রুটি|আওয়াজ|আওয়াজ|সাউন্ড|শব্দ',re.I)
    parts=[p.strip() for p in parts if len(p.strip())>=15 and event.search(p)]
    # One query remains unchanged. More than four independent event clauses
    # need narrowing; silently dropping later symptoms would be unsafe.
    joined=[]
    for part in parts:
        if joined and re.match(r'(?:Previously\b|Before\b|আগে(?:\s|$)|প্যাডেল\s+তখন)',part,re.I):
            joined[-1]+=' '+part
        else:joined.append(part)
    if len(joined)>4:
        raise ValueError('More than four event clauses; narrow the question')
    return list(dict.fromkeys(joined)) if len(joined)>1 else [question]


def missing_event_terms(original, translated):
    terms=[(r'সুতা\s*কাট',r'thread\s*(?:trimm|cut)|cut\w*\s+(?:the\s+)?thread'),
           (r'হাঁটু',r'knee'),(r'নিউট্রাল',r'neutral'),(r'থামানো',r'stop\w*|stationary')]
    return [source for source,target in terms if re.search(source,original) and not re.search(target,translated,re.I)]


def event_trigger(question):
    if re.search(r'neutral|নিউট্রাল',question,re.I)and re.search(r'thread\s*(?:trimm|cut)|cut\w*\s+(?:the\s+)?thread|সুতা\s*কাট',question,re.I):
        return 'neutral_after_thread_trim'
    if re.search(r'knee|হাঁটু',question,re.I)and re.search(r'treadle|pedal|প্যাডেল',question,re.I)and re.search(r'after|once|following|এরপর|পরে|যদি',question,re.I):
        return 'after_knee_switch'
    return None


def diagnostic_question(question):
    return bool(re.search(r'\b(?:cause|causing|fault|symptoms?|cannot|incompatible|simultaneously|unresponsive)\b|'
                          r'\b(?:can|allows?|permits?)\b.{0,80}\b(?:rais\w*|lower\w*|lift\w*)\b|'
                          r'সমস্যা|একসঙ্গে|প্যাডেল.*(?:ওঠে|উঠছে|উপরে\s+তোলা)',question,re.I))
