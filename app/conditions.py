"""Conservative comparisons for explicit pitch conditions; no unit conversion."""

import re
from decimal import Decimal


def pitch_condition(question, source):
    # Do not compare an explicitly different unit against manual millimetres.
    if re.search(r'\b(?:inches?|cm)\b',question,re.I):
        return None
    if (len(re.findall(r'\d+(?:\.\d+)?\s*mm\b',question,re.I))>1 or
            re.search(r'\bpitch\s+\d+(?:\.\d+)?\s*(?:inches?|cm)\b',source,re.I)):
        return None
    query=re.search(r'\bpitch\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*mm\b',question,re.I)
    if not query:
        return None
    value=Decimal(query[1])
    comparisons=[]
    for match in re.finditer(r'\bpitch\s+(\d+(?:\.\d+)?)\s*(?:mm\s*)?or\s+(less|more)\b',source,re.I):
        bound=Decimal(match[1]);comparisons.append(value<=bound if match[2].lower()=='less' else value>=bound)
    for match in re.finditer(r'\b(more|less)\s+than\s+pitch\s+(\d+(?:\.\d+)?)',source,re.I):
        bound=Decimal(match[2]);comparisons.append(value>bound if match[1].lower()=='more' else value<bound)
    # Mixed conditions stay available: choosing a table column remains necessary.
    return any(comparisons) if comparisons else None
