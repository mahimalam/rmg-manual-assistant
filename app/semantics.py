"""Source-derived note, setting and applicability relationships; no answer keys."""

import re
from collections import defaultdict


def normalize_pattern(pattern):
    return re.sub(r"\[\s*\]|[□▯?]", "?", pattern).upper().replace(" ", "")


def model_patterns(text):
    text = re.sub(r"\[\s*\]|[□▯]", "?", text)
    patterns = re.findall(r"(?:-|/\s*)([0-9][0-9A-Z?]{1,5})(?![0-9A-Z?])", text, re.I)
    # Restrict to explicit model/subclass notation, not dimensions or page ranges.
    result = ["-" + item.upper() for item in patterns if "?" in item]
    for match in re.finditer(r"\bsub[- ]?class\s+(-[0-9A-Z?]{2,5})\b", text, re.I):
        result.append(match[1].upper())
    for group in re.findall(r"\((-[0-9A-Z? /]+)\)", text, re.I):
        result.extend("-" + item.upper() for item in
                      re.findall(r"(?:-|/\s*)([0-9][0-9A-Z?]{1,4})", group, re.I))
    return list(dict.fromkeys(result))


def matches_model(pattern, subclass):
    regex = "".join("[A-Z0-9]" if char == "?" else re.escape(char)
                    for char in normalize_pattern(pattern))
    return re.fullmatch(regex, subclass.upper()) is not None


def extract_subclasses(question):
    question = question.translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789"))
    values = re.findall(r"\b[A-Z]{1,6}-?\d{3,5}[A-Z]?-([0-9A-Z]{2,5})\b", question, re.I)
    values += re.findall(r"\bsub[- ]?class\s+[-\"']?([0-9A-Z]{2,5})\b", question, re.I)
    return list(dict.fromkeys("-" + value.upper() for value in values))


def heading_key(text):
    text = " ".join(text.split())
    match = re.match(r"^(\d+(?:[-.]\d+)*)(?:\.)?\s+([A-Za-z].*)", text)
    if not match or len(text) > 180:
        return None
    letters = re.findall(r"[A-Za-z]", match[2])
    if "-" not in match[1] and (not letters or sum(c.isupper() for c in letters) / len(letters) < .8):
        return None
    return match[1].rstrip(".")


def merge_text_blocks(blocks):
    """Join continuing sentences within one column; keep a union source box."""
    output = []
    for raw in blocks:
        if len(raw) < 7 or raw[6] != 0 or not raw[4].strip():
            continue
        block = list(raw)
        merged = False
        for previous in reversed(output[-8:]):
            gap = block[1] - previous[3]
            aligned = abs(block[0] - previous[0]) < 14
            warning = re.fullmatch(r"\s*(?:WARNING|CAUTION|DANGER)\s*:?\s*", previous[4])
            incomplete = not re.search(r"[.!?:]\s*$", previous[4])
            # Numerical labels and headings are not sentence continuations.
            prose = len(previous[4].split()) >= 5 or warning
            starts_prose = re.match(r"\s*[a-z(・•]", block[4]) or warning
            if (-2 <= gap <= 20 and (aligned or warning) and prose and starts_prose and
                    incomplete and not heading_key(block[4]) and
                    not re.match(r"\s*\(?NOTE\s+\d+", block[4], re.I)):
                previous[:4] = [min(previous[0], block[0]), min(previous[1], block[1]),
                                max(previous[2], block[2]), max(previous[3], block[3])]
                previous[4] = previous[4].rstrip() + "\n" + block[4].lstrip()
                merged = True
                break
        if not merged:
            output.append(block)
    return output


def note_blocks(lines):
    """Reconstruct footer notes from line positions, including indented wraps."""
    lines = sorted(lines, key=lambda line: (round(line[1], 1), line[0]))
    output = []
    for index, line in enumerate(lines):
        if not re.match(r"\s*\(NOTE\s+\d+\)\s+\S", line[4], re.I):
            continue
        box, text, bottom = list(line[:4]), line[4].strip(), line[3]
        for following in lines[index + 1:]:
            gap = following[1] - bottom
            if gap > 12:
                break
            if gap < -2:  # Another column on the same baseline.
                continue
            if re.match(r"\s*\(NOTE\s+\d+\)", following[4], re.I) or heading_key(following[4]):
                break
            if following[0] < line[0] - 8 or following[0] > line[0] + 120:
                continue
            if re.fullmatch(r"[\d. –-]+", following[4].strip()):
                break
            text += "\n" + following[4].strip()
            box = [min(box[0], following[0]), min(box[1], following[1]),
                   max(box[2], following[2]), max(box[3], following[3])]
            bottom = following[3]
        output.append([*box, text, 0, 0])
    return output


def note_numbers(text):
    result = []
    for match in re.finditer(r"\bNOTES?\s+(\d+(?:\s*(?:,|and|&|to|[-–])\s*\d+)*)", text, re.I):
        value = match[1]
        numbers = re.findall(r"\d+", value)
        if len(numbers) == 2 and re.search(r"to|[-–]", value):
            start, end = map(int, numbers)
            if end >= start and end - start <= 50:
                numbers = [str(number) for number in range(start, end + 1)]
        result.extend(numbers)
    for match in re.finditer(r'(?<![\d*])\*\s*(\d{1,2})\b', text):
        if not re.match(r'\s*(?:mm|cm|kg|ms|rpm|sti/min)\b', text[match.end():], re.I):
            result.append('star:' + match[1])
    return list(dict.fromkeys(result))


def _requirements(text):
    pattern = (r"(Function\s+No\.?\s*(\d+)|DIP\s+switch\s*(\d+))\s+"
               r"(?:is\s+|must\s+be\s+)?(?:set|turned)\s+(?:to\s+)?[\"']?([\d.]+|ON|OFF)\b")
    requirements = []
    for match in re.finditer(pattern, text, re.I):
        before = text[:match.start()].rsplit('\n', 1)[-1]
        after = text[match.end():]
        # An exclusion or a range/OR condition cannot become a mandatory
        # single-value prerequisite. Its full wording remains in linked notes.
        if re.search(r'\bdisabled\b|\bnot\s+(?:enabled|drop)\b', before, re.I):
            continue
        if re.match(r'[\"\']?\s*(?:or|to|[-–])\s*[\"\']?[\d.]', after, re.I):
            continue
        requirements.append({"setting_id": f"function:{match[2]}" if match[2] else f"dip:{match[3]}",
                             "value": match[4].upper()})
    return requirements


def _setting_exclusions(unit, note):
    """A negative DIP condition excludes the matching positive function branch.

    Only the same short predicate with opposite polarity is recognized. Other
    natural-language relationships stay in the note rather than becoming rules.
    """
    if not (unit.get('setting_id') or '').startswith('function:'):
        return []
    match=re.search(r'([^.!?\n]+?)\s+(?:will|does|can)\s+not\s+(.+?)\s+if\s+DIP\s+switch\s+(\d+)\s+is\s+set\s+to\s+(ON|OFF)\b',note,re.I)
    if not match:
        return []
    def words(text):
        return [w[:-1] if w.endswith('s') and len(w)>3 else w
                for w in re.findall(r'[a-z]+',text.lower())]
    subject=re.sub(r'^.*?\)\s*','',match[1])
    predicate=words(subject+' '+match[2])
    exclusions=[]
    for line in unit['text'].splitlines():
        branch=re.match(r'\s*(\d+)\s*:\s*(.*)',line)
        if branch and words(re.sub(r'\(.*','',branch[2]))==predicate:
            exclusions.append({'when_value':branch[1],'setting_id':'dip:'+match[3],'value':match[4].upper()})
    return exclusions


def link_semantics(units):
    notes, settings, sections = defaultdict(list), defaultdict(list), defaultdict(list)
    for unit in units:
        unit.setdefault("section_key", "")
        section = (unit["manual_id"], unit["section_key"])
        sections[section].append(unit)
        definition = re.match(r"^\s*\(?NOTE\s+(\d+)\)?(?:\s|:)", unit["text"], re.I)
        unit["note_ids"] = [definition[1]] if definition else []
        star = re.match(r'^\s*\*\s*(\d{1,2})\s+[A-Za-z]', unit['text'])
        if star:
            unit['note_ids'] = ['star:' + star[1]]
        unit["note_references"] = [number for number in note_numbers(unit["text"])
                                   if number not in unit["note_ids"]]
        unit["setting_id"] = None
        unit['setting_value'] = None
        if unit["kind"] == "table" and len(unit["table_rows"]) == 1:
            if 'DIP switch' in unit['table_headers'] and 'State' in unit['table_headers']:
                number=unit['table_rows'][0][unit['table_headers'].index('DIP switch')]
                state=unit['table_rows'][0][unit['table_headers'].index('State')]
                if number and re.fullmatch(r'\d+',number.strip()) and state in ('ON','OFF'):
                    unit['setting_id']='dip:'+number.strip()
                    unit['setting_value']=state
            if any(re.search(r"setting\s+(?:range|details)", header, re.I)
                   for header in unit["table_headers"]):
                for header, cell in zip(unit["table_headers"], unit["table_rows"][0]):
                    if re.fullmatch(r"\s*(?:Function\s*)?No\.?\s*", header, re.I) and cell:
                        number = re.match(r"\s*(\d+)\b", cell)
                        if number:
                            unit["setting_id"] = f"function:{number[1]}"
        unit["setting_requirements"] = _requirements(unit["text"])
        unit['setting_exclusions']=[]
        unit["applicable_models"] = model_patterns(unit["title"])
        if unit['kind']=='table':
            # Explicit subclass columns establish scope without an answer key.
            unit['applicable_models']=list(dict.fromkeys(unit['applicable_models']+
                ['-'+m for header in unit['table_headers'] for m in
                 re.findall(r'(?:^|[,/\s])-(\d{2,4}[A-Z])(?=$|[,/\s])',header)]))
        exclusions = []
        for clause in re.split(r"\n\s*[・•]\s*|(?<=[.!?])\s+(?=[A-Z])", unit["text"]):
            if (re.search(r"not\s+(?:necessary|required|applicable|available)|does\s+not\s+apply", clause, re.I)
                    and re.search(r"sub[- ]?class", clause, re.I)):
                exclusions.extend(model_patterns(clause))
        unit["excluded_models"] = list(dict.fromkeys(exclusions))
        unit["is_applicability_rule"] = bool(unit["excluded_models"])
        for number in unit["note_ids"]:
            notes[(section, number)].append(unit)
        if unit["setting_id"]:
            settings[(section, unit["setting_id"])].append(unit)

    def unique(candidates, owner):
        # Prefer native wording to duplicate VLM interpretations on the same page.
        local = [item for item in candidates if item["extraction_method"] != "vlm"]
        selected = list({(item["page"], " ".join(item["text"].split())): item
                         for item in (local or candidates)}.values())
        if len(selected) == 1:
            return selected[0]
        same_page = [item for item in selected if item["page"] == owner["page"]]
        return same_page[0] if len(same_page) == 1 else None

    def add(owner, target, relation, **context):
        if target["id"] != owner["id"]:
            owner["dependencies"].append({"id": target["id"], "relation": relation, **context})

    for unit in units:
        if unit["kind"] == "table" and len(unit["table_rows"]) > 1:
            continue  # Full parent tables must not pull every row's notes.
        section = (unit["manual_id"], unit["section_key"])
        for number in unit["note_references"]:
            candidates = notes[(section, number)]
            if number.startswith('star:'):
                candidates = [item for (scope, label), items in notes.items()
                              if scope[0] == unit['manual_id'] and label == number
                              for item in items if item['page'] == unit['page']]
            target = unique(candidates, unit)
            if target is None:
                issue = f"Required NOTE {number} missing or ambiguous in section {unit['section_key']}"
                unit["blocking_issues"].append(issue)
                continue
            when = None
            for line in unit["text"].splitlines():
                if number in note_numbers(line):
                    value = re.match(r"\s*(\d+)\s*:", line)
                    if value:
                        when = value[1]
            add(unit, target, "note", when_value=when)
            if unit["setting_id"]:
                # Deterministic search enrichment supplies the note's owner;
                # canonical source text and citations stay separate.
                unit["footnotes"].append(target["text"])
                unit["source_refs"].extend(source for source in target["source_refs"]
                                           if source not in unit["source_refs"])
                if when is None:
                    unit["setting_requirements"].extend(target["setting_requirements"])
                unit['setting_exclusions'].extend(_setting_exclusions(unit,target['text']))
        for requirement in unit["setting_requirements"]:
            candidates = settings[(section, requirement["setting_id"])]
            if not candidates:
                candidates = [item for (scope, setting), items in settings.items()
                              if scope[0] == unit["manual_id"] and setting == requirement["setting_id"]
                              for item in items]
            if requirement['setting_id'].startswith('dip:'):
                candidates=[item for item in candidates if item.get('setting_value')==requirement['value']]
            target = unique(candidates, unit)
            if target:
                add(unit, target, "prerequisite", required_value=requirement["value"])
            elif requirement["setting_id"].startswith("function:"):
                unit["blocking_issues"].append(f"Required setting definition unresolved: {requirement['setting_id']}")
        # Applicability exceptions always accompany their section's procedures.
        for rule in sections[section]:
            if rule["is_applicability_rule"] and not unit["is_applicability_rule"]:
                add(unit, rule, "applicability")
        # Plain NOTE paragraphs are section-wide constraints, not ranked extras.
        if unit["kind"] == "procedure":
            for rule in sections[section]:
                if not rule["note_ids"] and re.search(r"\bNOTE\s*:", rule["text"], re.I):
                    add(unit, rule, "prerequisite")
        unit["dependencies"] = list({(item["id"], item["relation"], item.get("required_value"),
                                     item.get("when_value")): item for item in unit["dependencies"]}.values())
    return units
