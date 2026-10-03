"""Resumable local/VLM ingestion with permanent caches and atomic publication."""

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from app.db import utc_now
from app.knowledge import KnowledgeUnit, PageDraft, SCHEMA_VERSION, UnitDraft
from app.semantics import heading_key, merge_text_blocks, link_semantics, note_blocks, note_numbers
from app.tables import AMBIGUOUS_TABLE, native_table


PARSER_VERSION = "layout-v9"
PROMPT_VERSION = "manual-extraction-v1"
MAX_OUTPUT_TOKENS = 4096
INPUT_TOKEN_RESERVATION = 32000


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


@contextmanager
def manual_lock(runtime, manual_id):
    directory = Path(runtime) / "locks"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{manual_id}.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("This manual is being processed. Wait for that job to finish.") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@dataclass(frozen=True)
class BudgetPolicy:
    limit_usd: float
    input_usd_per_million: float
    output_usd_per_million: float
    max_calls: int

    def validate(self):
        amounts = (self.limit_usd, self.input_usd_per_million, self.output_usd_per_million)
        if any(not math.isfinite(value) or value <= 0 for value in amounts) or self.max_calls < 1:
            raise ValueError("Set a positive ingestion budget, verified rates and call allowance")

    @property
    def reservation(self):
        return (INPUT_TOKEN_RESERVATION * self.input_usd_per_million +
                MAX_OUTPUT_TOKENS * self.output_usd_per_million) / 1_000_000


def _reserve(db, manual_id, key, provider, budget):
    budget.validate()
    with db.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        previous = connection.execute("SELECT * FROM extraction_attempts WHERE cache_key=?",
                                      (key,)).fetchone()
        if previous:
            raise ValueError("A previous visual attempt needs review; it will not be resent automatically")
        count, spend = connection.execute(
            "SELECT COUNT(*),COALESCE(SUM(COALESCE(actual_usd,reserved_usd)),0) "
            "FROM extraction_attempts").fetchone()
        if count >= budget.max_calls or spend + budget.reservation > budget.limit_usd:
            raise ValueError("Ingestion budget or call allowance would be exceeded")
        connection.execute(
            "INSERT INTO extraction_attempts "
            "(id,cache_key,manual_id,status,reserved_usd,model,created_at) VALUES (?,?,?,?,?,?,?)",
            (uuid.uuid4().hex, key, manual_id, "pending", budget.reservation,
             provider.model, utc_now()),
        )


def _record_response(db, key, response, budget):
    usage = response.get("usage") or {}
    input_tokens, output_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
    cost = None
    if (type(input_tokens) is int and input_tokens >= 0 and
            type(output_tokens) is int and output_tokens >= 0):
        cost = (input_tokens * budget.input_usd_per_million +
                output_tokens * budget.output_usd_per_million) / 1_000_000
    with db.connect() as connection:
        connection.execute("UPDATE extraction_attempts SET status=?,actual_usd=?,usage_json=? "
                           "WHERE cache_key=?", ("responded", cost, json.dumps({
                               "usage": usage, "reported_model": response.get("model"),
                               "rates": [budget.input_usd_per_million,
                                         budget.output_usd_per_million],
                               "pricing_basis": "user-confirmed gateway rates; not billing receipt",
                           }), key))


def _paragraphs(text):
    """Split only at paragraph/sentence boundaries; retain order without overlap."""
    parts = re.split(r"\n\s*\n|(?=^\s*\(NOTE\s+\d+\))|(?=^\s*\*\s*\d+\s+[A-Za-z])", text.strip(), flags=re.M | re.I)
    output = []
    for part in parts:
        if len(part.split()) <= 250:
            output.append(part)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", part)
        current = []
        for sentence in sentences:
            if current and len(" ".join(current + [sentence]).split()) > 250:
                output.append(" ".join(current))
                current = []
            current.append(sentence)
        output.append(" ".join(current))
    return [part.strip() for part in output if part.strip()]


def _local_page(page):
    units, problems, table_rects = [], [], []
    blocks = page.get_text("blocks", sort=True)
    text = page.get_text("text", sort=True)
    method = "text"
    textpage = None
    if len(text.strip()) < 35:
        try:
            textpage = page.get_textpage_ocr(language="eng", dpi=200)
            ocr = page.get_text("text", textpage=textpage, sort=True)
            if len(ocr.strip()) > len(text.strip()):
                text, blocks, method = ocr, page.get_text("blocks", textpage=textpage), "ocr-eng"
        except Exception:
            problems.append("OCR unavailable or failed; inspect page image")
    headings = [(block[1], heading_key(block[4]), " ".join(block[4].split())) for block in blocks
                if len(block) >= 7 and block[6] == 0 and heading_key(block[4])]
    def context(y):
        before = [entry for entry in headings if entry[0] <= y + 2]
        return (before[-1][1], before[-1][2]) if before else ("", "Manual section")
    try:
        tables = page.find_tables().tables
        for number, table in enumerate(tables):
            grid = table.extract()
            if len(grid) < 2 or table.col_count < 2:
                continue
            section, heading = context(table.bbox[1])
            headers, rows, boxes, uncertainty = native_table(table, heading)
            if len(headers) != table.col_count:
                problems.append("Table headers could not be aligned")
                continue
            if not rows:
                continue
            table_rects.append(pymupdf.Rect(table.bbox))
            body = "\n".join("; ".join(f"{header}: {cell if cell is not None else '[merged/blank]'}"
                                      for header, cell in zip(headers, row)) for row in rows)
            table_text = body if len(body) <= 16000 else (
                f"Table {number + 1} with {len(rows)} rows. Headers: " + "; ".join(headers) +
                ". Full cells are retained in table_rows; search the linked row units.")
            table_unit = UnitDraft(kind="table", title=f"Table {number + 1}",
                                   text=table_text, table_headers=headers, table_rows=rows,
                                   bbox=tuple(table.bbox), cell_boxes=boxes,
                                   table_key=f"local-table:{number}",
                                   uncertainties=uncertainty)
            section, title = context(table.bbox[1])
            table_unit.section_key = section
            table_unit.title = f"{title} / Table {number + 1}"
            units.append(table_unit.model_dump())
            for row_index, row in enumerate(rows):
                # Row records repeat their headers and retain a parent-table link.
                refs = [value for header, value in zip(headers, row)
                        if re.search(r"\bpage\b", header, re.I) and value and value.strip() != "-"]
                references = [label for value in refs for label in re.findall(r"\b\d{1,3}\b", value)]
                external = [value.strip() for value in refs if not re.search(r"\d", value)]
                row_boxes = [pymupdf.Rect(cell) for cell in boxes[row_index] if cell]
                row_rectangle = pymupdf.Rect(table.bbox)
                if row_boxes:
                    row_rectangle = row_boxes[0]
                    for cell in row_boxes[1:]:
                        row_rectangle |= cell
                row_uncertainty=[issue for issue in uncertainty if issue!=AMBIGUOUS_TABLE]
                if any(value is None for value in row) or any('header unresolved' in h for h in headers):
                    row_uncertainty.append(AMBIGUOUS_TABLE)
                units.append(UnitDraft(kind="table", title=f"{title} / Table {number + 1}, row {row_index + 1}",
                                       text="; ".join(f"{header}: {cell if cell is not None else '[merged/blank]'}"
                                                      for header, cell in zip(headers, row)),
                                       bbox=tuple(row_rectangle), parent_key=f"local-table:{number}",
                                       table_headers=headers, table_rows=[row], cell_boxes=[boxes[row_index]],
                                       references=references + external,
                                       section_key=section,
                                       uncertainties=row_uncertainty).model_dump())
    except Exception:
        problems.append("Table parser failed; page needs visual inspection")
    eligible = [block for block in blocks if len(block) >= 7 and block[6] == 0 and not
                any((pymupdf.Rect(block[:4]) & area).get_area() > pymupdf.Rect(block[:4]).get_area() * .5
                    for area in table_rects)]
    for block in merge_text_blocks(eligible):
        if len(block) < 7 or block[6] != 0:
            continue
        rectangle = pymupdf.Rect(block[:4])
        if any((rectangle & area).get_area() > rectangle.get_area() * 0.5
               for area in table_rects):
            continue
        content = block[4].strip()
        if not content or (rectangle.y0 > page.rect.height * 0.94 and len(content) < 80):
            continue
        section, heading = context(rectangle.y0)
        for part in _paragraphs(content):
            if re.match(r"\s*\(NOTE\s+\d+\)", part, re.I):
                continue  # Reconstructed below from line positions, not mixed PDF blocks.
            if len(part) > 16000:
                problems.append("Oversized text block needs structured splitting")
                continue
            warnings = [part] if re.search(r"\b(WARNING|CAUTION|DANGER)\b", part) else []
            refs = re.findall(r"(?:[Pp]age|[Pp]ages|[Pp]\.)\s*(\d{1,3})\b", part)
            kind = "procedure" if re.search(r"(?:^|\n)\s*1[.)]\s", part) else "information"
            units.append(UnitDraft(kind=kind, title=heading, text=part,
                                   bbox=tuple(rectangle), warnings=warnings,
                                   section_key=section,
                                   references=list(dict.fromkeys(refs))).model_dump())
    dictionary = page.get_text("dict", textpage=textpage,
                               flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES)
    lines = [[*line["bbox"], "".join(span["text"] for span in line["spans"])]
             for block in dictionary["blocks"] if block["type"] == 0 for line in block["lines"]]
    for block in note_blocks(lines):
        section, heading = context(block[1])
        units.append(UnitDraft(kind="information", title=heading, text=block[4],
                               bbox=tuple(block[:4]), section_key=section,
                               references=re.findall(r"(?:[Pp]age|[Pp]ages|[Pp]\.)\s*(\d{1,3})\b",
                                                     block[4])).model_dump())
    image_count = len(page.get_image_info())
    drawing_count = len(page.get_drawings())
    if image_count or drawing_count > 5 or method == "ocr-eng" or not text.strip():
        problems.append("Visual content present; automatic figure interpretation not yet verified")
    label = page.get_label()
    if not label:
        candidates = []
        for block in blocks:
            if len(block) < 7 or block[6] != 0 or block[1] < page.rect.height * .94:
                continue
            printed = re.fullmatch(r"\s*[-–]?\s*(\d{1,3}|[ivx]+)\s*[-–]?\s*", block[4], re.I)
            if printed:
                candidates.append(printed[1])
        if len(set(candidates)) == 1:
            label = candidates[0]
    if not label:
        footer = "\n".join(line for line in text.splitlines()[-8:] if line.strip())
        match = re.search(r"(?:^|\n)\s*(\d{1,3}|[ivx]+)\s+[A-Z][A-Z0-9-]+\s*$", footer)
        label = match.group(1) if match else ""
    return {"units": units, "text": text, "printed_label": label, "method": method,
            "width": page.rect.width, "height": page.rect.height,
            "visual_recommended": bool(problems), "issues": problems}


def _validated_visual(response, local):
    if response.get("finish_reason") not in (None, "stop"):
        raise ValueError("Truncated visual extraction")
    content = response["content"].strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    def unique_fields(pairs):
        result={}
        for key,value in pairs:
            if key in result:
                raise ValueError('Duplicate extraction field: '+key)
            result[key]=value
        return result
    data = json.loads(content,object_pairs_hook=unique_fields)
    # Some providers label a fully populated table as information. Its cells
    # still need table validation and atomic row records, without another call.
    if isinstance(data, dict) and isinstance(data.get('units'), list):
        for unit in data['units']:
            if isinstance(unit, dict) and unit.get('table_headers') and unit.get('table_rows'):
                unit['kind'] = 'table'
                if not unit.get('text'):
                    unit['text'] = str(unit.get('title','Table')) + '\nColumn headers: ' + '; '.join(unit['table_headers'])
            if isinstance(unit,dict) and not unit.get('text') and unit.get('source_quote'):
                unit['text']=unit['source_quote']
    parsed = PageDraft.model_validate(data)
    if not parsed.units:
        raise ValueError("Visual extraction returned no units")
    for unit in parsed.units:
        if unit.bbox and (min(unit.bbox) < 0 or unit.bbox[2] > local["width"] or
                          unit.bbox[3] > local["height"]):
            raise ValueError("Visual source rectangle is outside the page")
    return [unit.model_dump() for unit in parsed.units]


def _visual_key(manual, page_index, local, model, base_url):
    return digest({"manual_id": manual["id"], "source": manual["sha256"], "page": page_index,
                  "context": local["text"], "printed_label": local["printed_label"],
                  "title": manual["title"], "dpi": 180,
                  "renderer": pymupdf.VersionBind, "text_context_version": "text-ocr-v1",
                  "schema": SCHEMA_VERSION, "prompt": PROMPT_VERSION,
                  "provider": base_url, "model": model})


def _visual_page(db, folder, manual, page, local, provider, budget, retry_failed=False):
    key = _visual_key(manual, page.number, local, provider.model, provider.base_url)
    path = folder / "visual" / f"{key}.json"
    if path.exists():
        response = json.loads(path.read_text())
        try:
            validated=_validated_visual(response, local)
            with db.connect()as connection:
                connection.execute("UPDATE extraction_attempts SET status='validated',error=NULL WHERE cache_key=? AND status='invalid'",
                                   (response.get('attempt_key',key),))
            return validated, key
        except (ValueError, KeyError, TypeError):
            if not retry_failed:
                raise ValueError("Invalid visual extraction saved for review; no automatic rebilling") from None
            history = folder / "visual" / "history"
            history.mkdir(parents=True, exist_ok=True)
            os.replace(path, history / f"{key}-{uuid.uuid4().hex}.json")
    if budget is None:
        raise ValueError("A verified ingestion budget is required before visual requests")
    with db.connect() as connection:
        previous = connection.execute("SELECT 1 FROM extraction_attempts WHERE cache_key=?", (key,)).fetchone()
    attempt_key = f"{key}:retry:{uuid.uuid4().hex}" if retry_failed and previous else key
    image = page.get_pixmap(dpi=180, alpha=False).tobytes("png")
    _reserve(db, manual["id"], attempt_key, provider, budget)
    try:
        response = provider.extract_page(image, local["text"], {
            "manual": manual["title"], "pdf_page": page.number + 1,
            "printed_label": local["printed_label"], "schema_version": SCHEMA_VERSION,
        })
        response["attempt_key"] = attempt_key
        response["extraction_config"] = {"model": provider.model, "base_url": provider.base_url,
                                          "prompt": PROMPT_VERSION, "schema": SCHEMA_VERSION,
                                          "parser": PARSER_VERSION, "renderer": pymupdf.VersionBind}
        response["source_context"] = {"manual_id": manual["id"], "pdf_page": page.number + 1,
                                      "manual_sha256": manual["sha256"],
                                      "image_sha256": hashlib.sha256(image).hexdigest(),
                                      "renderer": pymupdf.VersionBind, "dpi": 180}
        image_path = folder / "images" / f"{page.number + 1}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image)
        atomic_json(path, response)
        _record_response(db, attempt_key, response, budget)
    except Exception:
        with db.connect() as connection:
            connection.execute("UPDATE extraction_attempts SET status='uncertain',error=? "
                               "WHERE cache_key=?", ("Request interrupted or failed; inspect before retry", attempt_key))
        raise
    try:
        return _validated_visual(response, local), key
    except (ValueError, KeyError, TypeError):
        with db.connect() as connection:
            connection.execute("UPDATE extraction_attempts SET status='invalid',error=? "
                               "WHERE cache_key=?", ("Invalid extraction; saved response needs review", attempt_key))
        raise ValueError("Invalid visual extraction saved for review; no automatic rebilling") from None


def process_manual(db, runtime, manual_id, provider=None, budget=None, visual_pages=None,
                   progress=None, retry_failed=False, retriever=None):
    """Stage all units, then publish together. Existing knowledge survives failure."""
    runtime = Path(runtime)
    with manual_lock(runtime, manual_id):
        with db.connect() as connection:
            row = connection.execute("SELECT * FROM manuals WHERE id=?", (manual_id,)).fetchone()
        if not row:
            raise ValueError("Manual does not exist or was removed")
        manual = dict(row)
        requested = set(visual_pages or [])
        if any(type(number) is not int or number < 1 or number > manual["pages"] for number in requested):
            raise ValueError("Visual page is outside the manual")
        if requested and provider is None:
            raise ValueError("Select a visual extraction provider")
        folder = runtime / "knowledge" / manual_id
        folder.mkdir(parents=True, exist_ok=True)
        with db.connect() as connection:
            connection.execute("INSERT INTO ingestion_jobs (manual_id,status,updated_at) VALUES (?,?,?) "
                               "ON CONFLICT(manual_id) DO UPDATE SET status='extracting',error=NULL,"
                               "pages_done=0,updated_at=excluded.updated_at",
                               (manual_id, "extracting", utc_now()))
        pages = []
        try:
            with pymupdf.open(runtime / "manuals" / f"{manual_id}.pdf") as document:
                for index, page in enumerate(document):
                    local_key = digest([manual["sha256"], index, PARSER_VERSION, pymupdf.VersionBind])
                    local_path = folder / "local" / f"{local_key}.json"
                    if local_path.exists():
                        local = json.loads(local_path.read_text())
                    else:
                        local = _local_page(page)
                        atomic_json(local_path, local)
                    visual = folder / "pages" / f"{index + 1}.json"
                    corrected = visual.exists() and json.loads(visual.read_text()).get("review_status") == "user_checked"
                    if index + 1 in requested and not corrected:
                        visual_units, key = _visual_page(db, folder, manual, page, local, provider, budget, retry_failed)
                        atomic_json(visual, {"key": key, "units": visual_units})
                    # A later local-only rebuild retains completed visual enrichment.
                    overlay = json.loads(visual.read_text()) if visual.exists() else {}
                    extra = overlay.get("units", [])
                    pages.append({**local, "page": index + 1, "visual_units": extra,
                                  "visual_review": overlay.get("review_status", "unreviewed"),
                                  "visual_reused": overlay.get("reused_from")})
                    with db.connect() as connection:
                        connection.execute("UPDATE ingestion_jobs SET pages_done=?,updated_at=? "
                                           "WHERE manual_id=?", (index + 1, utc_now(), manual_id))
                    if progress:
                        progress(index + 1, manual["pages"])
            units = _assemble(manual, pages)
            release_id = digest([PARSER_VERSION, SCHEMA_VERSION, units])[:24]
            release_folder = folder / "releases" / release_id
            release_folder.mkdir(parents=True, exist_ok=True)
            canonical = release_folder / "units.jsonl"
            temporary = canonical.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as output:
                for unit in units:
                    output.write(json.dumps(unit, ensure_ascii=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, canonical)
            manifest = {"manual_id": manual_id, "manual_sha256": manual["sha256"],
                        "release_id": release_id, "parser": PARSER_VERSION,
                        "schema": SCHEMA_VERSION, "units": len(units), "pages": len(pages),
                        "blocked_units": sum(bool(unit["blocking_issues"]) for unit in units),
                        "numbered_notes": sum(bool(unit["note_ids"]) for unit in units),
                        "dependency_counts": {relation: sum(edge["relation"] == relation for unit in units
                                                            for edge in unit["dependencies"])
                                              for relation in ("note", "prerequisite", "applicability", "warning", "table", "procedure")},
                        "visual_pages": [p["page"] for p in pages if p["visual_units"]],
                        "reused_visual_pages": [p["page"] for p in pages if p["visual_reused"]],
                        "pages_needing_visual_review": [p["page"] for p in pages
                                                        if p["visual_recommended"] and not p["visual_units"]],
                        "review": "unreviewed", "issues": {str(p["page"]): p["issues"] for p in pages}}
            atomic_json(release_folder / "manifest.json", manifest)
            with db.connect() as connection:
                connection.execute("UPDATE ingestion_jobs SET status='indexing',updated_at=? WHERE manual_id=?",
                                   (utc_now(), manual_id))
            prepared = retriever.prepare(manual_id, units, release_id) if retriever else None
            with db.connect() as connection:
                if not connection.execute("SELECT 1 FROM manuals WHERE id=?", (manual_id,)).fetchone():
                    raise ValueError("Manual was removed before publication")
                connection.execute("DELETE FROM knowledge_units WHERE manual_id=?", (manual_id,))
                connection.execute("DELETE FROM chunks WHERE manual_id=?", (manual_id,))
                for ordinal, unit in enumerate(units):
                    connection.execute("INSERT INTO knowledge_units VALUES (?,?,?,?,?,?,?)", (
                        unit["id"], manual_id, unit["page"], unit["kind"], unit["search_text"],
                        json.dumps(unit, ensure_ascii=False), release_id))
                    if _searchable(unit):
                        connection.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?)", (
                            unit["id"], manual_id, unit["page"], ordinal,
                            unit["search_text"], unit["extraction_method"]))
                status = "visual-enriched" if manifest["visual_pages"] else "local-structured"
                connection.execute("UPDATE manuals SET status='ready',knowledge_release=?,knowledge_status=? WHERE id=?",
                                   (release_id, status, manual_id))
                connection.execute("UPDATE ingestion_jobs SET status='published',release_id=?,error=NULL,"
                                   "updated_at=? WHERE manual_id=?", (release_id, utc_now(), manual_id))
            if prepared:
                retriever.publish_prepared(prepared)
            return {**manifest, "pages_done": len(pages)}
        except Exception as exc:
            with db.connect() as connection:
                connection.execute("UPDATE ingestion_jobs SET status='failed',error=?,updated_at=? "
                                   "WHERE manual_id=?", (str(exc)[:300], utc_now(), manual_id))
            raise


def enrich_recommended(db, runtime, manual_id, provider, budget, retriever=None, progress=None):
    """Budget-bounded upload enrichment; local knowledge remains usable on failure."""
    budget.validate()
    runtime = Path(runtime)
    manual = next(dict(m) for m in db.manuals() if m['id']==manual_id)
    folder = runtime/'knowledge'/manual_id
    candidates=[]
    with db.connect() as connection:
        attempted={row[0] for row in connection.execute('SELECT cache_key FROM extraction_attempts')}
    for index in range(manual['pages']):
        if (folder/'pages'/f'{index+1}.json').exists():
            continue
        local_path=folder/'local'/(digest([manual['sha256'],index,PARSER_VERSION,pymupdf.VersionBind])+'.json')
        if not local_path.exists():
            continue
        local=json.loads(local_path.read_text())
        if not local['visual_recommended']:
            continue
        key=_visual_key(manual,index,local,provider.model,provider.base_url)
        if key in attempted:
            continue  # Timeout or invalid draft may already have been billed.
        table_ambiguity=any(u.get('table_rows') and u.get('uncertainties') for u in local['units'])
        priority=100 if not local['text'].strip() else 80 if local['method']=='ocr-eng' else 60 if table_ambiguity else 20
        # Covers and contents are low-value image calls, but remain inspectable.
        if (index==0 and priority==20 and not any(u.get('table_rows') for u in local['units'])) or re.search(r'\b(?:CONTENTS|TABLE OF CONTENTS)\b',local['text'][:800]):
            continue
        candidates.append((-priority,index+1))
    spend=db.ingestion_spend()
    allowance=min(budget.max_calls-spend['attempts'],
                  int(max(0,budget.limit_usd-spend['accounted_usd'])/budget.reservation))
    selected=[page for _,page in sorted(candidates)[:max(0,allowance)]]
    result={'selected_pages':selected,'deferred_pages':len(candidates)-len(selected),'error':None}
    if selected:
        try:
            process_manual(db,runtime,manual_id,provider,budget,selected,progress=progress,retriever=retriever)
        except Exception as exc:
            result['error']=str(exc)[:300]
            # Publish any completed cached pages, without sending another image.
            process_manual(db,runtime,manual_id,retriever=retriever)
    atomic_json(folder/'automatic-enrichment.json',result)
    return result


def _native_row_support(draft, local_units):
    """Check visual cell wording under matching native headers, preserving decimals."""
    if draft.kind!='table' or len(draft.table_rows)!=1:
        return False
    header_key=lambda text: ''.join(c.lower()for c in text if c.isalnum())
    cell_key=lambda text: re.sub(r'\s+','',text).lower()
    for native in local_units:
        if native['kind']!='table' or native.get('parent_key'):
            continue
        headers={header_key(h):i for i,h in enumerate(native['table_headers']) if 'unresolved' not in h}
        matching=[(i,headers[header_key(h)])for i,h in enumerate(draft.table_headers)if header_key(h)in headers]
        if len(matching)<2:
            continue
        aligned=all(value is not None and any(row[n] is not None and cell_key(value) in cell_key(row[n])
                        for row in native['table_rows'])
                    for i,n in matching for value in [draft.table_rows[0][i]])
        all_cells=[cell_key(c)for row in native['table_rows']for c in row if c]
        other=all(c is not None and any(cell_key(c)in text for text in all_cells)
                  for i,c in enumerate(draft.table_rows[0]) if i not in {i for i,_ in matching})
        if aligned and other:
            return True
    return False


def _assemble(manual, pages):
    units, labels = [], {}
    previous_section = ('', 'Manual section')
    for page in pages:
        local_units = []
        for raw in page['units']:
            item = dict(raw)
            if not item.get('section_key') and previous_section[0]:
                item['section_key'] = previous_section[0]
                if item['title'] == 'Manual section':
                    item['title'] = previous_section[1]
            local_units.append(item)
        scoped = [item for item in local_units if item.get('section_key')]
        if scoped:
            last = max(scoped, key=lambda item: item.get('bbox', None)[1] if item.get('bbox') else 0)
            previous_section = (last['section_key'], last['title'].split(' / Table')[0])
        if page["printed_label"]:
            labels.setdefault(page["printed_label"], []).append(page["page"])
        table_parents = {}
        # Keep local source wording alongside visual interpretations for cross-checking.
        visual = []
        footnote_texts = {' '.join(raw['text'].split()) for raw in local_units}
        for index, raw in enumerate(page["visual_units"]):
            for note in raw.get('footnotes', []):
                normalized = ' '.join(note.split())
                if re.match(r'^\s*(?:\*\s*\d{1,2}|\(?NOTE\s+\d+\)?)\s', note, re.I) and normalized not in footnote_texts:
                    visual.append(UnitDraft(kind='information', title=raw['title'], text=note,
                                            source_quote=note, section_key=raw.get('section_key','')).model_dump())
                    footnote_texts.add(normalized)
            parent = {**raw}
            if raw["kind"] == "table" and len(raw["table_rows"]) > 1:
                parent["table_key"] = f"visual-table:{index}"
                visual.append(parent)
                for row_index, row in enumerate(raw["table_rows"]):
                    child = {**raw, "title": f"{raw['title']}, row {row_index + 1}",
                             "text": "; ".join(f"{header}: {value if value is not None else '[merged/blank]'}"
                                              for header, value in zip(raw["table_headers"], row)),
                             "table_rows": [row], "cell_boxes": [],
                             "parent_key": parent["table_key"], "table_key": None}
                    refs=set(note_numbers(child['text']))
                    child['footnotes']=[note for note in raw['footnotes']
                        if not note_numbers(note) or refs.intersection(note_numbers(note))]
                    visual.append(child)
            else:
                visual.append(parent)
        for ordinal, raw in enumerate(local_units + visual):
            if ordinal >= len(local_units) and not raw.get("section_key"):
                scopes = {item.get("section_key") for item in local_units if item.get("section_key")}
                if len(scopes) == 1:
                    raw = {**raw, "section_key": next(iter(scopes))}
            draft = UnitDraft.model_validate(raw)
            method = page["method"] if ordinal < len(page["units"]) else "vlm"
            unit_id = f"{manual['id']}:{page['page']}:u{ordinal}-{digest(raw)[:8]}"
            blocking, flags = [], list(draft.uncertainties)
            if method == "vlm" and draft.uncertainties:
                blocking.extend(draft.uncertainties)
            if method == "vlm" and draft.source_quote and page["text"].strip():
                normalized = lambda value: " ".join(value.lower().split())
                if normalized(draft.source_quote) not in normalized(page["text"]) and not _native_row_support(draft,local_units):
                    flags.append("Visual quotation does not match the text layer; source review needed")
            if method == "vlm" and not draft.source_quote:
                flags.append("Visual extraction supplied no exact source quotation")
            if method!='vlm' and draft.kind=='table' and draft.uncertainties:
                if (draft.parent_key or len(draft.table_rows)==1) and any(issue in draft.uncertainties for issue in
                    (AMBIGUOUS_TABLE,'Headerless table relationship unresolved')):
                    blocking.append('Native table cell relationship unresolved; visual source review required')
                normal=lambda s: ''.join(c.lower() for c in s if c.isalnum())
                headers={normal(h) for h in draft.table_headers if 'unresolved' not in h}
                if any(v['kind']=='table' and not v.get('uncertainties') and
                       len(headers.intersection(normal(h)for h in v['table_headers']))>=2 for v in visual):
                    blocking.append('Ambiguous native table superseded by cached visual table with matching headers')
            if draft.kind == "table" and not draft.parent_key and draft.table_key:
                table_parents[draft.table_key] = unit_id
            parent = table_parents.get(draft.parent_key)
            unit = {**draft.model_dump(), "id": unit_id, "manual_id": manual["id"],
                    "manual_sha256": manual["sha256"], "page": page["page"],
                    "printed_page_label": page["printed_label"], "parent_id": parent,
                    "source_refs": [{"manual_id": manual["id"], "pdf_page_index": page["page"] - 1,
                                     "printed_page_label": page["printed_label"], "bbox": draft.bbox}],
                    "search_text": draft.build_search_text(), "extraction_method": method,
                    "review_status": page["visual_review"] if method == "vlm" else "unreviewed", "flags": flags,
                    "blocking_issues": blocking, "dependencies": []}
            if parent:
                unit["dependencies"].append({"id": parent, "relation": "table"})
            units.append(unit)
    by_page = {}
    for unit in units:
        by_page.setdefault(unit["page"], []).append(unit)
    global_warnings = [unit for unit in units if unit["page"] <= 4 and unit["warnings"]]
    for unit in units:
        if (unit["kind"] in ("procedure", "troubleshooting", "table") or unit["parent_id"] or
                re.search(r"\b(adjust|remove|install|disconnect|tighten|loosen|lubricate)\b", unit["text"], re.I)):
            warnings = global_warnings + [other for other in by_page[unit["page"]] if other["warnings"]]
            unit["dependencies"].extend({"id": other["id"], "relation": "warning"}
                                        for other in warnings if other["id"] != unit["id"])
        for label in unit["references"]:
            matches = labels.get(label, [])
            if len(matches) == 1:
                unit["dependencies"].extend({"id": other["id"], "relation": "procedure"}
                                            for other in by_page[matches[0]] if other["id"] != unit["id"])
            else:
                unit["flags"].append(f"Unresolved/ambiguous printed-page reference: {label}")
                if unit["kind"] in ("procedure", "troubleshooting"):
                    unit["blocking_issues"].append(f"Required source reference unresolved: {label}")
    units = link_semantics(units)
    from app.rules import setting_rules
    for unit in units:
        unit['setting_rules']=setting_rules(unit)
    by_id = {unit["id"]: unit for unit in units}
    if len(by_id) != len(units):
        raise ValueError("Duplicate canonical knowledge IDs")
    for unit in units:
        if any(edge["id"] not in by_id or by_id[edge["id"]]["manual_id"] != unit["manual_id"]
               for edge in unit["dependencies"]):
            raise ValueError("Canonical dependency is missing or belongs to another manual")
        unit["search_text"] = UnitDraft.model_validate({key: unit[key] for key in UnitDraft.model_fields}).build_search_text()
        for rule in unit['setting_rules']:
            if rule['effect']:
                conditions=', '.join(c['setting_id']+' = '+c['value']for c in rule['conditions'])
                exclusions=', '.join(c['setting_id']+' = '+c['value']for c in rule['exclusions'])
                qualifiers=('; enabled with '+conditions if conditions else'')+('; except '+exclusions if exclusions else'')
                unit['search_text']+='\nResolved setting branch '+rule['value']+qualifiers+': '+rule['description']
    return [KnowledgeUnit.model_validate(unit).model_dump() for unit in units]


def _searchable(unit):
    # Search atomic rows; retain complete parent tables in canonical knowledge.
    ambiguous = unit['kind']=='table' and any(flag in unit.get('flags',[]) for flag in
                (AMBIGUOUS_TABLE,'Headerless table relationship unresolved'))
    return not unit["blocking_issues"] and not ambiguous and not (unit["kind"] == "table" and
                                                not unit["parent_id"] and len(unit["table_rows"]) > 1)


def save_visual_correction(db, runtime, manual_id, page_number, content, retriever=None):
    """An explicit user correction is durable, but never becomes domain approval."""
    parsed = PageDraft.model_validate_json(content)
    if not parsed.units:
        raise ValueError("Correction must retain at least one unit")
    with manual_lock(runtime, manual_id):
        with db.connect() as connection:
            manual = connection.execute("SELECT * FROM manuals WHERE id=?", (manual_id,)).fetchone()
        if not manual or not 1 <= page_number <= manual["pages"]:
            raise ValueError("Correction source page does not exist")
        with pymupdf.open(Path(runtime) / "manuals" / f"{manual_id}.pdf") as document:
            rectangle = document[page_number - 1].rect
            for unit in parsed.units:
                if unit.bbox and (min(unit.bbox) < 0 or unit.bbox[2] > rectangle.width or
                                  unit.bbox[3] > rectangle.height):
                    raise ValueError("Correction rectangle is outside its source page")
        folder = Path(runtime) / "knowledge" / manual_id
        corrected = {"key": "user:" + digest(content), "units": [unit.model_dump() for unit in parsed.units],
                     "review_status": "user_checked", "corrected_at": utc_now(),
                     "manual_sha256": manual["sha256"], "pdf_page": page_number,
                     "review_basis": "user correction; not independent engineering approval"}
        atomic_json(folder / "corrections" / f"{page_number}-{digest(content)[:16]}.json", corrected)
        atomic_json(folder / "pages" / f"{page_number}.json", corrected)
    return process_manual(db, runtime, manual_id, retriever=retriever)


def _reuse_replacement_cache(runtime, old, new):
    """Reuse matching draft pages, never transfer engineering approval."""
    if old["title"] != new["title"]:
        return
    runtime = Path(runtime)
    previous = runtime / "knowledge" / old["id"]
    destination = runtime / "knowledge" / new["id"]
    overlays = list((previous / "pages").glob("*.json"))
    if overlays:
        with pymupdf.open(runtime / "manuals" / f"{old['id']}.pdf") as before:
            candidates = {}
            for path in overlays:
                index = int(path.stem) - 1
                page = before[index]
                image_hash = hashlib.sha256(page.get_pixmap(dpi=180, alpha=False).tobytes("png")).hexdigest()
                fingerprint = digest([image_hash, page.get_text("text", sort=True), page.get_label()])
                local_key = digest([old["sha256"], index, PARSER_VERSION, pymupdf.VersionBind])
                local_path = previous / "local" / f"{local_key}.json"
                if local_path.exists():
                    candidates[fingerprint] = (index, json.loads(path.read_text()), json.loads(local_path.read_text()))
        with pymupdf.open(runtime / "manuals" / f"{new['id']}.pdf") as after:
            for index, page in enumerate(after):
                image = page.get_pixmap(dpi=180, alpha=False).tobytes("png")
                fingerprint = digest([hashlib.sha256(image).hexdigest(), page.get_text("text", sort=True), page.get_label()])
                if fingerprint not in candidates:
                    continue
                previous_index, overlay, local = candidates[fingerprint]
                old_key = overlay["key"]
                local_key = digest([new["sha256"], index, PARSER_VERSION, pymupdf.VersionBind])
                atomic_json(destination / "local" / f"{local_key}.json", local)
                overlay = {**overlay, "review_status": "unreviewed",
                           "reused_from": {"manual_id": old["id"], "pdf_page": previous_index + 1}}
                raw_path = previous / "visual" / f"{old_key}.json"
                if raw_path.exists():
                    raw = json.loads(raw_path.read_text())
                    config = raw.get("extraction_config", {})
                    if (config.get("prompt") == PROMPT_VERSION and config.get("schema") == SCHEMA_VERSION and
                            config.get("renderer") == pymupdf.VersionBind):
                        _validated_visual(raw, local)
                        key = _visual_key(new, index, local, config["model"], config["base_url"])
                        raw["source_context"] = {**raw["source_context"], "manual_id": new["id"],
                                                 "manual_sha256": new["sha256"], "pdf_page": index + 1}
                        raw["reused_from"] = overlay["reused_from"]
                        atomic_json(destination / "visual" / f"{key}.json", raw)
                        overlay["key"] = key
                atomic_json(destination / "pages" / f"{index + 1}.json", overlay)
    vectors = runtime / "embeddings" / old["id"]
    if vectors.exists():
        shutil.copytree(vectors, runtime / "embeddings" / new["id"], dirs_exist_ok=True)


def replace_manual(db, runtime, old_id, filename, pdf, retriever):
    from app.ingest import add_pdf

    with db.connect() as connection:
        old = connection.execute("SELECT * FROM manuals WHERE id=?", (old_id,)).fetchone()
    if not old:
        raise ValueError("Manual to replace does not exist")
    new, _ = add_pdf(db, runtime, filename, pdf, publish_baseline=False)
    if new["id"] == old_id:
        return new
    with db.connect() as connection:
        new_record = connection.execute("SELECT * FROM manuals WHERE id=?", (new["id"],)).fetchone()
    _reuse_replacement_cache(runtime, dict(old), dict(new_record))
    process_manual(db, runtime, new["id"], retriever=retriever)
    retriever.sync()  # New vectors must exist before retiring the old revision.
    with manual_lock(runtime, old_id):
        with db.connect() as connection:
            connection.execute("UPDATE manuals SET status='retired' WHERE id=?", (old_id,))
    retriever.sync()
    return new
