"""PDF text extraction with page provenance and optional OCR fallback."""

import hashlib
import json
import re
import uuid
import shutil
from pathlib import Path

import pymupdf

from app.db import Database, utc_now


MAX_PDF_BYTES = 100 * 1024 * 1024


def _clean(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\x00", "")).strip()


def _split(text: str, limit: int = 1100, overlap: int = 180) -> list[str]:
    """Keep source wording intact; overlap helps when a procedure crosses windows."""
    text = _clean(text)
    if not text:
        return []
    parts = []
    start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            for separator in ("\n\n", "\n", ". ", "। "):
                cut = text.rfind(separator, start + limit // 2, end)
                if cut > start:
                    end = cut + len(separator)
                    break
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return [part for part in parts if part]


def extract(pdf: bytes, ocr: bool = True):
    if len(pdf) > MAX_PDF_BYTES:
        raise ValueError("PDF exceeds the 100 MB upload limit")
    if not pdf.startswith(b"%PDF-"):
        raise ValueError("The file is not a PDF")
    try:
        document = pymupdf.open(stream=pdf, filetype="pdf")
    except Exception as exc:
        raise ValueError("Cannot read this PDF") from exc
    if document.is_encrypted:
        raise ValueError("Password-protected PDFs are not supported")
    if document.page_count == 0:
        raise ValueError("The PDF has no pages")
    rows = []
    try:
        title = " ".join((document.metadata.get("title") or "").split())[:120]
        for page_number, page in enumerate(document, 1):
            text = page.get_text("text", sort=True)
            extraction = "text"
            if len(text.strip()) < 35 and ocr:
                try:
                    ocr_text = page.get_text("text", textpage=page.get_textpage_ocr(
                        language="eng", dpi=200))
                    if len(ocr_text.strip()) > len(text.strip()):
                        text = ocr_text
                        extraction = "ocr-eng"
                except Exception as exc:
                    if not text.strip():
                        raise ValueError(
                            f"Page {page_number} needs OCR, but Tesseract English OCR is unavailable"
                        ) from exc
            for ordinal, piece in enumerate(_split(text)):
                rows.append((page_number, ordinal, piece, extraction))
        return document.page_count, rows, title
    finally:
        document.close()


def add_pdf(db: Database, runtime: Path, filename: str, pdf: bytes, ocr: bool = True,
            publish_baseline: bool = True):
    digest = hashlib.sha256(pdf).hexdigest()
    with db.connect() as connection:
        existing = connection.execute("SELECT * FROM manuals WHERE sha256=?", (digest,)).fetchone()
    if existing:
        return dict(existing), False
    pages, pieces, embedded_title = extract(pdf, ocr=ocr)
    # A diagram-only manual can still be used through explicit page-image questions.
    manual_id = uuid.uuid4().hex
    directory = runtime / "manuals"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{manual_id}.pdf"
    target.write_bytes(pdf)
    title = embedded_title or Path(filename).stem[:120]
    try:
        with db.connect() as connection:
            connection.execute(
                "INSERT INTO manuals (id,title,filename,sha256,pages,status,created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (manual_id, title, Path(filename).name[:200], digest,
                 pages, "ready" if publish_baseline else "processing", utc_now()),
            )
            connection.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                [(f"{manual_id}:{page}:{ordinal}", manual_id, page, ordinal, text, method)
                 for page, ordinal, text, method in pieces],
            )
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return {"id": manual_id, "title": title, "pages": pages}, True


def delete_pdf(db: Database, runtime: Path, manual_id: str):
    from app.structured import manual_lock

    with manual_lock(runtime, manual_id):
        _delete_pdf(db, runtime, manual_id)


def _delete_pdf(db: Database, runtime: Path, manual_id: str):
    with db.connect() as connection:
        for query in connection.execute("SELECT id, sources_json,evidence_json FROM queries").fetchall():
            sources = json.loads(query["sources_json"]) + json.loads(query["evidence_json"])
            if any(source.get("manual_id") == manual_id or
                   source.get("id", "").startswith(f"{manual_id}:") for source in sources):
                connection.execute("DELETE FROM queries WHERE id=?", (query["id"],))
        connection.execute("DELETE FROM manuals WHERE id=?", (manual_id,))
    (runtime / "manuals" / f"{manual_id}.pdf").unlink(missing_ok=True)
    for directory in (runtime / "knowledge" / manual_id, runtime / "embeddings" / manual_id):
        if directory.exists():
            shutil.rmtree(directory)
