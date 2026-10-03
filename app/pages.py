"""Render a selected source page for local inspection or a targeted vision question."""

import pymupdf


def render_page(runtime, manual_id: str, page_number: int, dpi: int = 135) -> bytes:
    path = runtime / "manuals" / f"{manual_id}.pdf"
    with pymupdf.open(path) as document:
        if page_number < 1 or page_number > document.page_count:
            raise ValueError("Page number is outside this manual")
        return document[page_number - 1].get_pixmap(dpi=dpi, alpha=False).tobytes("png")
