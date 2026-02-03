import json
from typing import Any, Dict, List, Tuple

import fitz

from core.contracts import CanonicalPage, CanonicalSpan, PageRecord


def canonicalize_document(
    doc_id: str, pdf_path: str, pages: List[PageRecord]
) -> List[CanonicalPage]:
    pdf = fitz.open(pdf_path)
    try:
        canonical_pages: List[CanonicalPage] = []
        for page_record in pages:
            page_index = page_record.page_number - 1
            if page_record.di_json_path:
                canonical_pages.append(
                    _canonicalize_from_di(
                        doc_id=doc_id,
                        page_number=page_record.page_number,
                        di_json_path=page_record.di_json_path,
                    )
                )
            else:
                page = pdf.load_page(page_index)
                canonical_pages.append(
                    _canonicalize_from_native(
                        doc_id=doc_id,
                        page_number=page_record.page_number,
                        page=page,
                    )
                )
        return canonical_pages
    finally:
        pdf.close()


def _canonicalize_from_di(
    doc_id: str, page_number: int, di_json_path: str
) -> CanonicalPage:
    with open(di_json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    pages = payload.get("pages", [])
    page = next((p for p in pages if p.get("pageNumber") == page_number), None)
    if not page:
        return CanonicalPage(doc_id=doc_id, page_number=page_number, text="", spans=[])
    lines = page.get("lines", [])
    return _build_canonical_page(
        doc_id=doc_id,
        page_number=page_number,
        line_entries=[
            (
                line.get("content", ""),
                _polygon_from_di(line.get("polygon", [])),
            )
            for line in lines
        ],
        source_type="di",
    )


def _canonicalize_from_native(
    doc_id: str, page_number: int, page: fitz.Page
) -> CanonicalPage:
    words = page.get_text("words")
    if not words:
        return CanonicalPage(doc_id=doc_id, page_number=page_number, text="", spans=[])

    lines: Dict[Tuple[int, int], List[Tuple[float, float, float, float, str]]] = {}
    for x0, y0, x1, y1, word, block_no, line_no, _ in words:
        lines.setdefault((block_no, line_no), []).append((x0, y0, x1, y1, word))

    line_entries: List[Tuple[str, List[Dict[str, Any]]]] = []
    for (_, _), line_words in sorted(lines.items(), key=lambda item: item[0]):
        line_words.sort(key=lambda entry: entry[0])
        text = " ".join(word for _, _, _, _, word in line_words)
        polygon = _polygon_from_bbox(
            min(w[0] for w in line_words),
            min(w[1] for w in line_words),
            max(w[2] for w in line_words),
            max(w[3] for w in line_words),
        )
        line_entries.append((text, polygon))

    return _build_canonical_page(
        doc_id=doc_id,
        page_number=page_number,
        line_entries=line_entries,
        source_type="native",
    )


def _build_canonical_page(
    doc_id: str,
    page_number: int,
    line_entries: List[Tuple[str, List[Dict[str, Any]]]],
    source_type: str,
) -> CanonicalPage:
    spans: List[CanonicalSpan] = []
    text_parts: List[str] = []
    cursor = 0

    for line_text, polygon in line_entries:
        if not line_text:
            continue
        start = cursor
        end = start + len(line_text)
        spans.append(
            CanonicalSpan(
                text=line_text,
                char_start=start,
                char_end=end,
                polygons=[{"page_number": page_number, "polygon": polygon}],
                source_type=source_type,
                page_number=page_number,
            )
        )
        text_parts.append(line_text)
        cursor = end + 1

    page_text = "\n".join(text_parts)
    return CanonicalPage(
        doc_id=doc_id, page_number=page_number, text=page_text, spans=spans
    )


def _polygon_from_di(points: List[float]) -> List[Dict[str, float]]:
    polygon = []
    for i in range(0, len(points), 2):
        try:
            polygon.append({"x": float(points[i]), "y": float(points[i + 1])})
        except (IndexError, ValueError):
            break
    return polygon


def _polygon_from_bbox(
    x0: float, y0: float, x1: float, y1: float
) -> List[Dict[str, float]]:
    return [
        {"x": x0, "y": y0},
        {"x": x1, "y": y0},
        {"x": x1, "y": y1},
        {"x": x0, "y": y1},
    ]
