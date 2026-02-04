import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import fitz

from core.contracts import CanonicalPage, CanonicalSpan, PageRecord


@dataclass(frozen=True)
class TableBlock:
    markdown: str
    polygon: List[Dict[str, float]]
    bbox: Tuple[float, float, float, float]


def canonicalize_document(
    doc_id: str,
    pdf_path: str,
    pages: List[PageRecord],
    progress_cb=None,
) -> List[CanonicalPage]:
    pdf = fitz.open(pdf_path)
    try:
        canonical_pages: List[CanonicalPage] = []
        heading_stack: List[str] = []
        root = _heading_root(pdf_path, doc_id)
        total_pages = len(pages)
        for index, page_record in enumerate(pages, start=1):
            if progress_cb:
                progress_cb("canonicalize", index, total_pages)
            page_index = page_record.page_number - 1
            if page_record.di_json_path:
                canonical_pages.append(
                    _canonicalize_from_di(
                        doc_id=doc_id,
                        page_number=page_record.page_number,
                        di_json_path=page_record.di_json_path,
                        heading_stack=heading_stack,
                        heading_root=root,
                    )
                )
            else:
                page = pdf.load_page(page_index)
                canonical_pages.append(
                    _canonicalize_from_native(
                        doc_id=doc_id,
                        page_number=page_record.page_number,
                        page=page,
                        heading_stack=heading_stack,
                        heading_root=root,
                    )
                )
        return canonical_pages
    finally:
        pdf.close()


def _canonicalize_from_di(
    doc_id: str,
    page_number: int,
    di_json_path: str,
    heading_stack: List[str],
    heading_root: str,
) -> CanonicalPage:
    with open(di_json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    pages = payload.get("pages", [])
    page = next((p for p in pages if p.get("pageNumber") == page_number), None)
    if not page:
        return CanonicalPage(doc_id=doc_id, page_number=page_number, text="", spans=[])
    lines = page.get("lines", [])
    tables = _extract_tables_from_di(payload, page_number)
    table_bboxes = [t.bbox for t in tables]
    return _build_canonical_page(
        doc_id=doc_id,
        page_number=page_number,
        line_entries=[
            (
                line.get("content", ""),
                _polygon_from_di(line.get("polygon", [])),
            )
            for line in lines
            if not _polygon_overlaps_any(_polygon_from_di(line.get("polygon", [])), table_bboxes)
        ],
        source_type="di",
        heading_stack=heading_stack,
        heading_root=heading_root,
        table_blocks=tables,
    )


def _canonicalize_from_native(
    doc_id: str,
    page_number: int,
    page: fitz.Page,
    heading_stack: List[str],
    heading_root: str,
) -> CanonicalPage:
    words = page.get_text("words")
    if not words:
        return CanonicalPage(doc_id=doc_id, page_number=page_number, text="", spans=[])

    tables = _extract_tables_from_native(page, heading_root, heading_stack)
    table_bboxes = [t.bbox for t in tables]
    lines: Dict[Tuple[int, int], List[Tuple[float, float, float, float, str]]] = {}
    for x0, y0, x1, y1, word, block_no, line_no, _ in words:
        if _bbox_overlaps_any((x0, y0, x1, y1), table_bboxes):
            continue
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
        heading_stack=heading_stack,
        heading_root=heading_root,
        table_blocks=tables,
    )


def _build_canonical_page(
    doc_id: str,
    page_number: int,
    line_entries: List[Tuple[str, List[Dict[str, Any]]]],
    source_type: str,
    heading_stack: List[str],
    heading_root: str,
    table_blocks: List[TableBlock],
) -> CanonicalPage:
    spans: List[CanonicalSpan] = []
    text_parts: List[str] = []
    cursor = 0

    for line_text, polygon in line_entries:
        if not line_text:
            continue
        heading_level = _detect_heading_level(line_text)
        if heading_level:
            heading_stack[:] = _update_heading_stack(
                heading_stack, line_text, heading_level
            )
        heading_path = _build_heading_path(heading_root, heading_stack)
        section_id = heading_stack[-1] if heading_stack else heading_root
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
                heading_path=heading_path,
                section_id=section_id,
                is_table=False,
            )
        )
        text_parts.append(line_text)
        cursor = end + 1
    for table in table_blocks:
        heading_path = _build_heading_path(heading_root, heading_stack)
        section_id = heading_stack[-1] if heading_stack else heading_root
        table_text = _table_with_breadcrumb(table.markdown, heading_path)
        start = cursor
        end = start + len(table_text)
        spans.append(
            CanonicalSpan(
                text=table_text,
                char_start=start,
                char_end=end,
                polygons=[{"page_number": page_number, "polygon": table.polygon}],
                source_type=source_type,
                page_number=page_number,
                heading_path=heading_path,
                section_id=section_id,
                is_table=True,
            )
        )
        text_parts.append(table_text)
        cursor = end + 1

    page_text = "\n".join(text_parts)
    return CanonicalPage(
        doc_id=doc_id, page_number=page_number, text=page_text, spans=spans
    )


def _heading_root(pdf_path: str, doc_id: str) -> str:
    base = os.path.basename(pdf_path)
    stem = os.path.splitext(base)[0]
    return stem or doc_id


def _build_heading_path(root: str, stack: List[str]) -> str:
    if not stack:
        return root
    return "/".join([root, *stack])


def _update_heading_stack(stack: List[str], heading: str, level: int) -> List[str]:
    normalized = _normalize_heading(heading)
    if level <= 1:
        return [normalized]
    trimmed = stack[: level - 1]
    return [*trimmed, normalized]


def _detect_heading_level(text: str) -> Optional[int]:
    cleaned = text.strip()
    if len(cleaned) < 3:
        return None
    if re.search(r"\bMD&A\b", cleaned, re.IGNORECASE):
        return 1
    if re.search(r"Management'?s Discussion and Analysis", cleaned, re.IGNORECASE):
        return 1
    if re.match(r"^Note\s+\d+", cleaned, re.IGNORECASE):
        return 1
    if re.search(r"\bSignificant events\b", cleaned, re.IGNORECASE):
        return 2
    if re.search(r"\bItems of note\b", cleaned, re.IGNORECASE):
        return 2
    if re.search(r"\bSignificant legal proceedings\b", cleaned, re.IGNORECASE):
        return 2
    if cleaned.isupper() and len(cleaned) <= 80:
        return 1
    if cleaned.endswith(":") and len(cleaned) <= 80:
        return 2
    if cleaned.istitle() and len(cleaned) <= 80:
        return 2
    if re.match(r"^\d+(\.\d+)*\s+\S", cleaned) and len(cleaned) <= 100:
        return 2
    return None


def _normalize_heading(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _table_with_breadcrumb(markdown: str, heading_path: str) -> str:
    return f"[TABLE] {heading_path}\n{markdown}"


def _extract_tables_from_di(
    payload: Dict[str, Any],
    page_number: int,
) -> List[TableBlock]:
    tables = payload.get("tables", [])
    blocks: List[TableBlock] = []
    for table in tables:
        cells = table.get("cells", [])
        if not cells:
            continue
        if not _table_matches_page(table, page_number):
            continue
        max_row = max(cell.get("rowIndex", 0) for cell in cells)
        max_col = max(cell.get("columnIndex", 0) for cell in cells)
        grid = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
        for cell in cells:
            r = int(cell.get("rowIndex", 0))
            c = int(cell.get("columnIndex", 0))
            grid[r][c] = (cell.get("content") or "").strip()
        markdown = _rows_to_markdown(grid)
        polygon = _polygon_from_di(
            (table.get("boundingRegions") or [{}])[0].get("polygon", [])
        )
        bbox = _bbox_from_polygon(polygon)
        if not bbox:
            continue
        blocks.append(TableBlock(markdown=markdown, polygon=polygon, bbox=bbox))
    return blocks


def _table_matches_page(table: Dict[str, Any], page_number: int) -> bool:
    regions = table.get("boundingRegions") or []
    for region in regions:
        if int(region.get("pageNumber", 0)) == page_number:
            return True
    return False


def _extract_tables_from_native(
    page: fitz.Page,
    heading_root: str,
    heading_stack: List[str],
) -> List[TableBlock]:
    if not hasattr(page, "find_tables"):
        return []
    try:
        tables = page.find_tables()
    except Exception:
        return []
    blocks: List[TableBlock] = []
    for table in tables.tables:
        rows = table.extract()
        if not rows:
            continue
        markdown = _rows_to_markdown(rows)
        bbox = _table_bbox(table.bbox)
        polygon = _polygon_from_bbox(*bbox)
        blocks.append(TableBlock(markdown=markdown, polygon=polygon, bbox=bbox))
    return blocks


def _rows_to_markdown(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    header = ["" if cell is None else str(cell) for cell in rows[0]]
    body = rows[1:] if len(rows) > 1 else []
    header_line = "| " + " | ".join(header) + " |"
    sep_line = "| " + " | ".join(["---"] * len(header)) + " |"
    body_lines = [
        "| " + " | ".join("" if cell is None else str(cell) for cell in row) + " |"
        for row in body
    ]
    return "\n".join([header_line, sep_line, *body_lines])


def _bbox_from_polygon(points: List[Dict[str, float]]) -> Optional[Tuple[float, float, float, float]]:
    if not points:
        return None
    xs = [float(point["x"]) for point in points if "x" in point]
    ys = [float(point["y"]) for point in points if "y" in point]
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _table_bbox(bbox) -> Tuple[float, float, float, float]:
    if hasattr(bbox, "x0"):
        return float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)
    return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])


def _polygon_overlaps_any(
    polygon: List[Dict[str, float]], bboxes: List[Tuple[float, float, float, float]]
) -> bool:
    bbox = _bbox_from_polygon(polygon)
    if not bbox:
        return False
    return _bbox_overlaps_any(bbox, bboxes)


def _bbox_overlaps_any(
    bbox: Tuple[float, float, float, float], bboxes: List[Tuple[float, float, float, float]]
) -> bool:
    for candidate in bboxes:
        if _bboxes_overlap(bbox, candidate):
            return True
    return False


def _bboxes_overlap(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 <= bx1 and ax1 >= bx0 and ay0 <= by1 and ay1 >= by0


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
