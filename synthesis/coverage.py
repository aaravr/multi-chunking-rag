import re
from typing import Dict, List, Tuple

from core.contracts import RetrievedChunk

MIN_ITEMS = 3


def extract_coverage_items(
    question: str, chunks: List[RetrievedChunk]
) -> List[Dict[str, object]]:
    seen = set()
    items: List[Dict[str, object]] = []
    for chunk in chunks:
        cleaned = _normalize_text_for_match(chunk.text_content)
        matches = _extract_matches(cleaned)
        for raw, display in matches:
            key = display.lower()
            if key in seen:
                continue
            seen.add(key)
            items.append(
                {
                    "display": display,
                    "raw": raw,
                    "chunk_id": chunk.chunk_id,
                    "page_numbers": chunk.page_numbers,
                    "polygons": chunk.polygons,
                }
            )
    return items


def format_coverage_answer(
    question: str, chunks: List[RetrievedChunk]
) -> str:
    items = extract_coverage_items(question, chunks)
    if not items:
        sections = sorted(
            {c.heading_path or c.section_id or "unknown" for c in chunks}
        )
        pages = sorted({p for c in chunks for p in c.page_numbers})
        return (
            "Not found in retrieved evidence. "
            f"Searched sections: {', '.join(sections)}. "
            f"Searched pages: {pages}."
        )
    lines = []
    for item in items:
        pages = ",".join(str(p) for p in item["page_numbers"])
        lines.append(
            "- {display} | raw: {raw} (chunk_id={chunk_id}, pages={pages})".format(
                display=item["display"],
                raw=item["raw"],
                chunk_id=item["chunk_id"],
                pages=pages,
            )
        )
    return "Litigation matters:\n" + "\n".join(lines)


def _normalize_text_for_match(text: str) -> str:
    dehyphenated = re.sub(r"-\s*\n", "", text)
    collapsed = re.sub(r"\s+", " ", dehyphenated.replace("\n", " "))
    return collapsed.strip()


def _extract_matches(text: str) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []
    for match in _case_name_matches(text):
        results.append(match)
    for match in _heading_matches(text):
        results.append(match)
    for match in _matter_matches(text):
        results.append(match)
    return results


def _case_name_matches(text: str) -> List[Tuple[str, str]]:
    pattern = re.compile(
        r"\b([A-Z][\w&.'-]+(?:\s+[A-Z][\w&.'-]+){0,4})\s+v(?:\.|s\.|s)?\s+([A-Z][\w&.'-]+(?:\s+[A-Z][\w&.'-]+){0,4})\b",
        re.IGNORECASE,
    )
    results = []
    for match in pattern.finditer(text):
        left = match.group(1).strip()
        right = match.group(2).strip()
        display = f"{_title_case(left)} v. {_title_case(right)}"
        results.append((match.group(0).strip(), display))
    return results


def _heading_matches(text: str) -> List[Tuple[str, str]]:
    headings = [
        "class actions",
        "class action",
        "order execution only",
        "fees class actions",
        "legal proceedings",
        "litigation",
    ]
    results: List[Tuple[str, str]] = []
    lower = text.lower()
    for heading in headings:
        if heading in lower:
            pattern = re.compile(re.escape(heading), re.IGNORECASE)
            for match in pattern.finditer(text):
                raw = match.group(0).strip()
                results.append((raw, _title_case(raw)))
    return results


def _matter_matches(text: str) -> List[Tuple[str, str]]:
    pattern = re.compile(
        r"\b([A-Z][\w&.'-]+(?:\s+[A-Z][\w&.'-]+){0,4})\s+"
        r"(matter|class action|litigation|legal proceedings)\b",
        re.IGNORECASE,
    )
    results: List[Tuple[str, str]] = []
    for match in pattern.finditer(text):
        raw = match.group(0).strip()
        display = _title_case(match.group(1).strip())
        results.append((raw, display))
    return results


def _title_case(text: str) -> str:
    return " ".join(word.capitalize() for word in re.split(r"\s+", text) if word)
