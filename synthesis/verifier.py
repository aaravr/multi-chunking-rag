import os
import re
from typing import List, Tuple

from openai import OpenAI

from core.contracts import RetrievedChunk


SYSTEM = """You are an auditing verifier. Decide if the answer is fully supported by the provided sources.
Return YES or NO and a brief rationale."""


def verify_answer(question: str, answer: str, chunks: List[RetrievedChunk]) -> Tuple[str, str]:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for verification.")
    client = OpenAI(api_key=api_key)
    sources = "\n".join([f"[C{i+1}] {c.text_content}" for i, c in enumerate(chunks)])
    prompt = f"Question: {question}\nAnswer: {answer}\nSources:\n{sources}"
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    content = response.choices[0].message.content.strip()
    lines = content.splitlines()
    verdict = lines[0].strip().upper() if lines else "NO"
    rationale = "\n".join(lines[1:]).strip()
    return verdict, rationale


def verify_coverage(
    question: str, answer: str, chunks: List[RetrievedChunk]
) -> Tuple[str, str]:
    items = _parse_coverage_items(answer)
    if not items:
        sections = sorted({c.heading_path or c.section_id or "unknown" for c in chunks})
        pages = sorted({p for c in chunks for p in c.page_numbers})
        return (
            "NO",
            "Not found in retrieved evidence. "
            f"Searched sections: {', '.join(sections)}. "
            f"Searched pages: {pages}.",
        )
    chunk_map = {str(c.chunk_id): c.text_content for c in chunks}
    failures = []
    for item_text, chunk_id in items:
        chunk_text = chunk_map.get(chunk_id, "")
        item_tokens = _normalize_tokens(item_text)
        chunk_tokens = _normalize_tokens(chunk_text)
        if not item_tokens:
            failures.append(f"Missing tokens for '{item_text}': <no tokens>")
            continue
        matched = item_tokens.intersection(chunk_tokens)
        ratio = len(matched) / max(len(item_tokens), 1)
        if ratio < 0.7:
            missing = sorted(item_tokens - matched)
            failures.append(
                "Missing tokens for '{item}': {tokens}".format(
                    item=item_text, tokens=", ".join(missing)
                )
            )
    if failures:
        return "NO", "; ".join(failures)
    return "YES", "All listed items meet normalized token coverage."


def verify_coverage_attribute(
    question: str, answer: str, chunks: List[RetrievedChunk]
) -> Tuple[str, str]:
    cited_chunks = _extract_cited_chunks(answer, chunks)
    if not cited_chunks:
        cited_chunks = chunks
    span = _extract_numeric_span(answer)
    if not span:
        return "NO", "Missing numeric span in answer."
    span_norm = _normalize_span(span)
    for chunk in cited_chunks:
        chunk_norm = _normalize_span(chunk.text_content)
        if span_norm and span_norm in chunk_norm:
            return "YES", "Numeric span matches cited chunks."
    return "NO", "Numeric span not found in cited chunks."


def _parse_coverage_items(answer: str) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for line in answer.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        chunk_match = re.search(r"chunk_id=([0-9a-fA-F-]+)", line)
        if not chunk_match:
            continue
        item_text = line[2:].split(" (chunk_id=")[0].strip()
        if " | raw:" in item_text:
            item_text = item_text.split(" | raw:")[0].strip()
        items.append((item_text, chunk_match.group(1)))
    return items


def _normalize_tokens(text: str) -> set:
    normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return set(normalized.split())


def _extract_cited_chunks(
    answer: str, chunks: List[RetrievedChunk]
) -> List[RetrievedChunk]:
    indices = re.findall(r"\[C(\d+)\]", answer)
    if not indices:
        return []
    results = []
    for idx in indices:
        pos = int(idx) - 1
        if 0 <= pos < len(chunks):
            results.append(chunks[pos])
    return results


def _has_numeric_range(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.lower())
    has_currency = bool(re.search(r"\$|usd|cad", normalized))
    has_number = bool(re.search(r"\d", normalized))
    has_range = bool(
        re.search(r"\bto\b|\bthrough\b|\bbetween\b|\bfrom\b|-\s*\$", normalized)
    )
    return has_currency and has_number and has_range


def _extract_numeric_span(text: str) -> str:
    match = re.search(
        r"(nil|none|zero)?\s*(?:to|through|between|from|-)\s*(?:approximately\s+|approx\.\s+)?\$?\s*[\d,.]+(?:\s*(?:billion|million))?",
        text.lower(),
    )
    if match:
        return match.group(0)
    return ""


def _normalize_span(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", normalized).strip()
