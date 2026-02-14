import re
from typing import Dict, List, Optional, Tuple

from core.contracts import DocumentFact, RetrievedChunk

FACT_NAMES = [
    "default_currency",
    "reporting_period",
    "accounting_framework",
    "units",
    "consolidation_basis",
]


def extract_document_facts(doc_id: str, chunks: List[RetrievedChunk]) -> List[DocumentFact]:
    facts: List[DocumentFact] = []
    candidates = _collect_candidates(chunks)
    for fact_name in FACT_NAMES:
        found = candidates.get(fact_name, [])
        if not found:
            facts.append(
                DocumentFact(
                    doc_id=doc_id,
                    fact_name=fact_name,
                    value=None,
                    status="not_found",
                    confidence=0.0,
                    source_chunk_id=None,
                    page_numbers=[],
                    polygons=[],
                    evidence_excerpt=None,
                )
            )
            continue
        values = {entry[0] for entry in found}
        if len(values) > 1:
            facts.append(
                DocumentFact(
                    doc_id=doc_id,
                    fact_name=fact_name,
                    value=None,
                    status="ambiguous",
                    confidence=0.0,
                    source_chunk_id=None,
                    page_numbers=_merge_pages([entry[1] for entry in found]),
                    polygons=[],
                    evidence_excerpt=None,
                )
            )
            continue
        value, chunk = found[0]
        facts.append(
            DocumentFact(
                doc_id=doc_id,
                fact_name=fact_name,
                value=value,
                status="found",
                confidence=0.9,
                source_chunk_id=chunk.chunk_id,
                page_numbers=chunk.page_numbers,
                polygons=chunk.polygons,
                evidence_excerpt=_extract_excerpt(chunk.text_content),
            )
        )
    return facts


def _collect_candidates(chunks: List[RetrievedChunk]) -> Dict[str, List[Tuple[str, RetrievedChunk]]]:
    candidates: Dict[str, List[Tuple[str, RetrievedChunk]]] = {name: [] for name in FACT_NAMES}
    for chunk in chunks:
        if chunk.chunk_type == "table" or chunk.text_content.lstrip().startswith("[TABLE]"):
            continue
        text = chunk.text_content
        currency = _match_default_currency(text)
        if currency:
            candidates["default_currency"].append((currency, chunk))
        units = _match_units(text)
        if units:
            candidates["units"].append((units, chunk))
        framework = _match_framework(text)
        if framework:
            candidates["accounting_framework"].append((framework, chunk))
        consolidation = _match_consolidation(text)
        if consolidation:
            candidates["consolidation_basis"].append((consolidation, chunk))
        period = _match_reporting_period(text)
        if period:
            candidates["reporting_period"].append((period, chunk))
    return candidates


def _match_default_currency(text: str) -> Optional[str]:
    match = re.search(
        r"all amounts are in ([A-Za-z\s.]+?) (dollars|currency)",
        text,
        re.IGNORECASE,
    )
    if match:
        return f"{match.group(1).strip()} {match.group(2).strip()}"
    match = re.search(r"all amounts are in (Canadian|U\.S\.|US) dollars", text, re.IGNORECASE)
    if match:
        return f"{match.group(1).replace('U.S.', 'U.S.').replace('US', 'U.S.')} dollars"
    return None


def _match_units(text: str) -> Optional[str]:
    match = re.search(r"all amounts are in ([A-Za-z\s.]+?) (millions|billions)", text, re.IGNORECASE)
    if match:
        return f"{match.group(2).lower()}"
    return None


def _match_framework(text: str) -> Optional[str]:
    match = re.search(r"(IFRS|GAAP)", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def _match_consolidation(text: str) -> Optional[str]:
    match = re.search(r"consolidated", text, re.IGNORECASE)
    if match and "consolidated" in text.lower():
        return "consolidated"
    return None


def _match_reporting_period(text: str) -> Optional[str]:
    match = re.search(r"year ended ([A-Za-z]+ \d{1,2}, \d{4})", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _extract_excerpt(text: str) -> str:
    return text.strip().splitlines()[0][:200]


def _merge_pages(page_lists: List[List[int]]) -> List[int]:
    pages = sorted({page for pages in page_lists for page in pages})
    return pages
