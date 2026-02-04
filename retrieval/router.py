import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from core.contracts import RetrievedChunk
from core.config import settings
from retrieval import vector_search
from retrieval.hybrid import (
    bm25_heading_anchor,
    bm25_heading_anchor_candidates,
    hybrid_search,
    lexical_anchor_candidates,
)


@dataclass(frozen=True)
class QueryIntent:
    intent: str
    pages: List[int]
    coverage_type: Optional[str] = None
    status_filter: Optional[str] = None


LOCATION_PATTERNS = [
    re.compile(r"\bpage\s+(\d+)\b", re.IGNORECASE),
    re.compile(r"\bp\.?\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\bpages?\s+(\d+)\s*-\s*(\d+)\b", re.IGNORECASE),
]

COVERAGE_PATTERNS = [
    re.compile(r"\blist all\b", re.IGNORECASE),
    re.compile(r"\ball\s+\w+", re.IGNORECASE),
    re.compile(r"\bprovide all\b", re.IGNORECASE),
    re.compile(r"\bevery\b", re.IGNORECASE),
    re.compile(r"\blitigation events\b", re.IGNORECASE),
    re.compile(r"\blegal proceedings\b", re.IGNORECASE),
    re.compile(r"\bsummarize section\b", re.IGNORECASE),
    re.compile(r"\bitems of note\b", re.IGNORECASE),
    re.compile(r"\bsignificant events\b", re.IGNORECASE),
]

LIST_TERMS = [
    re.compile(r"\blist\b", re.IGNORECASE),
    re.compile(r"\ball\b", re.IGNORECASE),
    re.compile(r"\beach\b", re.IGNORECASE),
]

ATTRIBUTE_TERMS = [
    re.compile(r"\brange\b", re.IGNORECASE),
    re.compile(r"\baggregate\b", re.IGNORECASE),
    re.compile(r"\btotal\b", re.IGNORECASE),
    re.compile(r"\bamount\b", re.IGNORECASE),
    re.compile(r"\bexposure\b", re.IGNORECASE),
    re.compile(r"\bloss(?:es)?\b", re.IGNORECASE),
    re.compile(r"\bliabilit(?:y|ies)\b", re.IGNORECASE),
]

CLOSED_PATTERNS = [
    re.compile(r"\bwhich matters\b", re.IGNORECASE),
    re.compile(r"\bexplicitly closed\b", re.IGNORECASE),
    re.compile(r"\bsettled\b", re.IGNORECASE),
    re.compile(r"\bdiscontinued\b", re.IGNORECASE),
]

LITIGATION_PATTERNS = [
    re.compile(r"\blitigation\b", re.IGNORECASE),
    re.compile(r"\blegal proceedings\b", re.IGNORECASE),
    re.compile(r"\bcontingent liabilities\b", re.IGNORECASE),
    re.compile(r"\bnote\s*21\b", re.IGNORECASE),
    re.compile(r"\bmatters?\b", re.IGNORECASE),
]

ITEMS_OF_NOTE_PHRASES = [
    "Items of note",
    "FDIC special assessment",
    "acquisition-related intangibles",
    "adjusted net income",
]

ITEMS_OF_NOTE_NEGATIVE = [
    "Consolidated financial statements",
]

SECTION_TARGETS = [
    {
        "label": "significant_events",
        "pattern": re.compile(r"\bsignificant events\b", re.IGNORECASE),
        "anchor": "Significant events",
    },
    {
        "label": "items_of_note",
        "pattern": re.compile(r"\bitems of note\b", re.IGNORECASE),
        "anchor": "Items of note",
    },
    {
        "label": "note_21",
        "pattern": re.compile(
            r"\bnote\s*21\b|\bsignificant legal proceedings\b", re.IGNORECASE
        ),
        "anchor": "Note 21",
    },
]

logger = logging.getLogger(__name__)


def classify_query(query: str) -> QueryIntent:
    pages = _extract_pages(query)
    if pages:
        return QueryIntent(intent="location", pages=pages)
    if any(pattern.search(query) for pattern in CLOSED_PATTERNS):
        return QueryIntent(
            intent="coverage", pages=[], coverage_type="list", status_filter="closed"
        )
    for pattern in COVERAGE_PATTERNS:
        if pattern.search(query):
            coverage_type = _classify_coverage_type(query)
            return QueryIntent(
                intent="coverage", pages=[], coverage_type=coverage_type
            )
    return QueryIntent(intent="semantic", pages=[])


def search_with_intent(
    doc_id: str,
    query: str,
    top_k: int = 3,
) -> List[RetrievedChunk]:
    results, _ = search_with_intent_debug(doc_id, query, top_k=top_k)
    return results


def search_with_intent_debug(
    doc_id: str,
    query: str,
    top_k: int = 3,
) -> Tuple[List[RetrievedChunk], Dict[str, object]]:
    debug: Dict[str, object] = {
        "query": query,
        "query_type": None,
        "coverage_type": None,
        "status_filter": None,
        "anchor": {
            "chunk_id": None,
            "page_numbers": [],
            "heading_path": None,
            "section_id": None,
        },
        "anchor_heading_phrase": None,
        "anchor_decisions": [],
        "expansion": None,
        "top_chunks": [],
        "section_targeting": None,
    }
    intent = classify_query(query)
    debug["query_type"] = intent.intent
    debug["coverage_type"] = intent.coverage_type
    debug["status_filter"] = intent.status_filter
    if intent.intent == "location":
        candidates = vector_search.search_on_pages(
            doc_id, query, intent.pages, top_k=100
        )
        if settings.enable_reranker:
            from retrieval.rerank import rerank

            candidates = rerank(query, candidates)
        debug["top_chunks"] = _format_top_chunks(candidates)
        if not debug["expansion"]:
            debug["expansion"] = _summarize_expansion_from_chunks(candidates)
        _log_debug(debug)
        return candidates, debug
    if intent.intent == "coverage":
        anchor = []
        target = _match_section_target(query)
        if target and target["label"] == "items_of_note":
            anchor = _select_items_of_note_anchor(
                doc_id, query, debug["anchor_decisions"]
            )
            if anchor:
                debug["anchor_heading_phrase"] = ITEMS_OF_NOTE_PHRASES
        if not anchor and _is_litigation_query(query):
            anchor_phrase = _litigation_anchor_phrases()
            debug["anchor_heading_phrase"] = anchor_phrase
            anchor_hit = bm25_heading_anchor(doc_id, anchor_phrase)
            if anchor_hit:
                anchor = [anchor_hit]
        if not anchor:
            expanded_query = _expand_coverage_query(query)
            if settings.enable_hybrid_retrieval:
                anchor = hybrid_search(doc_id, expanded_query, top_k=1)
            else:
                anchor = vector_search.search(doc_id, expanded_query, top_k=1)
        if not anchor:
            if intent.coverage_type == "list":
                raise RuntimeError("CoverageListQuery has no anchor heading.")
            _log_debug(debug)
            return [], debug
        debug["anchor"] = _format_anchor(anchor[0])
        candidates, expansion = _expand_from_anchor(
            doc_id,
            anchor[0],
            allow_page_window=intent.coverage_type != "list",
        )
        debug["expansion"] = expansion
        if intent.coverage_type == "list" and not candidates:
            raise RuntimeError("CoverageListQuery expansion returned no chunks.")
        if settings.enable_reranker:
            from retrieval.rerank import rerank

            candidates = rerank(query, candidates)
        debug["top_chunks"] = _format_top_chunks(candidates)
        if not debug["expansion"]:
            debug["expansion"] = _summarize_expansion_from_chunks(candidates)
        _log_debug(debug)
        return candidates, debug
    target = _match_section_target(query)
    if target:
        if target["label"] == "items_of_note":
            anchor = _select_items_of_note_anchor(
                doc_id, query, debug["anchor_decisions"]
            )
            anchor_phrase = ITEMS_OF_NOTE_PHRASES
        else:
            anchor_phrase = target["anchor"]
            anchor = vector_search.search(doc_id, anchor_phrase, top_k=1)
        debug["section_targeting"] = {
            "label": target["label"],
            "anchor_phrase": anchor_phrase,
        }
        if anchor:
            debug["anchor"] = _format_anchor(anchor[0])
            candidates, expansion = _expand_from_anchor(doc_id, anchor[0])
            debug["expansion"] = expansion
            debug["top_chunks"] = _format_top_chunks(candidates)
            _log_debug(debug)
            return candidates, debug
    if settings.enable_hybrid_retrieval:
        candidates = hybrid_search(doc_id, query, top_k=100)
    else:
        candidates = vector_search.search(doc_id, query, top_k=100)
    if settings.enable_reranker:
        from retrieval.rerank import rerank

        candidates = rerank(query, candidates)
    results = candidates[:top_k]
    debug["top_chunks"] = _format_top_chunks(results)
    if not debug["expansion"]:
        debug["expansion"] = _summarize_expansion_from_chunks(results)
    _log_debug(debug)
    return results, debug


def _expand_coverage_query(query: str) -> str:
    expansions = [
        "significant legal proceedings",
        "contingent liabilities",
        "Note 21",
        "litigation",
    ]
    return f"{query} " + " ".join(expansions)


def _classify_coverage_type(query: str) -> str:
    has_list = any(pattern.search(query) for pattern in LIST_TERMS)
    has_attribute = any(pattern.search(query) for pattern in ATTRIBUTE_TERMS)
    if has_list:
        return "list"
    if has_attribute:
        return "attribute"
    return "list"


def _expand_from_anchor(
    doc_id: str, anchor: RetrievedChunk, allow_page_window: bool = True
) -> Tuple[List[RetrievedChunk], Dict[str, object]]:
    expansion: Dict[str, object] = {
        "heading_path": anchor.heading_path,
        "section_id": anchor.section_id,
        "page_numbers": anchor.page_numbers,
        "method": None,
    }
    candidates: List[RetrievedChunk] = []
    if anchor.heading_path or anchor.section_id:
        candidates = vector_search.fetch_by_section(
            doc_id,
            heading_path=anchor.heading_path,
            section_id=anchor.section_id,
        )
        expansion["method"] = "section"
    if not candidates and anchor.macro_id is not None:
        candidates = vector_search.fetch_by_macro_id(doc_id, anchor.macro_id)
        expansion["method"] = "macro_id"
    if not candidates and allow_page_window:
        candidates = vector_search.fetch_by_page_window(
            doc_id, anchor.page_numbers, window=2
        )
        expansion["method"] = "page_window"
    return candidates, expansion


def _format_anchor(anchor: RetrievedChunk) -> Dict[str, object]:
    return {
        "chunk_id": anchor.chunk_id,
        "page_numbers": anchor.page_numbers,
        "heading_path": anchor.heading_path,
        "section_id": anchor.section_id,
        "chunk_type": anchor.chunk_type,
    }


def _format_top_chunks(chunks: List[RetrievedChunk]) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for chunk in chunks[:10]:
        snippet = chunk.text_content[:160].replace("\n", " ").strip()
        results.append(
            {
                "chunk_id": chunk.chunk_id,
                "page_numbers": chunk.page_numbers,
                "heading_path": chunk.heading_path,
                "section_id": chunk.section_id,
                "chunk_type": chunk.chunk_type,
                "snippet": snippet,
            }
        )
    return results


def _summarize_expansion_from_chunks(
    chunks: List[RetrievedChunk],
) -> Dict[str, object]:
    pages = sorted({p for chunk in chunks for p in chunk.page_numbers})
    heading_paths = sorted({c.heading_path for c in chunks if c.heading_path})
    section_ids = sorted({c.section_id for c in chunks if c.section_id})
    return {
        "heading_path": heading_paths,
        "section_id": section_ids,
        "page_numbers": pages,
        "method": "results",
    }


def _match_section_target(query: str) -> Optional[Dict[str, object]]:
    for target in SECTION_TARGETS:
        if target["pattern"].search(query):
            return target
    return None


def _log_debug(debug: Dict[str, object]) -> None:
    try:
        logger.info("Query debug: %s", json.dumps(debug, default=str))
    except Exception:
        logger.info("Query debug: %s", debug)


def _is_litigation_query(query: str) -> bool:
    return any(pattern.search(query) for pattern in LITIGATION_PATTERNS)


def _litigation_anchor_phrases() -> List[str]:
    return [
        "Significant legal proceedings",
        "Contingent liabilities",
        "Note 21",
    ]


def _select_items_of_note_anchor(
    doc_id: str, query: str, decisions: List[Dict[str, object]]
) -> List[RetrievedChunk]:
    if settings.enable_hybrid_retrieval:
        candidates = bm25_heading_anchor_candidates(
            doc_id, ITEMS_OF_NOTE_PHRASES, top_k=25
        )
    else:
        candidates = lexical_anchor_candidates(
            doc_id, ITEMS_OF_NOTE_PHRASES, top_k=25
        )
    explicit_note = _explicit_note_request(query)
    for candidate in candidates:
        reasons = []
        is_table_text = candidate.text_content.lstrip().startswith("[TABLE]")
        if candidate.chunk_type not in {"narrative", "heading"} or is_table_text:
            reasons.append("reject_chunk_type")
        text = f"{candidate.heading_path} {candidate.text_content}"
        lower = text.lower()
        if not _contains_any_phrase(lower, ITEMS_OF_NOTE_PHRASES):
            reasons.append("missing_positive_phrase")
        if not explicit_note:
            if any(neg.lower() in lower for neg in ITEMS_OF_NOTE_NEGATIVE):
                reasons.append("negative_phrase")
            if re.search(r"\\bnote\\s+\\d+\\b", lower):
                reasons.append("negative_note_reference")
        if reasons:
            decisions.append(
                {
                    "chunk_id": candidate.chunk_id,
                    "chunk_type": candidate.chunk_type,
                    "reasons": reasons,
                }
            )
            continue
        decisions.append(
            {
                "chunk_id": candidate.chunk_id,
                "chunk_type": candidate.chunk_type,
                "reasons": ["accepted"],
            }
        )
        return [candidate]
    return []


def _explicit_note_request(query: str) -> bool:
    if re.search(r"\\bnote\\s+\\d+\\b", query, re.IGNORECASE):
        return True
    if re.search(r"derivative instruments", query, re.IGNORECASE):
        return True
    return False


def _contains_any_phrase(text: str, phrases: List[str]) -> bool:
    return any(phrase.lower() in text for phrase in phrases)


def _extract_pages(query: str) -> List[int]:
    pages: List[int] = []
    for pattern in LOCATION_PATTERNS:
        matches = pattern.findall(query)
        for match in matches:
            if isinstance(match, tuple):
                start, end = match
                pages.extend(range(int(start), int(end) + 1))
            else:
                pages.append(int(match))
    return sorted(set(pages))
