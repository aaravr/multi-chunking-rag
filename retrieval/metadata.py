from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from core.config import settings
from core.contracts import DocumentFact, RetrievedChunk
from retrieval import vector_search
from storage import repo
from storage.db import get_connection

FACT_ALIASES: Dict[str, List[str]] = {
    "default_currency": ["default currency", "currency"],
    "reporting_period": ["reporting period", "year ended"],
    "accounting_framework": ["accounting framework", "ifrs", "gaap"],
    "units": ["units", "in millions", "in billions"],
    "consolidation_basis": ["consolidation basis", "consolidated"],
}

HEADING_PHRASES = [
    "Consolidated financial statements",
    "Basis of presentation",
    "Significant accounting policies",
    "Presentation currency",
    "Functional currency",
]

CURRENCY_KEYWORDS = [
    "all amounts are in",
    "canadian dollars",
    "unless otherwise stated",
    "unless otherwise indicated",
    "presentation currency",
]


def detect_fact_name(query: str) -> Optional[str]:
    lowered = query.lower()
    for fact_name, aliases in FACT_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            return fact_name
    return None


def handle_metadata_query(
    doc_id: str, query: str, use_cache: bool = True
) -> Tuple[str, List[RetrievedChunk], Dict[str, object]]:
    fact_name = detect_fact_name(query)
    if not fact_name:
        return "", [], {"fact_name": None}
    fact = _fetch_fact(doc_id, fact_name) if use_cache else None
    if fact and fact.status == "found" and fact.source_chunk_id:
        chunk = _fact_to_chunk(fact)
        answer = f"{fact_name}: {fact.value} [C1]"
        return answer, [chunk], {"fact_name": fact_name, "status": fact.status}
    searched_pages = list(range(1, settings.front_matter_pages + 1))
    results = vector_search.search_on_pages(doc_id, query, searched_pages, top_k=3)
    candidates = _filter_narrative(results)
    heading_hits = _heading_phrase_candidates(doc_id)
    candidates.extend(heading_hits)
    bm25_hits = _bm25_narrative_candidates(doc_id, CURRENCY_KEYWORDS, top_k=3)
    candidates.extend(bm25_hits)
    candidates = _dedupe_chunks(candidates)
    if candidates:
        value = _extract_fact_value(fact_name, candidates)
        if value:
            answer = f"{fact_name}: {value} [C1]"
            searched = _merge_pages([searched_pages] + [c.page_numbers for c in candidates])
            return answer, [candidates[0]], {
                "fact_name": fact_name,
                "status": "found",
                "searched_pages": searched,
            }
    status = "not_found"
    if fact and fact.status in {"not_found", "ambiguous"}:
        status = fact.status
    label = "ambiguous" if status == "ambiguous" else "not found"
    searched = _merge_pages([searched_pages] + [c.page_numbers for c in candidates])
    answer = f"{fact_name} {label}. Searched pages: {searched}."
    return answer, [], {
        "fact_name": fact_name,
        "status": status,
        "searched_pages": searched,
    }


def _fetch_fact(doc_id: str, fact_name: str) -> Optional[DocumentFact]:
    with get_connection() as conn:
        return repo.fetch_document_fact(conn, doc_id, fact_name)


def _fact_to_chunk(fact: DocumentFact) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=fact.source_chunk_id or "",
        doc_id=fact.doc_id,
        page_numbers=fact.page_numbers,
        macro_id=0,
        child_id=0,
        chunk_type="narrative",
        text_content=fact.evidence_excerpt or "",
        char_start=0,
        char_end=len(fact.evidence_excerpt or ""),
        polygons=fact.polygons,
        source_type="native",
        score=1.0,
        heading_path="",
        section_id="",
    )


def _extract_fact_value(fact_name: str, chunks: List[RetrievedChunk]) -> Optional[str]:
    if fact_name == "default_currency":
        for chunk in chunks:
            if chunk.chunk_type == "table" or chunk.text_content.lstrip().startswith("[TABLE]"):
                continue
            text = chunk.text_content
            if "Canadian dollars" in text:
                return "Canadian dollars"
            if "U.S. dollars" in text or "US dollars" in text:
                return "U.S. dollars"
    return None


def _filter_narrative(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    return [
        chunk
        for chunk in chunks
        if chunk.chunk_type != "table"
        and not chunk.text_content.lstrip().startswith("[TABLE]")
    ]


def _heading_phrase_candidates(doc_id: str) -> List[RetrievedChunk]:
    rows = []
    with get_connection() as conn:
        with conn.cursor() as cursor:
            for phrase in HEADING_PHRASES:
                cursor.execute(
                    """
                    SELECT chunk_id, doc_id, page_numbers, macro_id, child_id, chunk_type,
                           text_content, char_start, char_end, polygons, source_type,
                           heading_path, section_id, 0.0 AS score
                    FROM chunks
                    WHERE doc_id = %s
                      AND chunk_type <> 'table'
                      AND text_content NOT LIKE '[TABLE]%%'
                      AND (heading_path ILIKE %s OR section_id ILIKE %s)
                    ORDER BY page_numbers[1] NULLS LAST, macro_id, child_id
                    LIMIT 1
                    """,
                    (doc_id, f"%{phrase}%", f"%{phrase}%"),
                )
                row = cursor.fetchone()
                if row:
                    rows.append(row)
    return vector_search._rows_to_chunks(rows)


def _bm25_narrative_candidates(
    doc_id: str, phrases: List[str], top_k: int = 3
) -> List[RetrievedChunk]:
    rows = _fetch_narrative_rows(doc_id)
    if not rows:
        return []
    corpus = [row[6].lower().split() for row in rows]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(" ".join(phrases).lower().split())
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    selected = [rows[idx] for idx, score in ranked if score > 0]
    return vector_search._rows_to_chunks(selected)


def _fetch_narrative_rows(doc_id: str) -> List[Tuple]:
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT chunk_id, doc_id, page_numbers, macro_id, child_id, chunk_type,
                       text_content, char_start, char_end, polygons, source_type,
                       heading_path, section_id, 0.0 AS score
                FROM chunks
                WHERE doc_id = %s
                  AND chunk_type <> 'table'
                  AND text_content NOT LIKE '[TABLE]%%'
                """,
                (doc_id,),
            )
            return cursor.fetchall()


def _dedupe_chunks(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    seen = set()
    deduped = []
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        deduped.append(chunk)
    return deduped


def _merge_pages(page_lists: List[List[int]]) -> List[int]:
    pages = sorted({page for pages in page_lists for page in pages})
    return pages
