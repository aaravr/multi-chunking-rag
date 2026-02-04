import os
from typing import List, Optional

from openai import OpenAI

from core.contracts import RetrievedChunk
from typing import Tuple

from synthesis import prompts
from synthesis import coverage
from synthesis.verifier import verify_answer


def synthesize_answer(question: str, chunks: List[RetrievedChunk]) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for synthesis.")
    client = OpenAI(api_key=api_key)
    sources = _format_sources(chunks)
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": prompts.SYSTEM},
            {"role": "user", "content": prompts.USER.format(question=question, sources=sources)},
        ],
        temperature=0.1,
    )
    answer = response.choices[0].message.content.strip()
    if os.getenv("ENABLE_VERIFIER", "").lower() in {"1", "true", "yes", "y", "on"}:
        verdict, rationale = verify_answer(question, answer, chunks)
        return f"{answer}\n\nVerifier: {verdict}\n{rationale}"
    return answer


def synthesize_coverage_answer(
    question: str,
    chunks: List[RetrievedChunk],
    min_items: int = coverage.MIN_ITEMS,
    mode: str = "llm_fallback",
    status_filter: Optional[str] = None,
) -> Tuple[str, str]:
    deterministic_answer = coverage.format_coverage_answer(question, chunks)
    deterministic_items = coverage.extract_coverage_items(question, chunks)
    mode_normalized = (mode or "llm_fallback").strip().lower()
    if mode_normalized == "deterministic":
        return deterministic_answer, "deterministic"

    api_key = os.getenv("OPENAI_API_KEY", "")
    if mode_normalized not in {"deterministic", "llm_fallback", "llm_always"}:
        mode_normalized = "llm_fallback"
    if mode_normalized == "llm_fallback" and len(deterministic_items) >= min_items:
        return deterministic_answer, "deterministic"
    if not api_key:
        return deterministic_answer, "deterministic"

    client = OpenAI(api_key=api_key)
    sources = _format_sources_with_ids(chunks)
    if status_filter == "closed":
        system_prompt = prompts.COVERAGE_CLOSED_SYSTEM
        user_prompt = prompts.COVERAGE_CLOSED_USER
    else:
        system_prompt = prompts.COVERAGE_SYSTEM
        user_prompt = prompts.COVERAGE_USER
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_prompt.format(question=question, sources=sources),
            },
        ],
        temperature=0.1,
    )
    answer = response.choices[0].message.content.strip()
    if answer:
        return answer, "llm"
    return deterministic_answer, "deterministic"


def synthesize_coverage_attribute(
    question: str, chunks: List[RetrievedChunk]
) -> Tuple[str, str]:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for synthesis.")
    client = OpenAI(api_key=api_key)
    sources = _format_sources(chunks)
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": prompts.COVERAGE_ATTRIBUTE_SYSTEM},
            {
                "role": "user",
                "content": prompts.COVERAGE_ATTRIBUTE_USER.format(
                    question=question, sources=sources
                ),
            },
        ],
        temperature=0.1,
    )
    answer = response.choices[0].message.content.strip()
    return answer, "llm"


def _format_sources(chunks: List[RetrievedChunk]) -> str:
    lines: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        lines.append(f"[C{idx}] {chunk.text_content}")
    return "\n".join(lines)


def _format_sources_with_ids(chunks: List[RetrievedChunk]) -> str:
    lines: List[str] = []
    for chunk in chunks:
        pages = ",".join(str(p) for p in chunk.page_numbers)
        lines.append(
            f"chunk_id={chunk.chunk_id} pages={pages} text={chunk.text_content}"
        )
    return "\n".join(lines)
