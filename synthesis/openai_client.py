import os
from typing import List

from openai import OpenAI

from core.contracts import RetrievedChunk
from synthesis import prompts


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
    return response.choices[0].message.content.strip()


def _format_sources(chunks: List[RetrievedChunk]) -> str:
    lines: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        lines.append(f"[C{idx}] {chunk.text_content}")
    return "\n".join(lines)
