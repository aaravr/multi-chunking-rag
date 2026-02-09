from core.contracts import RetrievedChunk
from retrieval import router
from synthesis import openai_client
from synthesis.verifier import verify_coverage


class DummyResponse:
    def __init__(self, content: str):
        class Choice:
            def __init__(self, content: str):
                class Message:
                    def __init__(self, content: str):
                        self.content = content

                self.message = Message(content)

        self.choices = [Choice(content)]


class DummyClient:
    def __init__(self, *_args, **_kwargs):
        pass

    class chat:
        class completions:
            @staticmethod
            def create(*_args, **_kwargs):
                return DummyResponse(
                    "\n".join(
                        [
                            "- Fresco/Gaudet | raw: closed via settlement ($153m) (chunk_id=c1, pages=120)",
                            "- Cerberus | raw: closed via settlement (US$770m) (chunk_id=c2, pages=121)",
                            "- Frayce | raw: closed after appeal process (chunk_id=c3, pages=122)",
                        ]
                    )
                )


def _chunk(chunk_id: str, text: str, page: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="doc",
        page_numbers=[page],
        macro_id=0,
        child_id=0,
        chunk_type="narrative",
        text_content=text,
        char_start=0,
        char_end=len(text),
        polygons=[],
        source_type="native",
        score=0.0,
        heading_path="Notes > Note 21 > Significant legal proceedings",
        section_id="note-21-legal",
    )


def test_closed_matters_regression(monkeypatch):
    query = "Which matters are explicitly 'closed' (and what closed them)"
    intent = router.classify_query(query)
    assert intent.intent == "coverage"
    assert intent.status_filter == "closed"

    anchor = _chunk("a1", "Note 21 Significant legal proceedings", 120)
    chunks = [
        _chunk("c1", "Fresco/Gaudet closed via settlement ($153m).", 120),
        _chunk("c2", "Cerberus closed via settlement (US$770m).", 121),
        _chunk("c3", "Frayce closed after appeal process.", 122),
    ]

    monkeypatch.setattr(router, "bm25_heading_anchor", lambda *_args, **_kwargs: anchor)
    monkeypatch.setattr(
        router.vector_search,
        "fetch_by_section",
        lambda *_args, **_kwargs: chunks,
    )
    monkeypatch.setattr(openai_client, "OpenAI", DummyClient)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    results, debug = router.search_with_intent_debug("doc", query, top_k=3)
    answer, mode_used = openai_client.synthesize_coverage_answer(
        query, results, mode="llm_always", status_filter=intent.status_filter
    )
    verdict, _ = verify_coverage(query, answer, results)

    assert debug["anchor"]["heading_path"]
    assert "Fresco/Gaudet" in answer
    assert "Cerberus" in answer
    assert "Frayce" in answer
    assert mode_used == "llm"
    assert verdict == "YES"
