from core.contracts import RetrievedChunk
from retrieval import router
from synthesis import openai_client
from synthesis.verifier import verify_coverage_attribute


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
                    "Answer: nil to approximately $0.7 billion [C1]"
                )


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        doc_id="doc",
        page_numbers=[120],
        macro_id=0,
        chunk_type="narrative",
        text_content=text,
        char_start=0,
        char_end=len(text),
        polygons=[],
        source_type="native",
        score=0.0,
        heading_path="Notes > Note 21 > Litigation",
        section_id="note-21-litigation",
    )


def test_coverage_attribute_query_flow(monkeypatch):
    query = "What is the aggregate range of reasonably possible losses..."
    intent = router.classify_query(query)
    assert intent.intent == "coverage"
    assert intent.coverage_type == "attribute"

    monkeypatch.setattr(openai_client, "OpenAI", DummyClient)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    chunks = [
        _chunk(
            "Reasonably possible losses range from nil to approximately $0.7 billion."
        )
    ]
    answer, mode_used = openai_client.synthesize_coverage_attribute(
        query, chunks
    )
    assert mode_used == "llm"
    assert "nil to approximately $0.7 billion" in answer

    verdict, _ = verify_coverage_attribute(query, answer, chunks)
    assert verdict == "YES"
