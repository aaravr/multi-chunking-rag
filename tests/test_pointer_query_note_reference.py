import os

from core.contracts import RetrievedChunk
from retrieval import router
from synthesis import openai_client


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
                return DummyResponse("See Note 21 [C1]")


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        doc_id="doc",
        page_numbers=[120],
        macro_id=0,
        child_id=0,
        chunk_type="narrative",
        text_content=text,
        char_start=0,
        char_end=len(text),
        polygons=[],
        source_type="native",
        score=0.0,
        heading_path="Notes/Note 21/Significant legal proceedings",
        section_id="note-21",
    )


def test_pointer_query_note_reference(monkeypatch):
    query = "Where can I find the discussion of significant legal proceedings?"
    intent = router.classify_query(query)
    assert intent.intent == "coverage"
    assert intent.coverage_type == "pointer"

    monkeypatch.setattr(openai_client, "OpenAI", DummyClient)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    os.environ["ENABLE_VERIFIER"] = "false"
    answer = openai_client.synthesize_answer(query, [_chunk("See Note 21")])
    assert "Note 21" in answer
