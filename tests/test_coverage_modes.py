from core.contracts import RetrievedChunk
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
                return DummyResponse("- Fresco | raw: Fresco matter (chunk_id=c1, pages=1)")


def _chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="doc",
        page_numbers=[1],
        macro_id=0,
        child_id=0,
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


def test_coverage_mode_deterministic_bypasses_llm(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise AssertionError("LLM should not be called in deterministic mode.")

    monkeypatch.setattr(openai_client, "OpenAI", _raise)
    chunks = [
        _chunk("c1", "Fresco matter"),
        _chunk("c2", "Cerberus matter"),
        _chunk("c3", "Frayce matter"),
    ]
    answer, mode_used = openai_client.synthesize_coverage_answer(
        "list all litigation events", chunks, mode="deterministic"
    )
    assert mode_used == "deterministic"
    assert "Fresco" in answer


def test_coverage_mode_llm_always_uses_llm(monkeypatch):
    monkeypatch.setattr(openai_client, "OpenAI", DummyClient)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    chunks = [_chunk("c1", "Fresco matter")]
    answer, mode_used = openai_client.synthesize_coverage_answer(
        "list all litigation events", chunks, mode="llm_always"
    )
    assert mode_used == "llm"
    assert "chunk_id=c1" in answer
