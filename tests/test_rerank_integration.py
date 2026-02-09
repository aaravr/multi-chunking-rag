from core.contracts import RetrievedChunk
from retrieval import rerank as rerank_module


class DummyEncoder:
    def predict(self, pairs):
        return [0.9 if "best" in pair[1] else 0.1 for pair in pairs]


def test_rerank_improves_order(monkeypatch):
    candidates = [
        RetrievedChunk(
            chunk_id="c1",
            doc_id="doc",
            page_numbers=[1],
            macro_id=0,
            child_id=0,
            chunk_type="narrative",
            text_content="irrelevant",
            char_start=0,
            char_end=1,
            polygons=[],
            source_type="native",
            heading_path="doc/SEC",
            section_id="SEC",
            score=0.1,
        ),
        RetrievedChunk(
            chunk_id="c2",
            doc_id="doc",
            page_numbers=[1],
            macro_id=0,
            child_id=0,
            chunk_type="narrative",
            text_content="best answer",
            char_start=2,
            char_end=3,
            polygons=[],
            source_type="native",
            heading_path="doc/SEC",
            section_id="SEC",
            score=0.2,
        ),
    ]
    monkeypatch.setattr(
        rerank_module,
        "_get_cross_encoder",
        lambda: DummyEncoder(),
    )
    reranked = rerank_module.rerank("query", candidates)
    assert reranked[0].chunk_id == "c2"
