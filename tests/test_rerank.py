from core.contracts import RetrievedChunk
from retrieval import rerank as rerank_module


class DummyEncoder:
    def __init__(self, scores):
        self._scores = scores

    def predict(self, pairs):
        return self._scores


def test_rerank_preserves_lineage(monkeypatch):
    candidates = [
        RetrievedChunk(
            chunk_id="c1",
            doc_id="doc",
            page_numbers=[1],
            macro_id=0,
            child_id=0,
            chunk_type="narrative",
            text_content="A",
            char_start=0,
            char_end=1,
            polygons=[{"page_number": 1, "polygon": [{"x": 0, "y": 0}]}],
            source_type="native",
            heading_path="doc/SEC",
            section_id="SEC",
            score=0.1,
        ),
        RetrievedChunk(
            chunk_id="c2",
            doc_id="doc",
            page_numbers=[2],
            macro_id=0,
            child_id=0,
            chunk_type="narrative",
            text_content="B",
            char_start=2,
            char_end=3,
            polygons=[{"page_number": 2, "polygon": [{"x": 1, "y": 1}]}],
            source_type="di",
            heading_path="doc/SEC2",
            section_id="SEC2",
            score=0.2,
        ),
    ]
    monkeypatch.setattr(
        rerank_module,
        "_get_cross_encoder",
        lambda: DummyEncoder([0.2, 0.1]),
    )
    reranked = rerank_module.rerank("query", candidates)
    assert reranked[0].chunk_id == "c1"
    assert reranked[0].heading_path == "doc/SEC"
    assert reranked[0].section_id == "SEC"
    assert reranked[0].polygons
