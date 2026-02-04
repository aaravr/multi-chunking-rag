from retrieval.hybrid import _rrf_merge


class Dummy:
    def __init__(self, chunk_id):
        self.chunk_id = chunk_id
        self.score = 0.0


def test_rrf_merge_orders_by_combined_rank():
    a = Dummy("a")
    b = Dummy("b")
    c = Dummy("c")
    vector_hits = [a, b]
    bm25_hits = [b, c]
    merged = _rrf_merge(vector_hits, bm25_hits, top_k=3, k=1)
    ids = [h.chunk_id for h in merged]
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c"}
