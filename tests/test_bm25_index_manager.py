import time

from rank_bm25 import BM25Okapi

from retrieval import bm25_index


def _rows():
    return [
        (
            "c1",
            "doc",
            [1],
            0,
            0,
            "narrative",
            "Items of note: FDIC special assessment ($0.3 billion after tax).",
            0,
            10,
            [],
            "native",
            "MD&A/Items of note",
            "items-of-note",
            0.0,
        ),
        (
            "c2",
            "doc",
            [2],
            0,
            0,
            "narrative",
            "Adjusted measures are non-GAAP measures and do not include items of note.",
            0,
            10,
            [],
            "native",
            "MD&A/Adjusted measures",
            "adjusted-measures",
            0.0,
        ),
        (
            "c3",
            "doc",
            [3],
            0,
            0,
            "narrative",
            "Risk metrics include LCR and NSFR ratios.",
            0,
            10,
            [],
            "native",
            "Risk",
            "risk",
            0.0,
        ),
    ]


def test_bm25_cached_equivalence(tmp_path, monkeypatch):
    rows = _rows()
    monkeypatch.setattr(bm25_index, "_fetch_chunk_rows", lambda _doc_id: rows)
    monkeypatch.setattr(bm25_index, "_fetch_corpus_version", lambda _doc_id: "v1")

    manager = bm25_index.BM25IndexManager(cache_dir=tmp_path)
    index = manager.build_index("doc")

    corpus = [row[6].lower().split() for row in rows]
    baseline = BM25Okapi(corpus).get_scores("items note".split())
    cached = index.bm25.get_scores("items note".split())

    baseline_ranked = sorted(enumerate(baseline), key=lambda x: x[1], reverse=True)[:2]
    cached_ranked = sorted(enumerate(cached), key=lambda x: x[1], reverse=True)[:2]

    assert [idx for idx, _ in baseline_ranked] == [idx for idx, _ in cached_ranked]


def test_bm25_cached_latency(tmp_path, monkeypatch):
    rows = _rows()
    monkeypatch.setattr(bm25_index, "_fetch_chunk_rows", lambda _doc_id: rows)
    monkeypatch.setattr(bm25_index, "_fetch_corpus_version", lambda _doc_id: "v1")

    manager = bm25_index.BM25IndexManager(cache_dir=tmp_path)
    index = manager.build_index("doc")

    start = time.perf_counter()
    index.bm25.get_scores("items note".split())
    elapsed = time.perf_counter() - start

    assert elapsed < 0.2
