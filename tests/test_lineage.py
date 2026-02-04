import torch

from core.contracts import CanonicalPage, CanonicalSpan
from embedding import late_chunking


class DummyTokenized:
    def __init__(self, offsets):
        self.offsets = offsets
        self.attention_mask = torch.ones(1, len(offsets), dtype=torch.long)
        self.input_ids = torch.ones(1, len(offsets), dtype=torch.long)


class DummyEmbedder:
    def __init__(self, max_length=8192):
        self.max_length = max_length

    def tokenize_full(self, text):
        parts = text.split()
        offsets = []
        cursor = 0
        for part in parts:
            start = text.index(part, cursor)
            end = start + len(part)
            offsets.append((start, end))
            cursor = end
        return offsets

    def tokenize(self, text):
        return DummyTokenized(self.tokenize_full(text))

    def encode(self, tokenized):
        return torch.ones(len(tokenized.offsets), 3)


def test_chunk_lineage_fields_present(monkeypatch):
    monkeypatch.setattr(late_chunking, "ModernBERTEmbedder", DummyEmbedder)

    span = CanonicalSpan(
        text="Hello world",
        char_start=0,
        char_end=11,
        polygons=[{"page_number": 1, "polygon": [{"x": 0, "y": 0}]}],
        source_type="native",
        page_number=1,
        heading_path="doc/SECTION",
        section_id="SECTION",
        is_table=False,
    )
    page = CanonicalPage(
        doc_id="doc-1",
        page_number=1,
        text="Hello world",
        spans=[span],
    )
    chunks = late_chunking.late_chunk_embeddings([page], macro_max_tokens=8, child_target_tokens=4)
    assert chunks
    chunk = chunks[0]
    assert chunk.doc_id
    assert chunk.page_numbers
    assert chunk.char_start is not None
    assert chunk.char_end is not None
    assert chunk.polygons
    assert chunk.source_type
    assert chunk.macro_id is not None
    assert chunk.child_id is not None
    assert chunk.embedding_model
    assert chunk.embedding_dim
    assert chunk.heading_path
    assert chunk.section_id
