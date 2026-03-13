"""Tests for the Classifier Agent (MASTER_PROMPT §4.8).

Validates:
- Deterministic classification from filename patterns
- Deterministic classification from content signals
- Embedding-based similarity matching (ModernBERT)
- SGDClassifier incremental learning
- Memory-based classification (keyword/filename)
- Feedback reinforcement loop
- LLM response parsing
- Memory export/import with embeddings
- Fallback to unknown for unrecognizable documents
"""

import json

import numpy as np
import pytest

from agents.classifier_agent import (
    ClassificationMemory,
    ClassifierAgent,
    EMBEDDING_SIMILARITY_THRESHOLD,
    MEMORY_MIN_OBSERVATIONS,
    MEMORY_TRUST_THRESHOLD,
    SGD_MIN_SAMPLES,
    reset_classification_memory,
)
from agents.contracts import ClassificationResult, new_id
from agents.message_bus import MessageBus


@pytest.fixture(autouse=True)
def reset_memory():
    """Reset singleton classification memory between tests."""
    reset_classification_memory()
    yield
    reset_classification_memory()


@pytest.fixture
def memory():
    return ClassificationMemory(embedding_dim=768)


@pytest.fixture
def small_memory():
    """Memory with small embedding dim for fast tests."""
    return ClassificationMemory(embedding_dim=4)


@pytest.fixture
def classifier():
    bus = MessageBus()
    return ClassifierAgent(bus=bus, gateway=None, memory=ClassificationMemory())


# ── Deterministic filename classification ─────────────────────────────


class TestDeterministicFilename:
    def test_10k_filename(self, classifier):
        result = classifier.classify(
            doc_id="doc1", filename="Company_10-K_2024.pdf"
        )
        assert result.document_type == "10-K"
        assert result.classification_label == "sec_filing"
        assert result.classification_method == "deterministic"
        assert result.confidence >= 0.80

    def test_10q_filename(self, classifier):
        result = classifier.classify(
            doc_id="doc1", filename="Q3_10-Q_Filing.pdf"
        )
        assert result.document_type == "10-Q"
        assert result.classification_label == "sec_filing"

    def test_annual_report_filename(self, classifier):
        result = classifier.classify(
            doc_id="doc1", filename="Annual_Report_2024.pdf"
        )
        assert result.document_type == "annual_report"
        assert result.classification_label == "financial_report"

    def test_pillar3_filename(self, classifier):
        result = classifier.classify(
            doc_id="doc1", filename="Pillar 3 Disclosures 2024.pdf"
        )
        assert result.document_type == "pillar3_disclosure"
        assert result.classification_label == "basel_regulatory"

    def test_contract_filename(self, classifier):
        result = classifier.classify(
            doc_id="doc1", filename="Service_Agreement_v2.pdf"
        )
        assert result.document_type == "contract"
        assert result.classification_label == "legal_document"

    def test_esg_filename(self, classifier):
        result = classifier.classify(
            doc_id="doc1", filename="ESG_Report_2024.pdf"
        )
        assert result.document_type == "esg_report"
        assert result.classification_label == "sustainability"


# ── Deterministic content classification ──────────────────────────────


class TestDeterministicContent:
    def test_sec_header_in_content(self, classifier):
        result = classifier.classify(
            doc_id="doc1",
            filename="filing.pdf",
            front_matter_text="UNITED STATES SECURITIES AND EXCHANGE COMMISSION\nWashington, D.C.",
        )
        assert result.classification_label == "sec_filing"
        assert result.confidence >= 0.90

    def test_10k_form_in_content(self, classifier):
        result = classifier.classify(
            doc_id="doc1",
            filename="document.pdf",
            front_matter_text="Form 10-K\nANNUAL REPORT PURSUANT TO SECTION 13",
        )
        assert result.document_type == "10-K"
        assert result.confidence >= 0.90

    def test_pillar3_in_content(self, classifier):
        result = classifier.classify(
            doc_id="doc1",
            filename="report.pdf",
            front_matter_text="Pillar 3 Disclosures as at 31 December 2024",
        )
        assert result.document_type == "pillar3_disclosure"
        assert result.classification_label == "basel_regulatory"

    def test_content_overrides_weak_filename(self, classifier):
        """Content signals should override weak filename matches."""
        result = classifier.classify(
            doc_id="doc1",
            filename="report.pdf",
            front_matter_text="ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d)",
        )
        assert result.document_type == "10-K"
        assert result.confidence >= 0.90


# ── Embedding-based similarity matching ───────────────────────────────


class TestEmbeddingSimilarity:
    def test_store_and_lookup_by_embedding(self, small_memory):
        """Storing an embedding and looking up an identical vector should match."""
        emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        small_memory.store_embedding(emb, "10-K", "sec_filing")

        result = small_memory.lookup_by_embedding(emb)
        assert result is not None
        doc_type, label, similarity = result
        assert doc_type == "10-K"
        assert label == "sec_filing"
        assert similarity >= 0.99

    def test_similar_embedding_matches(self, small_memory):
        """A very similar embedding should match."""
        emb1 = np.array([1.0, 0.1, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([1.0, 0.15, 0.0, 0.0], dtype=np.float32)
        small_memory.store_embedding(emb1, "10-K", "sec_filing")

        result = small_memory.lookup_by_embedding(emb2)
        assert result is not None
        assert result[0] == "10-K"
        assert result[2] > EMBEDDING_SIMILARITY_THRESHOLD

    def test_dissimilar_embedding_no_match(self, small_memory):
        """An orthogonal embedding should not match."""
        emb1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        small_memory.store_embedding(emb1, "10-K", "sec_filing")

        result = small_memory.lookup_by_embedding(emb2)
        assert result is None

    def test_best_match_returned(self, small_memory):
        """When multiple embeddings are stored, the most similar one wins."""
        emb_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        query = np.array([0.95, 0.05, 0.0, 0.0], dtype=np.float32)

        small_memory.store_embedding(emb_a, "10-K", "sec_filing")
        small_memory.store_embedding(emb_b, "contract", "legal_document")

        result = small_memory.lookup_by_embedding(query)
        assert result is not None
        assert result[0] == "10-K"

    def test_embedding_dim_mismatch_rejected(self, small_memory):
        """Embedding with wrong dimension should be rejected."""
        emb = np.array([1.0, 0.0], dtype=np.float32)  # dim=2, expected dim=4
        small_memory.store_embedding(emb, "10-K", "sec_filing")
        assert len(small_memory._embeddings) == 0


# ── SGD Classifier ────────────────────────────────────────────────────


class TestSGDClassifier:
    def test_sgd_not_ready_with_few_samples(self, small_memory):
        """SGD should not predict until it has enough samples."""
        emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        small_memory.store_embedding(emb, "10-K", "sec_filing")
        result = small_memory.predict_sgd(emb)
        assert result is None

    def test_sgd_predicts_after_enough_samples(self, small_memory):
        """After enough training samples, SGD should produce predictions."""
        # Train with enough diverse samples
        rng = np.random.RandomState(42)
        for _ in range(SGD_MIN_SAMPLES):
            emb = rng.randn(4).astype(np.float32)
            emb[0] = abs(emb[0]) + 1.0  # bias toward positive x
            small_memory.store_embedding(emb, "10-K", "sec_filing")

        for _ in range(SGD_MIN_SAMPLES):
            emb = rng.randn(4).astype(np.float32)
            emb[1] = abs(emb[1]) + 1.0  # bias toward positive y
            small_memory.store_embedding(emb, "contract", "legal_document")

        # Now predict
        query = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)
        result = small_memory.predict_sgd(query)
        assert result is not None
        doc_type, label, confidence = result
        assert confidence > 0

    def test_sgd_trains_incrementally(self, small_memory):
        """Each store_embedding call should increment the SGD sample count."""
        assert small_memory._sgd_sample_count == 0
        emb1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        small_memory.store_embedding(emb1, "10-K", "sec_filing")
        # With only 1 class, SGD defers fitting but still counts
        assert small_memory._sgd_sample_count == 1
        # Add second class to trigger bootstrap fit
        emb2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        small_memory.store_embedding(emb2, "contract", "legal_document")
        assert small_memory._sgd_sample_count == 2
        assert small_memory._sgd_fitted is True


# ── Memory-based classification (keyword/filename) ────────────────────


class TestMemoryClassification:
    def test_memory_stores_pattern(self, memory):
        pid = memory.store_pattern(
            document_type="10-K",
            classification_label="sec_filing",
            filename="Company_10K.pdf",
            title_keywords=["Annual Report"],
        )
        assert pid is not None
        patterns = memory.get_all_patterns()
        assert len(patterns) == 1
        assert patterns[0].document_type == "10-K"

    def test_memory_lookup_requires_min_observations(self, memory):
        memory.store_pattern(
            document_type="10-K",
            classification_label="sec_filing",
            filename="test.pdf",
        )
        result = memory.lookup(filename="test.pdf")
        assert result is None

    def test_memory_lookup_returns_trusted_pattern(self, memory):
        pid = memory.store_pattern(
            document_type="10-K",
            classification_label="sec_filing",
            filename="test.pdf",
        )
        for _ in range(MEMORY_MIN_OBSERVATIONS):
            memory.store_pattern(
                document_type="10-K",
                classification_label="sec_filing",
                filename="test.pdf",
            )
            memory.record_feedback(pid, correct=True)

        result = memory.lookup(filename="test.pdf")
        assert result is not None
        assert result.document_type == "10-K"

    def test_memory_feedback_updates_accuracy(self, memory):
        pid = memory.store_pattern(
            document_type="10-K",
            classification_label="sec_filing",
            filename="test.pdf",
        )
        memory.record_feedback(pid, correct=True)
        memory.record_feedback(pid, correct=True)
        memory.record_feedback(pid, correct=False)

        patterns = memory.get_all_patterns()
        assert patterns[0].success_count == 2

    def test_memory_deduplicates_same_pattern(self, memory):
        pid1 = memory.store_pattern(
            document_type="10-K",
            classification_label="sec_filing",
            filename="test.pdf",
        )
        pid2 = memory.store_pattern(
            document_type="10-K",
            classification_label="sec_filing",
            filename="test.pdf",
        )
        assert pid1 == pid2
        patterns = memory.get_all_patterns()
        assert len(patterns) == 1
        assert patterns[0].total_count == 2


# ── Memory export/import ─────────────────────────────────────────────


class TestMemoryPersistence:
    def test_export_import_patterns_roundtrip(self, memory):
        memory.store_pattern(
            document_type="10-K",
            classification_label="sec_filing",
            filename="filing.pdf",
            title_keywords=["Annual Report"],
        )
        memory.store_pattern(
            document_type="contract",
            classification_label="legal_document",
            filename="agreement.pdf",
        )

        exported = memory.export_json()
        data = json.loads(exported)
        assert len(data["patterns"]) == 2

        new_memory = ClassificationMemory()
        count = new_memory.import_json(exported)
        assert count == 2
        assert len(new_memory.get_all_patterns()) == 2

    def test_export_import_embeddings_roundtrip(self, small_memory):
        emb = np.array([1.0, 0.0, 0.5, 0.0], dtype=np.float32)
        small_memory.store_embedding(emb, "10-K", "sec_filing")

        exported = small_memory.export_json()
        data = json.loads(exported)
        assert len(data["embeddings"]) == 1

        new_memory = ClassificationMemory(embedding_dim=4)
        count = new_memory.import_json(exported)
        assert count >= 1
        assert len(new_memory._embeddings) == 1

        # Verify lookup still works after import
        result = new_memory.lookup_by_embedding(emb)
        assert result is not None
        assert result[0] == "10-K"

    def test_memory_stats_include_ml(self, small_memory):
        # Need 2 classes for SGD to fit
        emb1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        small_memory.store_embedding(emb1, "10-K", "sec_filing")
        small_memory.store_embedding(emb2, "contract", "legal_document")
        small_memory.store_pattern("10-K", "sec_filing", "a.pdf")

        stats = small_memory.get_stats()
        assert stats["total_patterns"] == 1
        assert stats["embedding_count"] == 2
        assert stats["sgd_sample_count"] == 2
        assert stats["sgd_fitted"] is True


# ── LLM response parsing ─────────────────────────────────────────────


class TestLLMResponseParsing:
    def test_parse_clean_json(self, classifier):
        response = '{"document_type": "10-K", "classification_label": "sec_filing", "confidence": 0.92}'
        result = classifier._parse_llm_response(response)
        assert result == ("10-K", "sec_filing", 0.92)

    def test_parse_json_in_code_block(self, classifier):
        response = '```json\n{"document_type": "contract", "classification_label": "legal_document", "confidence": 0.85}\n```'
        result = classifier._parse_llm_response(response)
        assert result == ("contract", "legal_document", 0.85)

    def test_parse_caps_confidence(self, classifier):
        response = '{"document_type": "10-K", "classification_label": "sec_filing", "confidence": 0.99}'
        result = classifier._parse_llm_response(response)
        assert result[2] == 0.95

    def test_parse_invalid_json(self, classifier):
        result = classifier._parse_llm_response("This is not JSON")
        assert result is None


# ── Fallback behavior ─────────────────────────────────────────────────


class TestFallback:
    def test_unknown_document_gets_default(self, classifier):
        result = classifier.classify(
            doc_id="doc1",
            filename="random_file.pdf",
            front_matter_text="Some random text that matches nothing",
        )
        assert result.document_type == "unknown"
        assert result.classification_label == "unclassified"
        assert result.confidence == 0.0
        assert result.classification_method == "default"

    def test_low_confidence_filename_still_returns_result(self, classifier):
        result = classifier.classify(
            doc_id="doc1",
            filename="policy_update.pdf",
        )
        assert result.document_type == "policy_document"
        assert result.classification_label == "governance"
        assert result.classification_method == "deterministic"


# ── Message bus integration ───────────────────────────────────────────


class TestMessageBusIntegration:
    def test_handle_message(self, classifier):
        from agents.message_bus import create_message

        msg = create_message(
            from_agent="orchestrator",
            to_agent="classifier",
            message_type="classification_request",
            payload={
                "doc_id": "doc1",
                "filename": "Annual_Report_2024.pdf",
                "front_matter_text": "",
                "page_count": 50,
            },
            query_id=new_id(),
        )
        result = classifier.handle_message(msg)
        assert isinstance(result, ClassificationResult)
        assert result.document_type == "annual_report"


# ── Self-learning integration ─────────────────────────────────────────


class TestSelfLearning:
    def test_classification_stores_in_memory(self):
        memory = ClassificationMemory()
        bus = MessageBus()
        classifier = ClassifierAgent(bus=bus, gateway=None, memory=memory)

        classifier.classify(doc_id="doc1", filename="Company_10-K.pdf")

        patterns = memory.get_all_patterns()
        assert len(patterns) >= 1
        assert any(p.document_type == "10-K" for p in patterns)

    def test_repeated_classifications_increase_count(self):
        memory = ClassificationMemory()
        bus = MessageBus()
        classifier = ClassifierAgent(bus=bus, gateway=None, memory=memory)

        classifier.classify(doc_id="doc1", filename="Company_10-K.pdf")
        classifier.classify(doc_id="doc2", filename="Company_10-K.pdf")

        patterns = memory.get_all_patterns()
        ten_k_patterns = [p for p in patterns if p.document_type == "10-K"]
        assert len(ten_k_patterns) >= 1
        assert ten_k_patterns[0].total_count >= 2

    def test_feedback_improves_accuracy(self):
        memory = ClassificationMemory()
        bus = MessageBus()
        classifier = ClassifierAgent(bus=bus, gateway=None, memory=memory)

        classifier.classify(doc_id="doc1", filename="Company_10-K.pdf")

        patterns = memory.get_all_patterns()
        pid = patterns[0].pattern_id

        classifier.handle_feedback(pid, correct=True)
        classifier.handle_feedback(pid, correct=True)

        updated = memory.get_all_patterns()
        assert updated[0].success_count == 2


# ── Embedding-based self-learning via classifier ─────────────────────


class TestEmbeddingClassification:
    """Tests for embedding-based classification in the ClassifierAgent.

    Uses a mock embedder to avoid loading the full ModernBERT model in tests.
    """

    def _make_classifier_with_mock_embedder(self, memory, embed_fn):
        """Create a ClassifierAgent with a mock embedder."""

        class MockEmbedder:
            def embed_text(self, text):
                return embed_fn(text)

        bus = MessageBus()
        return ClassifierAgent(
            bus=bus, gateway=None, memory=memory, embedder=MockEmbedder()
        )

    def test_embedding_stored_on_classify(self):
        """Classification should store the embedding in memory."""
        memory = ClassificationMemory(embedding_dim=4)
        emb = [1.0, 0.0, 0.0, 0.0]
        classifier = self._make_classifier_with_mock_embedder(
            memory, lambda text: emb
        )

        classifier.classify(
            doc_id="doc1",
            filename="Form_10-K.pdf",
            front_matter_text="ANNUAL REPORT PURSUANT TO SECTION 13",
        )

        assert len(memory._embeddings) == 1
        assert memory._sgd_sample_count == 1

    def test_embedding_similarity_used_for_matching(self):
        """After storing an embedding, a similar document should match via embedding."""
        memory = ClassificationMemory(embedding_dim=4)
        call_count = [0]

        def embed_fn(text):
            call_count[0] += 1
            if call_count[0] <= 1:
                return [1.0, 0.0, 0.0, 0.0]
            return [0.99, 0.01, 0.0, 0.0]  # very similar

        classifier = self._make_classifier_with_mock_embedder(memory, embed_fn)

        # First classification: deterministic (10-K in filename)
        classifier.classify(
            doc_id="doc1",
            filename="Form_10-K.pdf",
            front_matter_text="ANNUAL REPORT PURSUANT TO SECTION 13",
        )

        # Second classification: unknown filename, but similar embedding
        result = classifier.classify(
            doc_id="doc2",
            filename="unknown_doc.pdf",
            front_matter_text="Some financial report text",
        )

        assert result.classification_method == "embedding_similarity"
        assert result.document_type == "10-K"
        assert result.confidence > EMBEDDING_SIMILARITY_THRESHOLD
