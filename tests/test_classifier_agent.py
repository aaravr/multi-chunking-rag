"""Tests for the Classifier Agent (MASTER_PROMPT §4.8).

Validates:
- LangGraph flow: deterministic → pgvector_similarity → sgd → llm → fallback
- LangChain ChatPromptTemplate for LLM prompts
- pgvector embedding similarity (in-memory fallback mode for unit tests)
- sklearn SGDClassifier incremental learning
- Pattern memory + feedback loop
- Memory export/import with embeddings
- Neo4j knowledge graph integration (mocked)
"""

import json
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from agents.classifier_agent import (
    ClassificationMemory,
    ClassifierAgent,
    EmbeddingStore,
    PatternStore,
    SGDClassifierWrapper,
    EMBEDDING_SIMILARITY_THRESHOLD,
    MEMORY_MIN_OBSERVATIONS,
    MEMORY_TRUST_THRESHOLD,
    SGD_MIN_SAMPLES,
    CLASSIFICATION_PROMPT,
    reset_classification_memory,
    _neo4j_available,
    _neo4j_store_classification,
)
from agents.contracts import ClassificationResult, new_id
from agents.message_bus import MessageBus


@pytest.fixture(autouse=True)
def reset_memory():
    reset_classification_memory()
    yield
    reset_classification_memory()


@pytest.fixture
def memory():
    return ClassificationMemory(embedding_dim=768, use_db=False)


@pytest.fixture
def small_memory():
    return ClassificationMemory(embedding_dim=4, use_db=False)


@pytest.fixture
def classifier():
    bus = MessageBus()
    return ClassifierAgent(bus=bus, gateway=None, memory=ClassificationMemory(use_db=False))


# ── LangChain prompt template ─────────────────────────────────────────


class TestLangChainPrompt:
    def test_prompt_template_formats(self):
        """LangChain ChatPromptTemplate should produce system + human messages."""
        messages = CLASSIFICATION_PROMPT.format_messages(
            filename="test.pdf",
            page_count=10,
            front_matter_text="Some text",
        )
        assert len(messages) == 2
        assert messages[0].type == "system"
        assert messages[1].type == "human"
        assert "test.pdf" in messages[1].content
        assert "Some text" in messages[1].content

    def test_prompt_template_has_required_fields(self):
        """Template should reference all input variables."""
        input_vars = CLASSIFICATION_PROMPT.input_variables
        assert "filename" in input_vars
        assert "page_count" in input_vars
        assert "front_matter_text" in input_vars


# ── LangGraph flow: deterministic tier ────────────────────────────────


class TestDeterministicFilename:
    def test_10k_filename(self, classifier):
        result = classifier.classify(doc_id="doc1", filename="Company_10-K_2024.pdf")
        assert result.document_type == "10-K"
        assert result.classification_label == "sec_filing"
        assert result.classification_method == "deterministic"
        assert result.confidence >= 0.80

    def test_10q_filename(self, classifier):
        result = classifier.classify(doc_id="doc1", filename="Q3_10-Q_Filing.pdf")
        assert result.document_type == "10-Q"
        assert result.classification_label == "sec_filing"

    def test_annual_report_filename(self, classifier):
        result = classifier.classify(doc_id="doc1", filename="Annual_Report_2024.pdf")
        assert result.document_type == "annual_report"
        assert result.classification_label == "financial_report"

    def test_pillar3_filename(self, classifier):
        result = classifier.classify(doc_id="doc1", filename="Pillar 3 Disclosures 2024.pdf")
        assert result.document_type == "pillar3_disclosure"
        assert result.classification_label == "basel_regulatory"

    def test_contract_filename(self, classifier):
        result = classifier.classify(doc_id="doc1", filename="Service_Agreement_v2.pdf")
        assert result.document_type == "contract"
        assert result.classification_label == "legal_document"

    def test_esg_filename(self, classifier):
        result = classifier.classify(doc_id="doc1", filename="ESG_Report_2024.pdf")
        assert result.document_type == "esg_report"
        assert result.classification_label == "sustainability"


class TestDeterministicContent:
    def test_sec_header_in_content(self, classifier):
        result = classifier.classify(
            doc_id="doc1", filename="filing.pdf",
            front_matter_text="UNITED STATES SECURITIES AND EXCHANGE COMMISSION\nWashington, D.C.",
        )
        assert result.classification_label == "sec_filing"
        assert result.confidence >= 0.90

    def test_10k_form_in_content(self, classifier):
        result = classifier.classify(
            doc_id="doc1", filename="document.pdf",
            front_matter_text="Form 10-K\nANNUAL REPORT PURSUANT TO SECTION 13",
        )
        assert result.document_type == "10-K"
        assert result.confidence >= 0.90

    def test_pillar3_in_content(self, classifier):
        result = classifier.classify(
            doc_id="doc1", filename="report.pdf",
            front_matter_text="Pillar 3 Disclosures as at 31 December 2024",
        )
        assert result.document_type == "pillar3_disclosure"
        assert result.classification_label == "basel_regulatory"

    def test_content_overrides_weak_filename(self, classifier):
        result = classifier.classify(
            doc_id="doc1", filename="report.pdf",
            front_matter_text="ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d)",
        )
        assert result.document_type == "10-K"
        assert result.confidence >= 0.90


# ── pgvector embedding similarity (in-memory fallback) ────────────────


class TestPgvectorSimilarity:
    def test_store_and_lookup_by_embedding(self, small_memory):
        emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        small_memory.store_embedding(emb, "10-K", "sec_filing")
        result = small_memory.lookup_by_embedding(emb)
        assert result is not None
        doc_type, label, similarity = result
        assert doc_type == "10-K"
        assert label == "sec_filing"
        assert similarity >= 0.99

    def test_similar_embedding_matches(self, small_memory):
        emb1 = np.array([1.0, 0.1, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([1.0, 0.15, 0.0, 0.0], dtype=np.float32)
        small_memory.store_embedding(emb1, "10-K", "sec_filing")
        result = small_memory.lookup_by_embedding(emb2)
        assert result is not None
        assert result[0] == "10-K"
        assert result[2] > EMBEDDING_SIMILARITY_THRESHOLD

    def test_dissimilar_embedding_no_match(self, small_memory):
        emb1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        small_memory.store_embedding(emb1, "10-K", "sec_filing")
        result = small_memory.lookup_by_embedding(emb2)
        assert result is None

    def test_best_match_returned(self, small_memory):
        emb_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        query = np.array([0.95, 0.05, 0.0, 0.0], dtype=np.float32)
        small_memory.store_embedding(emb_a, "10-K", "sec_filing")
        small_memory.store_embedding(emb_b, "contract", "legal_document")
        result = small_memory.lookup_by_embedding(query)
        assert result is not None
        assert result[0] == "10-K"

    def test_embedding_dim_mismatch_rejected(self, small_memory):
        emb = np.array([1.0, 0.0], dtype=np.float32)
        small_memory.store_embedding(emb, "10-K", "sec_filing")
        assert len(small_memory._mem_embeddings) == 0

    def test_embedding_count_tracked(self, small_memory):
        emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        small_memory.store_embedding(emb, "10-K", "sec_filing")
        stats = small_memory.get_stats()
        assert stats["embedding_count"] == 1
        assert stats["vector_backend"] == "in-memory"

    def test_in_memory_cosine_exact(self, small_memory):
        """Verify in-memory cosine similarity computes correctly."""
        emb = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        small_memory.store_embedding(emb, "report", "financial_report")
        # Identical vector should give similarity = 1.0
        result = small_memory.lookup_by_embedding(emb)
        assert result is not None
        assert result[2] >= 0.999


# ── SGD Classifier ────────────────────────────────────────────────────


class TestSGDClassifier:
    def test_sgd_not_ready_with_few_samples(self, small_memory):
        emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        small_memory.store_embedding(emb, "10-K", "sec_filing")
        result = small_memory.predict_sgd(emb)
        assert result is None

    def test_sgd_predicts_after_enough_samples(self, small_memory):
        rng = np.random.RandomState(42)
        for _ in range(SGD_MIN_SAMPLES):
            emb = rng.randn(4).astype(np.float32)
            emb[0] = abs(emb[0]) + 1.0
            small_memory.store_embedding(emb, "10-K", "sec_filing")
        for _ in range(SGD_MIN_SAMPLES):
            emb = rng.randn(4).astype(np.float32)
            emb[1] = abs(emb[1]) + 1.0
            small_memory.store_embedding(emb, "contract", "legal_document")

        query = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)
        result = small_memory.predict_sgd(query)
        assert result is not None
        assert result[2] > 0

    def test_sgd_trains_incrementally(self, small_memory):
        assert small_memory._sgd_sample_count == 0
        emb1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        small_memory.store_embedding(emb1, "10-K", "sec_filing")
        assert small_memory._sgd_sample_count == 1
        emb2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        small_memory.store_embedding(emb2, "contract", "legal_document")
        assert small_memory._sgd_sample_count == 2
        assert small_memory._sgd_fitted is True


# ── Pattern memory (keyword/filename) ─────────────────────────────────


class TestMemoryClassification:
    def test_memory_stores_pattern(self, memory):
        pid = memory.store_pattern("10-K", "sec_filing", "Company_10K.pdf", ["Annual Report"])
        assert pid is not None
        assert len(memory.get_all_patterns()) == 1

    def test_memory_lookup_requires_min_observations(self, memory):
        memory.store_pattern("10-K", "sec_filing", "test.pdf")
        assert memory.lookup(filename="test.pdf") is None

    def test_memory_lookup_returns_trusted_pattern(self, memory):
        pid = memory.store_pattern("10-K", "sec_filing", "test.pdf")
        for _ in range(MEMORY_MIN_OBSERVATIONS):
            memory.store_pattern("10-K", "sec_filing", "test.pdf")
            memory.record_feedback(pid, correct=True)
        result = memory.lookup(filename="test.pdf")
        assert result is not None
        assert result.document_type == "10-K"

    def test_memory_feedback_updates_accuracy(self, memory):
        pid = memory.store_pattern("10-K", "sec_filing", "test.pdf")
        memory.record_feedback(pid, correct=True)
        memory.record_feedback(pid, correct=True)
        memory.record_feedback(pid, correct=False)
        assert memory.get_all_patterns()[0].success_count == 2

    def test_memory_deduplicates(self, memory):
        pid1 = memory.store_pattern("10-K", "sec_filing", "test.pdf")
        pid2 = memory.store_pattern("10-K", "sec_filing", "test.pdf")
        assert pid1 == pid2
        assert memory.get_all_patterns()[0].total_count == 2


# ── Memory persistence ────────────────────────────────────────────────


class TestMemoryPersistence:
    def test_export_import_patterns(self, memory):
        memory.store_pattern("10-K", "sec_filing", "filing.pdf", ["Annual Report"])
        memory.store_pattern("contract", "legal_document", "agreement.pdf")
        exported = memory.export_json()
        new_memory = ClassificationMemory(use_db=False)
        count = new_memory.import_json(exported)
        assert count == 2
        assert len(new_memory.get_all_patterns()) == 2

    def test_export_import_embeddings(self, small_memory):
        emb = np.array([1.0, 0.0, 0.5, 0.0], dtype=np.float32)
        small_memory.store_embedding(emb, "10-K", "sec_filing")
        exported = small_memory.export_json()
        new_memory = ClassificationMemory(embedding_dim=4, use_db=False)
        count = new_memory.import_json(exported)
        assert count >= 1
        assert len(new_memory._mem_embeddings) == 1
        result = new_memory.lookup_by_embedding(emb)
        assert result is not None
        assert result[0] == "10-K"

    def test_memory_stats_include_ml(self, small_memory):
        emb1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        small_memory.store_embedding(emb1, "10-K", "sec_filing")
        small_memory.store_embedding(emb2, "contract", "legal_document")
        small_memory.store_pattern("10-K", "sec_filing", "a.pdf")
        stats = small_memory.get_stats()
        assert stats["total_patterns"] == 1
        assert stats["embedding_count"] == 2
        assert stats["vector_backend"] == "in-memory"
        assert stats["sgd_sample_count"] == 2
        assert stats["sgd_fitted"] is True


# ── LLM response parsing ─────────────────────────────────────────────


class TestLLMResponseParsing:
    def test_parse_clean_json(self, classifier):
        result = classifier._parse_llm_response(
            '{"document_type": "10-K", "classification_label": "sec_filing", "confidence": 0.92}'
        )
        assert result == ("10-K", "sec_filing", 0.92)

    def test_parse_json_in_code_block(self, classifier):
        result = classifier._parse_llm_response(
            '```json\n{"document_type": "contract", "classification_label": "legal_document", "confidence": 0.85}\n```'
        )
        assert result == ("contract", "legal_document", 0.85)

    def test_parse_caps_confidence(self, classifier):
        result = classifier._parse_llm_response(
            '{"document_type": "10-K", "classification_label": "sec_filing", "confidence": 0.99}'
        )
        assert result[2] == 0.95

    def test_parse_invalid_json(self, classifier):
        assert classifier._parse_llm_response("This is not JSON") is None


# ── Fallback ──────────────────────────────────────────────────────────


class TestFallback:
    def test_unknown_document(self, classifier):
        result = classifier.classify(
            doc_id="doc1", filename="random_file.pdf",
            front_matter_text="Some random text that matches nothing",
        )
        assert result.document_type == "unknown"
        assert result.classification_label == "unclassified"
        assert result.classification_method == "default"

    def test_low_confidence_filename_fallback(self, classifier):
        result = classifier.classify(doc_id="doc1", filename="policy_update.pdf")
        assert result.document_type == "policy_document"
        assert result.classification_label == "governance"
        assert result.classification_method == "deterministic"


# ── Message bus integration ───────────────────────────────────────────


class TestMessageBusIntegration:
    def test_handle_message(self, classifier):
        from agents.message_bus import create_message
        msg = create_message(
            from_agent="orchestrator", to_agent="classifier",
            message_type="classification_request",
            payload={"doc_id": "doc1", "filename": "Annual_Report_2024.pdf",
                     "front_matter_text": "", "page_count": 50},
            query_id=new_id(),
        )
        result = classifier.handle_message(msg)
        assert isinstance(result, ClassificationResult)
        assert result.document_type == "annual_report"


# ── Self-learning integration ─────────────────────────────────────────


class TestSelfLearning:
    def test_classification_stores_in_memory(self):
        memory = ClassificationMemory(use_db=False)
        bus = MessageBus()
        classifier = ClassifierAgent(bus=bus, gateway=None, memory=memory)
        classifier.classify(doc_id="doc1", filename="Company_10-K.pdf")
        assert any(p.document_type == "10-K" for p in memory.get_all_patterns())

    def test_repeated_classifications_increase_count(self):
        memory = ClassificationMemory(use_db=False)
        bus = MessageBus()
        classifier = ClassifierAgent(bus=bus, gateway=None, memory=memory)
        classifier.classify(doc_id="doc1", filename="Company_10-K.pdf")
        classifier.classify(doc_id="doc2", filename="Company_10-K.pdf")
        ten_k = [p for p in memory.get_all_patterns() if p.document_type == "10-K"]
        assert ten_k[0].total_count >= 2

    def test_feedback_improves_accuracy(self):
        memory = ClassificationMemory(use_db=False)
        bus = MessageBus()
        classifier = ClassifierAgent(bus=bus, gateway=None, memory=memory)
        classifier.classify(doc_id="doc1", filename="Company_10-K.pdf")
        pid = memory.get_all_patterns()[0].pattern_id
        classifier.handle_feedback(pid, correct=True)
        classifier.handle_feedback(pid, correct=True)
        assert memory.get_all_patterns()[0].success_count == 2


# ── Embedding-based classification via ClassifierAgent ────────────────


class TestEmbeddingClassification:
    def _make_classifier_with_mock_embedder(self, memory, embed_fn):
        class MockEmbedder:
            def embed_text(self, text):
                return embed_fn(text)
        bus = MessageBus()
        return ClassifierAgent(bus=bus, gateway=None, memory=memory, embedder=MockEmbedder())

    def test_embedding_stored_on_classify(self):
        memory = ClassificationMemory(embedding_dim=4, use_db=False)
        classifier = self._make_classifier_with_mock_embedder(
            memory, lambda text: [1.0, 0.0, 0.0, 0.0]
        )
        classifier.classify(
            doc_id="doc1", filename="Form_10-K.pdf",
            front_matter_text="ANNUAL REPORT PURSUANT TO SECTION 13",
        )
        assert len(memory._mem_embeddings) == 1
        assert memory._sgd_sample_count == 1

    def test_embedding_similarity_used_for_matching(self):
        memory = ClassificationMemory(embedding_dim=4, use_db=False)
        call_count = [0]

        def embed_fn(text):
            call_count[0] += 1
            if call_count[0] <= 1:
                return [1.0, 0.0, 0.0, 0.0]
            return [0.99, 0.01, 0.0, 0.0]

        classifier = self._make_classifier_with_mock_embedder(memory, embed_fn)

        # First: deterministic match stores embedding
        classifier.classify(
            doc_id="doc1", filename="Form_10-K.pdf",
            front_matter_text="ANNUAL REPORT PURSUANT TO SECTION 13",
        )

        # Second: unknown filename, should match via pgvector similarity (in-memory fallback)
        result = classifier.classify(
            doc_id="doc2", filename="unknown_doc.pdf",
            front_matter_text="Some financial report text",
        )
        assert result.classification_method == "embedding_similarity"
        assert result.document_type == "10-K"
        assert result.confidence > EMBEDDING_SIMILARITY_THRESHOLD


# ── Neo4j knowledge graph integration ─────────────────────────────────


class TestNeo4jIntegration:
    @patch("agents.classifier_agent._neo4j_available", return_value=True)
    @patch("agents.classifier_agent._neo4j_store_classification")
    def test_classification_stored_in_neo4j(self, mock_store, mock_avail):
        """Classifications should be stored in Neo4j when enabled."""
        memory = ClassificationMemory(use_db=False)
        bus = MessageBus()
        classifier = ClassifierAgent(bus=bus, gateway=None, memory=memory)
        classifier.classify(doc_id="doc1", filename="Company_10-K.pdf", page_count=50)
        mock_store.assert_called_once()
        call_kwargs = mock_store.call_args
        assert call_kwargs[1]["doc_id"] == "doc1" or call_kwargs[0][0] == "doc1"

    @patch("agents.classifier_agent._neo4j_available", return_value=False)
    @patch("agents.classifier_agent._neo4j_store_classification")
    def test_neo4j_skipped_when_disabled(self, mock_store, mock_avail):
        """Neo4j should not be called when disabled."""
        memory = ClassificationMemory(use_db=False)
        bus = MessageBus()
        classifier = ClassifierAgent(bus=bus, gateway=None, memory=memory)
        classifier.classify(doc_id="doc1", filename="Company_10-K.pdf")
        mock_store.assert_not_called()


# ── pgvector backend selection ────────────────────────────────────────


class TestBackendSelection:
    def test_use_db_false_uses_in_memory(self):
        memory = ClassificationMemory(embedding_dim=4, use_db=False)
        assert memory._use_db is False
        stats = memory.get_stats()
        assert stats["vector_backend"] == "in-memory"

    def test_default_no_db_url_uses_in_memory(self):
        """When DATABASE_URL is empty, should default to in-memory."""
        with patch("agents.classifier_agent.settings") as mock_settings:
            mock_settings.database_url = ""
            mock_settings.neo4j.enabled = False
            memory = ClassificationMemory(embedding_dim=4)
            assert memory._use_db is False
