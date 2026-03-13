"""Tests for the Classifier Agent (MASTER_PROMPT §4.8).

Validates:
- Deterministic classification from filename patterns
- Deterministic classification from content signals
- Memory-based classification (self-learning)
- Feedback reinforcement loop
- LLM response parsing
- Memory export/import
- Fallback to unknown for unrecognizable documents
"""

import json

import pytest

from agents.classifier_agent import (
    ClassificationMemory,
    ClassifierAgent,
    MEMORY_MIN_OBSERVATIONS,
    MEMORY_TRUST_THRESHOLD,
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
    return ClassificationMemory()


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
            filename="report.pdf",  # generic filename
            front_matter_text="ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d)",
        )
        assert result.document_type == "10-K"
        assert result.confidence >= 0.90


# ── Memory-based classification ───────────────────────────────────────


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
        """Pattern must have enough observations to be trusted."""
        memory.store_pattern(
            document_type="10-K",
            classification_label="sec_filing",
            filename="test.pdf",
        )
        # Only 1 observation, below threshold
        result = memory.lookup(filename="test.pdf")
        assert result is None

    def test_memory_lookup_returns_trusted_pattern(self, memory):
        """Trusted patterns (high accuracy + enough observations) are returned."""
        pid = memory.store_pattern(
            document_type="10-K",
            classification_label="sec_filing",
            filename="test.pdf",
        )
        # Simulate enough observations with positive feedback
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
    def test_export_import_roundtrip(self, memory):
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
        assert len(data) == 2

        # Import into new memory
        new_memory = ClassificationMemory()
        count = new_memory.import_json(exported)
        assert count == 2
        assert len(new_memory.get_all_patterns()) == 2

    def test_memory_stats(self, memory):
        memory.store_pattern("10-K", "sec_filing", "a.pdf")
        memory.store_pattern("contract", "legal_document", "b.pdf")
        stats = memory.get_stats()
        assert stats["total_patterns"] == 2
        assert stats["total_observations"] == 2


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
        """Confidence should be capped at 0.95 for LLM classifications."""
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
        """Filename match below 0.85 threshold, no content or memory match,
        should still return the filename match as fallback."""
        result = classifier.classify(
            doc_id="doc1",
            filename="policy_update.pdf",
        )
        # policy pattern matches filename at 0.80 confidence (below 0.85)
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
        """Each classification should be stored in memory for future learning."""
        memory = ClassificationMemory()
        bus = MessageBus()
        classifier = ClassifierAgent(bus=bus, gateway=None, memory=memory)

        classifier.classify(doc_id="doc1", filename="Company_10-K.pdf")

        patterns = memory.get_all_patterns()
        assert len(patterns) >= 1
        assert any(p.document_type == "10-K" for p in patterns)

    def test_repeated_classifications_increase_count(self):
        """Classifying similar documents should reinforce memory patterns."""
        memory = ClassificationMemory()
        bus = MessageBus()
        classifier = ClassifierAgent(bus=bus, gateway=None, memory=memory)

        classifier.classify(doc_id="doc1", filename="Company_10-K.pdf")
        classifier.classify(doc_id="doc2", filename="Company_10-K.pdf")

        patterns = memory.get_all_patterns()
        # Should have merged into one pattern with count >= 2
        ten_k_patterns = [p for p in patterns if p.document_type == "10-K"]
        assert len(ten_k_patterns) >= 1
        assert ten_k_patterns[0].total_count >= 2

    def test_feedback_improves_accuracy(self):
        """Positive feedback should increase pattern accuracy."""
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
