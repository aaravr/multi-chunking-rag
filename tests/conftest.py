"""Shared pytest configuration and fixtures.

Test stratification markers:
    unit         — No external dependencies (DB, LLM, network). Default for unmarked tests.
    integration  — Requires PostgreSQL via TEST_DATABASE_URL.
    external     — Requires external services (Azure DI, OpenAI API, Kafka).
    slow         — Tests that take >5 seconds to run.
    legacy       — Tests for deprecated modules retained for backward compatibility.

Usage:
    pytest tests/ -v -m unit               # unit tests only (CI default)
    pytest tests/ -v -m integration        # integration tests only
    pytest tests/ -v -m "not external"     # skip external-dependency tests
    pytest tests/ -v -m legacy             # deprecated module tests only
    pytest tests/ -v -m "not legacy"       # skip deprecated module tests
    pytest tests/ -v                       # all tests
"""

import pytest

# Test modules that exercise deprecated code paths.
# These are auto-tagged with the 'legacy' marker so that default CI
# can skip them with ``-m "not legacy"``.
_LEGACY_MODULES = frozenset({
    "test_feedback_and_retraining",
})


def pytest_collection_modifyitems(config, items):
    """Auto-apply stratification markers to tests.

    1. Tests in _LEGACY_MODULES get the 'legacy' marker automatically.
    2. Tests with no stratification marker get 'unit' automatically.

    This ensures every test is categorized without requiring manual annotation
    on every test file. Tests that need DB or external services must explicitly
    add @pytest.mark.integration or @pytest.mark.external.
    """
    stratification_markers = {"unit", "integration", "external", "legacy"}
    unit_marker = pytest.mark.unit
    legacy_marker = pytest.mark.legacy

    for item in items:
        item_markers = {m.name for m in item.iter_markers()}

        # Auto-apply legacy marker to deprecated module tests
        module_name = item.module.__name__.rsplit(".", 1)[-1] if item.module else ""
        if module_name in _LEGACY_MODULES and "legacy" not in item_markers:
            item.add_marker(legacy_marker)
            item_markers.add("legacy")

        # Auto-apply unit marker to any test without a stratification marker
        if not item_markers & stratification_markers:
            item.add_marker(unit_marker)
