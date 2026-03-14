"""Shared pytest configuration and fixtures.

Test stratification markers:
    unit         — No external dependencies (DB, LLM, network). Default for unmarked tests.
    integration  — Requires PostgreSQL via TEST_DATABASE_URL.
    external     — Requires external services (Azure DI, OpenAI API, Kafka).
    slow         — Tests that take >5 seconds to run.

Usage:
    pytest tests/ -v -m unit               # unit tests only
    pytest tests/ -v -m integration        # integration tests only
    pytest tests/ -v -m "not external"     # skip external-dependency tests
"""

import pytest


def pytest_collection_modifyitems(config, items):
    """Auto-apply 'unit' marker to tests that have no stratification marker.

    This ensures every test is categorized without requiring manual annotation
    on every test file. Tests that need DB or external services must explicitly
    add @pytest.mark.integration or @pytest.mark.external.
    """
    stratification_markers = {"unit", "integration", "external"}
    unit_marker = pytest.mark.unit

    for item in items:
        item_markers = {m.name for m in item.iter_markers()}
        if not item_markers & stratification_markers:
            item.add_marker(unit_marker)
