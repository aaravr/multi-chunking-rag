"""WO-010: Assert model loads once per process."""

import pytest

from embedding import model_registry


def test_model_singleton_same_instance():
    """get_embedding_model() returns the same instance on repeated calls."""
    model_registry._reset_for_testing()
    m1 = model_registry.get_embedding_model()
    m2 = model_registry.get_embedding_model()
    assert m1 is m2


def test_model_singleton_reset_allows_new_instance():
    """After reset, a new instance is created."""
    model_registry._reset_for_testing()
    m1 = model_registry.get_embedding_model()
    model_registry._reset_for_testing()
    m2 = model_registry.get_embedding_model()
    assert m1 is not m2
