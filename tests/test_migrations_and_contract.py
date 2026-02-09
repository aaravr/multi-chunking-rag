import os

import pytest

from core import config
from storage.schema_contract import check_schema_contract
from storage.setup_db import run_setup


@pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"),
    reason="TEST_DATABASE_URL not set; skipping migration test.",
)
def test_migrations_and_contract():
    test_db = os.environ["TEST_DATABASE_URL"]
    config.settings.database_url = test_db
    run_setup()
    check_schema_contract()
