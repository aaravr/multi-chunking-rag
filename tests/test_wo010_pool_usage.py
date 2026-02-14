"""WO-010: Assert DB access uses connection pool."""


def test_no_direct_psycopg2_connect_outside_pool():
    """No psycopg2.connect() in app code; only db_pool and setup_db may use it."""
    allowed_paths = {"storage/db_pool.py", "storage/setup_db.py"}
    check_paths = [
        "storage/db.py",
        "storage/repo.py",
        "retrieval/vector_search.py",
        "retrieval/bm25_index.py",
        "retrieval/hybrid.py",
        "retrieval/metadata.py",
        "ingestion/ingest_pipeline.py",
        "app/poc_app.py",
    ]
    violations = []
    for path in check_paths:
        try:
            with open(path, "r") as f:
                content = f.read()
            if "psycopg2.connect(" in content:
                violations.append(path)
        except FileNotFoundError:
            pass
    assert not violations, f"Direct psycopg2.connect found (use get_connection): {violations}"


def test_db_module_reexports_pool():
    """storage.db delegates to db_pool."""
    with open("storage/db.py") as f:
        content = f.read()
    assert "db_pool" in content
