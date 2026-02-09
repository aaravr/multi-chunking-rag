from retrieval import router


def test_classify_numeric_list_items_of_note():
    intent = router.classify_query(
        "List the items of note affecting 2024 net income and aggregate impact"
    )
    assert intent.intent == "coverage"
    assert intent.coverage_type == "numeric_list"


def test_classify_pointer_query():
    intent = router.classify_query(
        "Where can I find the discussion of significant events?"
    )
    assert intent.intent == "coverage"
    assert intent.coverage_type == "pointer"


def test_classify_attribute_query():
    intent = router.classify_query(
        "What is the aggregate range of reasonably possible losses?"
    )
    assert intent.intent == "coverage"
    assert intent.coverage_type == "attribute"


def test_classify_closed_matters_query():
    intent = router.classify_query(
        "Which matters are explicitly closed and what closed them"
    )
    assert intent.intent == "coverage"
    assert intent.coverage_type == "list"
    assert intent.status_filter == "closed"
