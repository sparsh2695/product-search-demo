from app import MODEL, _is_browse_all

BROWSE_INTENT_CASES = [
    # (query, expect_browse_all)
    ("all products", True),
    ("show me everything", True),
    ("browse catalog", True),
    ("what do you have", True),
    ("gift for my dad", False),
    ("everyday jewelry", False),
    ("what winter clothes do you have", False),  # category-specific, not "everything"
    ("show me your jewelry", False),
    ("show me all your winter coats", False),
]


def test_browse_intent_classification():
    failures = []
    for query, expected in BROWSE_INTENT_CASES:
        embedding = MODEL.encode([query], normalize_embeddings=True)
        got = _is_browse_all(embedding)
        if got != expected:
            failures.append(f"{query!r}: expected {expected!r}, got {got!r}")

    assert not failures, "\n" + "\n".join(failures)
