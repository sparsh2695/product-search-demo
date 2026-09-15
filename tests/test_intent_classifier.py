from app import MODEL, _excluded_category

GENDER_INTENT_CASES = [
    # (query, expected excluded category)
    ("something warm for winter", None),
    ("gear for outdoor hiking", None),
    ("everyday jewelry", None),
    ("gift for my dad", "women's clothing"),
    ("something for my grandpa", "women's clothing"),
    ("shirt for the guy in my life", "women's clothing"),
    ("his birthday present", "women's clothing"),
    ("something for my grandma", "men's clothing"),
    ("dress for the girl in my life", "men's clothing"),
]


def test_gender_intent_classification():
    failures = []
    for query, expected in GENDER_INTENT_CASES:
        embedding = MODEL.encode([query], normalize_embeddings=True)
        got = _excluded_category(embedding)
        if got != expected:
            failures.append(f"{query!r}: expected {expected!r}, got {got!r}")

    assert not failures, "\n" + "\n".join(failures)
