from app import search

NO_MATCH_CASES = [
    # (query, expect_empty)
    ("halloween costume", True),
    ("kitchen appliance", True),
    ("cooking pot", True),
    ("gift for my dad", False),
    ("everyday jewelry", False),
    ("storage for my laptop", False),
]


def test_no_match_floor():
    failures = []
    for query, expect_empty in NO_MATCH_CASES:
        results = search(query)
        got_empty = len(results) == 0
        if got_empty != expect_empty:
            failures.append(f"{query!r}: expected empty={expect_empty}, got empty={got_empty}")

    assert not failures, "\n" + "\n".join(failures)
