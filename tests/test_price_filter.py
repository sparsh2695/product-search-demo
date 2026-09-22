from app import search


def test_price_filtered_search():
    results = search("jewelry", max_price=20.0)
    assert results, "expected at least one result"
    assert all(r["price"] < 20.0 for r in results)
    assert all(r["category"] == "jewelery" for r in results)


def test_price_range_with_no_matches():
    # electronics start at $64 -- nothing in the catalog is electronics under $50
    assert search("electronics", max_price=50.0) == []


def test_price_range_empties_candidate_pool():
    # cheapest jewelry is $9.99 -- min_price=$1000 leaves zero eligible products
    assert search("jewelry", min_price=1000.0) == []
