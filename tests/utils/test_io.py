from recap import URI


def test_virtural_uri():
    from chesscog.utils.io import _DATA_DIR
    a = URI("data://a/b")
    assert str(a) == str(_DATA_DIR / "a" / "b")
