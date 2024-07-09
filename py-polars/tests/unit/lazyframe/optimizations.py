import polars as pl


def test_remove_double_sort() -> None:
    assert (
        pl.LazyFrame({"a": [1, 2, 3, 3]}).sort("a").sort("a").explain().count("SORT")
        == 1
    )
