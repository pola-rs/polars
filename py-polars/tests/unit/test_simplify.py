import polars as pl


def test_flatten_alias() -> None:
    assert (
        """len().alias("bar")"""
        in pl.LazyFrame({"a": [1, 2]})
        .select(pl.len().alias("foo").alias("bar"))
        .explain()
    )
