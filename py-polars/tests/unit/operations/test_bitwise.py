import pytest

import polars as pl


@pytest.mark.parametrize("op", ["and_", "or_"])
def test_bitwise_integral_schema(op: str) -> None:
    df = pl.LazyFrame({"a": [1, 2], "b": [3, 4]})
    q = df.select(getattr(pl.col("a"), op)(pl.col("b")))
    assert q.collect_schema()["a"] == df.collect_schema()["a"]
