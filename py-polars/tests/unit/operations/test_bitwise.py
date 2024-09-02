import pytest

import polars as pl


@pytest.mark.parametrize("op", ["and_", "or_"])
def test_bitwise_integral_schema(op: str) -> None:
    df = pl.LazyFrame({"a": [1, 2], "b": [3, 4]})
    q = df.select(getattr(pl.col("a"), op)(pl.col("b")))
    assert q.collect_schema()["a"] == df.collect_schema()["a"]


@pytest.mark.parametrize("op", ["and_", "or_", "xor"])
def test_bitwise_single_null_value_schema(op: str) -> None:
    df = pl.DataFrame({"a": [True, True]})
    q = df.select(getattr(pl.col("a"), op)(None))
    result_schema = q.collect_schema()
    assert result_schema.len() == 1
    assert "a" in result_schema
