import pytest

import polars as pl


@pytest.mark.may_fail_auto_streaming
def test_invalid_broadcast() -> None:
    df = pl.DataFrame(
        {
            "a": [100, 103],
            "group": [0, 1],
        }
    )
    with pytest.raises(pl.exceptions.ShapeError):
        df.select(pl.col("group").filter(pl.col("group") == 0), "a")


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Null,
        pl.Int32,
        pl.String,
        pl.Enum(["foo"]),
        pl.Binary,
        pl.List(pl.Int32),
        pl.Struct({"a": pl.Int32}),
        pl.Array(pl.Int32, 1),
        pl.List(pl.List(pl.Int32)),
    ],
)
def test_null_literals(dtype: pl.DataType) -> None:
    assert (
        pl.DataFrame([pl.Series("a", [1, 2], pl.Int64)])
        .with_columns(pl.lit(None).cast(dtype).alias("b"))
        .collect_schema()
        .dtypes()
    ) == [pl.Int64, dtype]
