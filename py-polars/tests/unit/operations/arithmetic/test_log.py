import pytest

import polars as pl


@pytest.mark.parametrize(
    ("self_dtype", "base_dtype", "expected_dtype"),
    [
        (pl.Float32, pl.Float32, pl.Float32),
        (pl.Float32, pl.Float64, pl.Float32),
        (pl.Float64, pl.Float32, pl.Float64),
        (pl.Float64, pl.Float64, pl.Float64),
        (pl.Float32, pl.Int32, pl.Float32),
        (pl.Float64, pl.Int32, pl.Float64),
        (pl.Int32, pl.Float32, pl.Float32),
        (pl.Int32, pl.Float64, pl.Float64),
    ],
)
def test_log_dtype_24517(
    self_dtype: pl.DataType, base_dtype: pl.DataType, expected_dtype: pl.DataType
) -> None:
    df = pl.DataFrame(
        {
            "a": [2],
            "b": [10],
        },
        schema={"a": self_dtype, "b": base_dtype},
    )
    q = df.lazy().select(pl.col("a").log(pl.col("b")).alias("c"))

    assert q.collect().schema.dtypes()[0] == expected_dtype
    assert q.collect_schema().dtypes()[0] == expected_dtype
