import numpy as np

import polars as pl
from polars.testing import assert_frame_equal


def test_arr_min_max() -> None:
    s = pl.Series("a", [[1, 2], [4, 3]], dtype=pl.Array(pl.Int64, 2))
    assert s.arr.max().to_list() == [2, 4]
    assert s.arr.min().to_list() == [1, 3]


def test_array_min_max_dtype_12123() -> None:
    df = pl.LazyFrame(
        [pl.Series("a", [[1.0, 3.0], [2.0, 5.0]]), pl.Series("b", [1.0, 2.0])],
        schema_overrides={
            "a": pl.Array(pl.Float64, 2),
        },
    )

    df = df.with_columns(
        max=pl.col("a").arr.max().alias("max"),
        min=pl.col("a").arr.min().alias("min"),
    )

    assert df.schema == {
        "a": pl.Array(pl.Float64, 2),
        "b": pl.Float64,
        "max": pl.Float64,
        "min": pl.Float64,
    }

    out = df.select(pl.col("max") * pl.col("b"), pl.col("min") * pl.col("b")).collect()

    assert_frame_equal(out, pl.DataFrame({"max": [3.0, 10.0], "min": [1.0, 4.0]}))


def test_arr_sum() -> None:
    s = pl.Series("a", [[1, 2], [4, 3]], dtype=pl.Array(pl.Int64, 2))
    assert s.arr.sum().to_list() == [3, 7]


def test_arr_unique() -> None:
    df = pl.DataFrame(
        {"a": pl.Series("a", [[1, 1], [4, 3]], dtype=pl.Array(pl.Int64, 2))}
    )

    out = df.select(pl.col("a").arr.unique(maintain_order=True))
    expected = pl.DataFrame({"a": [[1], [4, 3]]})
    assert_frame_equal(out, expected)


def test_array_to_numpy() -> None:
    s = pl.Series([[1, 2], [3, 4], [5, 6]], dtype=pl.Array(pl.Int64, 2))
    assert (s.to_numpy() == np.array([[1, 2], [3, 4], [5, 6]])).all()
