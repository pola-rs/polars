import datetime

import numpy as np
import pytest

import polars as pl
from polars import ComputeError
from polars.testing import assert_frame_equal, assert_series_equal


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


def test_array_any_all() -> None:
    s = pl.Series(
        [[True, True], [False, True], [False, False], [None, None], None],
        dtype=pl.Array(pl.Boolean, 2),
    )

    expected_any = pl.Series([True, True, False, False, None])
    assert_series_equal(s.arr.any(), expected_any)

    expected_all = pl.Series([True, False, False, True, None])
    assert_series_equal(s.arr.all(), expected_all)

    s = pl.Series([[1, 2], [3, 4], [5, 6]], dtype=pl.Array(pl.Int64, 2))
    with pytest.raises(ComputeError, match="expected boolean elements in array"):
        s.arr.any()
    with pytest.raises(ComputeError, match="expected boolean elements in array"):
        s.arr.all()


def test_array_sort() -> None:
    s = pl.Series([[2, None, 1], [1, 3, 2]], dtype=pl.Array(pl.UInt32, 3))

    desc = s.arr.sort(descending=True)
    expected = pl.Series([[None, 2, 1], [3, 2, 1]], dtype=pl.Array(pl.UInt32, 3))
    assert_series_equal(desc, expected)

    asc = s.arr.sort(descending=False)
    expected = pl.Series([[None, 1, 2], [1, 2, 3]], dtype=pl.Array(pl.UInt32, 3))
    assert_series_equal(asc, expected)


def test_array_reverse() -> None:
    s = pl.Series([[2, None, 1], [1, None, 2]], dtype=pl.Array(pl.UInt32, 3))

    s = s.arr.reverse()
    expected = pl.Series([[1, None, 2], [2, None, 1]], dtype=pl.Array(pl.UInt32, 3))
    assert_series_equal(s, expected)


def test_array_arg_min_max() -> None:
    s = pl.Series("a", [[1, 2, 4], [3, 2, 1]], dtype=pl.Array(pl.UInt32, 3))
    expected = pl.Series("a", [0, 2], dtype=pl.UInt32)
    assert_series_equal(s.arr.arg_min(), expected)
    expected = pl.Series("a", [2, 0], dtype=pl.UInt32)
    assert_series_equal(s.arr.arg_max(), expected)


def test_array_get() -> None:
    # test index literal
    s = pl.Series(
        "a",
        [[1, 2, 3, 4], [5, 6, None, None], [7, 8, 9, 10]],
        dtype=pl.Array(pl.Int64, 4),
    )
    out = s.arr.get(1)
    expected = pl.Series("a", [2, 6, 8], dtype=pl.Int64)
    assert_series_equal(out, expected)

    # test index expr
    out = s.arr.get(pl.Series([1, -2, 4]))
    expected = pl.Series("a", [2, None, None], dtype=pl.Int64)
    assert_series_equal(out, expected)

    # test logical type
    s = pl.Series(
        "a",
        [
            [datetime.date(1999, 1, 1), datetime.date(2000, 1, 1)],
            [datetime.date(2001, 10, 1), None],
            [None, None],
        ],
        dtype=pl.Array(pl.Date, 2),
    )
    out = s.arr.get(pl.Series([1, -2, 4]))
    expected = pl.Series(
        "a",
        [datetime.date(2000, 1, 1), datetime.date(2001, 10, 1), None],
        dtype=pl.Date,
    )
    assert_series_equal(out, expected)
