from __future__ import annotations

import datetime
from typing import Any

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

    # test nulls_last
    s = pl.Series([[None, 1, 2], [-1, None, 9]], dtype=pl.Array(pl.Int8, 3))
    assert_series_equal(
        s.arr.sort(nulls_last=True),
        pl.Series([[1, 2, None], [-1, 9, None]], dtype=pl.Array(pl.Int8, 3)),
    )
    assert_series_equal(
        s.arr.sort(nulls_last=False),
        pl.Series([[None, 1, 2], [None, -1, 9]], dtype=pl.Array(pl.Int8, 3)),
    )


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


def test_arr_first_last() -> None:
    s = pl.Series(
        "a",
        [[1, 2, 3], [None, 5, 6], [None, None, None]],
        dtype=pl.Array(pl.Int64, 3),
    )

    first = s.arr.first()
    expected_first = pl.Series(
        "a",
        [1, None, None],
        dtype=pl.Int64,
    )
    assert_series_equal(first, expected_first)

    last = s.arr.last()
    expected_last = pl.Series(
        "a",
        [3, 6, None],
        dtype=pl.Int64,
    )
    assert_series_equal(last, expected_last)


@pytest.mark.parametrize(
    ("data", "set", "dtype"),
    [
        ([1, 2], [[1, 2], [3, 4]], pl.Int64),
        ([True, False], [[True, False], [True, True]], pl.Boolean),
        (["a", "b"], [["a", "b"], ["c", "d"]], pl.String),
        ([b"a", b"b"], [[b"a", b"b"], [b"c", b"d"]], pl.Binary),
        (
            [{"a": 1}, {"a": 2}],
            [[{"a": 1}, {"a": 2}], [{"b": 1}, {"a": 3}]],
            pl.Struct([pl.Field("a", pl.Int64)]),
        ),
    ],
)
def test_is_in_array(data: list[Any], set: list[list[Any]], dtype: pl.DataType) -> None:
    df = pl.DataFrame(
        {"a": data, "arr": set},
        schema={"a": dtype, "arr": pl.Array(dtype, 2)},
    )
    out = df.select(is_in=pl.col("a").is_in(pl.col("arr"))).to_series()
    expected = pl.Series("is_in", [True, False])
    assert_series_equal(out, expected)


def test_array_join() -> None:
    df = pl.DataFrame(
        {
            "a": [["ab", "c", "d"], ["e", "f", "g"], [None, None, None], None],
            "separator": ["&", None, "*", "_"],
        },
        schema={
            "a": pl.Array(pl.String, 3),
            "separator": pl.String,
        },
    )
    out = df.select(pl.col("a").arr.join("-"))
    assert out.to_dict(as_series=False) == {
        "a": ["ab-c-d", "e-f-g", "null-null-null", None]
    }
    out = df.select(pl.col("a").arr.join(pl.col("separator")))
    assert out.to_dict(as_series=False) == {
        "a": ["ab&c&d", None, "null*null*null", None]
    }


@pytest.mark.parametrize(
    ("array", "data", "expected", "dtype"),
    [
        ([[1, 2], [3, 4]], [1, 5], [True, False], pl.Int64),
        ([[True, False], [True, True]], [True, False], [True, False], pl.Boolean),
        ([["a", "b"], ["c", "d"]], ["a", "b"], [True, False], pl.String),
        ([[b"a", b"b"], [b"c", b"d"]], [b"a", b"b"], [True, False], pl.Binary),
        (
            [[{"a": 1}, {"a": 2}], [{"b": 1}, {"a": 3}]],
            [{"a": 1}, {"a": 2}],
            [True, False],
            pl.Struct([pl.Field("a", pl.Int64)]),
        ),
    ],
)
def test_array_contains_expr(
    array: list[list[Any]], data: list[Any], expected: list[bool], dtype: pl.DataType
) -> None:
    df = pl.DataFrame(
        {
            "array": array,
            "data": data,
        },
        schema={
            "array": pl.Array(dtype, 2),
            "data": dtype,
        },
    )
    out = df.select(contains=pl.col("array").arr.contains(pl.col("data"))).to_series()
    expected_series = pl.Series("contains", expected)
    assert_series_equal(out, expected_series)


@pytest.mark.parametrize(
    ("array", "data", "expected", "dtype"),
    [
        ([[1, 2], [3, 4]], 1, [True, False], pl.Int64),
        ([[True, False], [True, True]], True, [True, True], pl.Boolean),
        ([["a", "b"], ["c", "d"]], "a", [True, False], pl.String),
        ([[b"a", b"b"], [b"c", b"d"]], b"a", [True, False], pl.Binary),
    ],
)
def test_array_contains_literal(
    array: list[list[Any]], data: Any, expected: list[bool], dtype: pl.DataType
) -> None:
    df = pl.DataFrame(
        {
            "array": array,
        },
        schema={
            "array": pl.Array(dtype, 2),
        },
    )
    out = df.select(contains=pl.col("array").arr.contains(data)).to_series()
    expected_series = pl.Series("contains", expected)
    assert_series_equal(out, expected_series)


@pytest.mark.parametrize(
    ("arr", "data", "expected", "dtype"),
    [
        ([[1, 2], [3, None], None], 1, [1, 0, None], pl.Int64),
        ([[True, False], [True, None], None], True, [1, 1, None], pl.Boolean),
        ([["a", "b"], ["c", None], None], "a", [1, 0, None], pl.String),
        ([[b"a", b"b"], [b"c", None], None], b"a", [1, 0, None], pl.Binary),
    ],
)
def test_array_count_matches(
    arr: list[list[Any] | None], data: Any, expected: list[Any], dtype: pl.DataType
) -> None:
    df = pl.DataFrame({"arr": arr}, schema={"arr": pl.Array(dtype, 2)})
    out = df.select(count_matches=pl.col("arr").arr.count_matches(data))
    assert out.to_dict(as_series=False) == {"count_matches": expected}
