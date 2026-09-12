from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


def test_is_first_distinct() -> None:
    lf = pl.LazyFrame({"a": [4, 1, 4]})
    result = lf.select(pl.col("a").is_first_distinct()).collect()["a"]
    expected = pl.Series("a", [True, True, False])
    assert_series_equal(result, expected)


def test_is_first_distinct_bool_bit_chunk_index_calc() -> None:
    # The fast path activates on sizes >=64 and processes in chunks of 64-bits.
    # It calculates the indexes using the bit counts, which needs to be from the
    # correct side.
    assert pl.arange(0, 64, eager=True).filter(
        pl.Series([True] + 63 * [False]).is_first_distinct()
    ).to_list() == [0, 1]

    assert pl.arange(0, 64, eager=True).filter(
        pl.Series([False] + 63 * [True]).is_first_distinct()
    ).to_list() == [0, 1]

    assert pl.arange(0, 64, eager=True).filter(
        pl.Series(2 * [True] + 2 * [False] + 60 * [None]).is_first_distinct()
    ).to_list() == [0, 2, 4]

    assert pl.arange(0, 64, eager=True).filter(
        pl.Series(2 * [False] + 2 * [None] + 60 * [True]).is_first_distinct()
    ).to_list() == [0, 2, 4]


def test_is_first_distinct_struct() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3, 2, None, 2, 1], "b": [0, 2, 3, 2, None, 2, 0]})
    result = lf.select(pl.struct("a", "b").is_first_distinct())
    expected = pl.LazyFrame({"a": [True, True, True, False, True, False, False]})
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        [[1, 2], [3], [1, 2], [4, None], [4, None], [], []],
        [[True, None], [True], [True, None], [False], [False], [], []],
        [[b"1", b"2"], [b"3"], [b"1", b"2"], [b"4", None], [b"4", None], [], []],
        [["a", "b"], ["&"], ["a", "b"], ["...", None], ["...", None], [], []],
        [
            [datetime.date(2000, 10, 1), datetime.date(2001, 1, 30)],
            [datetime.date(1949, 10, 1)],
            [datetime.date(2000, 10, 1), datetime.date(2001, 1, 30)],
            [datetime.date(1998, 7, 1), None],
            [datetime.date(1998, 7, 1), None],
            [],
            [],
        ],
    ],
)
def test_is_first_last_distinct_list(data: list[list[Any] | None]) -> None:
    lf = pl.LazyFrame({"a": data})
    result = lf.select(
        first=pl.col("a").is_first_distinct(), last=pl.col("a").is_last_distinct()
    )
    expected = pl.LazyFrame(
        {
            "first": [True, True, False, True, False, True, False],
            "last": [False, True, True, False, True, False, True],
        }
    )
    assert_frame_equal(result, expected)


def test_is_first_last_distinct_list_inner_nested() -> None:
    df = pl.DataFrame({"a": [[[1, 2]], [[1, 2]]]})
    err_msg = "only allowed if the inner type is not nested"
    with pytest.raises(InvalidOperationError, match=err_msg):
        df.select(pl.col("a").is_first_distinct())
    with pytest.raises(InvalidOperationError, match=err_msg):
        df.select(pl.col("a").is_last_distinct())


def test_is_first_distinct_various() -> None:
    # numeric
    s = pl.Series([1, 1, None, 2, None, 3, 3])
    expected = [True, False, True, True, False, True, False]
    assert s.is_first_distinct().to_list() == expected
    # str
    s = pl.Series(["x", "x", None, "y", None, "z", "z"])
    expected = [True, False, True, True, False, True, False]
    assert s.is_first_distinct().to_list() == expected
    # boolean
    s = pl.Series([True, True, None, False, None, False, False])
    expected = [True, False, True, True, False, False, False]
    assert s.is_first_distinct().to_list() == expected
    # struct
    s = pl.Series(
        [
            {"x": 1, "y": 2},
            {"x": 1, "y": 2},
            None,
            {"x": 2, "y": 1},
            None,
            {"x": 3, "y": 2},
            {"x": 3, "y": 2},
        ]
    )
    expected = [True, False, True, True, False, True, False]
    assert s.is_first_distinct().to_list() == expected
    # list
    s = pl.Series([[1, 2], [1, 2], None, [2, 3], None, [3, 4], [3, 4]])
    expected = [True, False, True, True, False, True, False]
    assert s.is_first_distinct().to_list() == expected


def test_is_last_distinct() -> None:
    # numeric
    s = pl.Series([1, 1, None, 2, None, 3, 3])
    expected = [False, True, False, True, True, False, True]
    assert s.is_last_distinct().to_list() == expected
    # str
    s = pl.Series(["x", "x", None, "y", None, "z", "z"])
    expected = [False, True, False, True, True, False, True]
    assert s.is_last_distinct().to_list() == expected
    # boolean
    s = pl.Series([True, True, None, False, None, False, False])
    expected = [False, True, False, False, True, False, True]
    assert s.is_last_distinct().to_list() == expected
    # struct
    s = pl.Series(
        [
            {"x": 1, "y": 2},
            {"x": 1, "y": 2},
            None,
            {"x": 2, "y": 1},
            None,
            {"x": 3, "y": 2},
            {"x": 3, "y": 2},
        ]
    )
    expected = [False, True, False, True, True, False, True]
    assert s.is_last_distinct().to_list() == expected


@pytest.mark.parametrize("dtypes", [pl.Int32, pl.String, pl.Boolean, pl.List(pl.Int32)])
def test_is_first_last_distinct_all_null(dtypes: PolarsDataType) -> None:
    s = pl.Series([None, None, None], dtype=dtypes)
    assert s.is_first_distinct().to_list() == [True, False, False]
    assert s.is_last_distinct().to_list() == [False, False, True]


@pytest.mark.parametrize(
    ("value", "dtype"),
    [
        (7, pl.Int64),
        (7, pl.Int32),
        (2.5, pl.Float64),
        (True, pl.Boolean),
        ("hello", pl.String),
        (b"hello", pl.Binary),
        (7, pl.Datetime("us")),
        (None, pl.Int64),
        (None, pl.String),
    ],
)
def test_distinct_of_a_repeated_element(value: Any, dtype: PolarsDataType) -> None:
    # A chunk that repeats a single element answers the whole `distinct` family without
    # one element being hashed, so the answers must match the ones read off a chunk that
    # holds every element in a slot of its own.
    for length in (1, 2, 5):
        flat = pl.Series("a", [value] * length, dtype=dtype).to_frame()
        repeated = (
            pl.select(pl.repeat(value, length, dtype=dtype))
            .to_series()
            .rename("a")
            .to_frame()
        )

        for expr in (
            pl.col("a").is_first_distinct(),
            pl.col("a").is_last_distinct(),
            pl.col("a").is_unique(),
            pl.col("a").is_duplicated(),
        ):
            assert_frame_equal(repeated.select(expr), flat.select(expr))


@pytest.mark.parametrize(
    ("value", "dtype"),
    [
        ([1, 2, 3], pl.List(pl.Int64)),
        ([], pl.List(pl.Int64)),
        ([None, None], pl.List(pl.Int64)),
        (["a", None], pl.List(pl.String)),
        ([1, 2], pl.Array(pl.Int64, 2)),
        ([None, None], pl.Array(pl.Int64, 2)),
        (["a", "b"], pl.Array(pl.String, 2)),
        ({"x": 1, "y": "a"}, pl.Struct({"x": pl.Int64, "y": pl.String})),
        ({"x": None, "y": None}, pl.Struct({"x": pl.Int64, "y": pl.String})),
        (None, pl.List(pl.Int64)),
        (None, pl.Struct({"x": pl.Int64})),
    ],
)
def test_distinct_of_a_repeated_nested_element(
    value: Any, dtype: PolarsDataType
) -> None:
    # The nested arms of the `distinct` family group on the rows to find their answer,
    # which for a nested type row-encodes the whole column first. A chunk that repeats a
    # single element is answered off the representation instead, and has to answer the
    # same thing as a chunk holding every element in a slot of its own.
    for length in (1, 2, 5):
        flat = pl.Series("a", [value] * length, dtype=dtype).to_frame()
        one = pl.Series("a", [value], dtype=dtype)
        repeated = (
            pl.DataFrame({"i": range(length)})
            .with_columns(pl.lit(one).first().alias("a"))
            .drop("i")
        )

        for expr in (
            pl.col("a").is_unique(),
            pl.col("a").is_duplicated(),
        ):
            assert_frame_equal(repeated.select(expr), flat.select(expr))

        if not isinstance(dtype, pl.Array):
            for expr in (
                pl.col("a").is_first_distinct(),
                pl.col("a").is_last_distinct(),
            ):
                assert_frame_equal(repeated.select(expr), flat.select(expr))
