from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.testing import assert_series_equal

if TYPE_CHECKING:
    from typing import Any

    from polars._typing import PolarsDataType


def test_search_sorted() -> None:
    for seed in [1, 2, 3]:
        np.random.seed(seed)
        arr = np.sort(np.random.randn(10) * 100)
        s = pl.Series(arr)

        for v in range(int(np.min(arr)), int(np.max(arr)), 20):
            assert np.searchsorted(arr, v) == s.search_sorted(v)

    a = pl.Series([1, 2, 3])
    b = pl.Series([1, 2, 2, -1])
    assert a.search_sorted(b).to_list() == [0, 1, 1, 0]
    b = pl.Series([1, 2, 2, None, 3])
    assert a.search_sorted(b).to_list() == [0, 1, 1, 0, 2]

    a = pl.Series(["b", "b", "d", "d"])
    b = pl.Series(["a", "b", "c", "d", "e"])
    assert a.search_sorted(b, side="left").to_list() == [0, 0, 2, 2, 4]
    assert a.search_sorted(b, side="right").to_list() == [0, 2, 2, 4, 4]

    a = pl.Series([1, 1, 4, 4])
    b = pl.Series([0, 1, 2, 4, 5])
    assert a.search_sorted(b, side="left").to_list() == [0, 0, 2, 2, 4]
    assert a.search_sorted(b, side="right").to_list() == [0, 2, 2, 4, 4]


def test_search_sorted_multichunk() -> None:
    for seed in [1, 2, 3]:
        np.random.seed(seed)
        arr = np.sort(np.random.randn(10) * 100)
        q = len(arr) // 4
        a, b, c, d = map(
            pl.Series, (arr[:q], arr[q : 2 * q], arr[2 * q : 3 * q], arr[3 * q :])
        )
        s = pl.concat([a, b, c, d], rechunk=False)
        assert s.n_chunks() == 4

        for v in range(int(np.min(arr)), int(np.max(arr)), 20):
            assert np.searchsorted(arr, v) == s.search_sorted(v)

    a = pl.concat(
        [
            pl.Series([None, None, None], dtype=pl.Int64),
            pl.Series([None, 1, 1, 2, 3]),
            pl.Series([4, 4, 5, 6, 7, 8, 8]),
        ],
        rechunk=False,
    )
    assert a.n_chunks() == 3
    b = pl.Series([-10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, None])
    left_ref = pl.Series(
        [4, 4, 4, 6, 7, 8, 10, 11, 12, 13, 15, 0], dtype=pl.get_index_type()
    )
    right_ref = pl.Series(
        [4, 4, 6, 7, 8, 10, 11, 12, 13, 15, 15, 4], dtype=pl.get_index_type()
    )
    assert_series_equal(a.search_sorted(b, side="left"), left_ref)
    assert_series_equal(a.search_sorted(b, side="right"), right_ref)


def test_search_sorted_right_nulls() -> None:
    a = pl.Series([1, 2, None, None])
    assert a.search_sorted(None, side="left") == 2
    assert a.search_sorted(None, side="right") == 4


def test_raise_literal_numeric_search_sorted_18096() -> None:
    df = pl.DataFrame({"foo": [1, 4, 7], "bar": [2, 3, 5]})

    with pytest.raises(pl.exceptions.InvalidOperationError):
        df.with_columns(idx=pl.col("foo").search_sorted("bar"))


def test_search_sorted_list() -> None:
    series = pl.Series([[1], [2], [3]])
    assert series.search_sorted([2]) == 1
    assert series.search_sorted(pl.Series([[3], [2]])).to_list() == [2, 1]
    assert series.search_sorted(pl.lit([3], dtype=pl.List(pl.Int64()))).to_list() == [2]
    with pytest.raises(
        TypeError, match="If you were trying to search for multiple values"
    ):
        series.search_sorted([[1]])  # type: ignore[list-item]


def test_search_sorted_array() -> None:
    dtype = pl.Array(pl.Int64(), 1)
    series = pl.Series([[1], [2], [3]], dtype=dtype)
    assert series.search_sorted([2]) == 1
    assert series.search_sorted(pl.Series([[3], [2]], dtype=dtype)).to_list() == [2, 1]
    assert series.search_sorted(pl.lit([3])).to_list() == [2]
    assert series.search_sorted(pl.lit([3], dtype=dtype)).to_list() == [2]
    with pytest.raises(
        TypeError, match="If you were trying to search for multiple values"
    ):
        series.search_sorted([[1]])  # type: ignore[list-item]


def test_search_sorted_struct() -> None:
    v0 = {"a": 0, "b": 0}
    v1 = {"a": 1, "b": 2}
    series = pl.Series([v1, v0]).sort()
    assert series.search_sorted(v0) == 0
    assert series.search_sorted(v1) == 1
    assert series.search_sorted([v1, v0]).to_list() == [1, 0]


def assert_find_existing_values(series: pl.Series, values: list[Any]) -> None:
    as_list = series.to_list()
    for value in values:
        idx = series.search_sorted(value, "left")
        assert as_list.index(value) == idx
        lookup = series[idx]
        if isinstance(lookup, pl.Series):
            lookup = lookup.to_list()
        assert lookup == value


@given(values=st.lists(st.integers(min_value=-10, max_value=10) | st.none()))
def test_nulls_and_ascending(values: list[int | None]) -> None:
    series = pl.Series(values, dtype=pl.Int64())
    for descending in [True, False]:
        for nulls_last in [True, False]:
            sorted_series = series.sort(descending=descending, nulls_last=nulls_last)
            assert_find_existing_values(sorted_series, values)


@given(
    values=st.lists(
        st.none() | st.lists(st.integers(min_value=-10, max_value=10) | st.none())
    )
)
@example([[1], [-2], None, [None, 3], [3, None]])
@example([[None], [0]])
@example([[], [None], [0]])
def test_search_sorted_list_with_nulls(values: list[list[int | None] | None]) -> None:
    """For all nulls_last options, values can be found in arbitrary lists."""
    series = pl.Series(values, dtype=pl.List(pl.Int64()))
    for nulls_last in [True, False]:
        sorted_series = series.sort(descending=False, nulls_last=nulls_last)
        assert_find_existing_values(sorted_series, values)


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        ([[1, 2], [3, 4]], pl.List(pl.Int64())),
        ([[1, 2], [3, 4]], pl.Array(pl.Int64(), 2)),
        ([{"a": 2, "b": 3}, {"a": 7, "b": 9}], None),
    ],
)
def test_nested_dtypes_dont_support_ascending(
    values: list[Any], dtype: PolarsDataType
) -> None:
    # Due to issues with row encoding implementation details, search an
    # descending list/array/struct Series is unsupported, for now at least.
    series = pl.Series(values, dtype=dtype).sort(descending=True)
    with pytest.raises(
        InvalidOperationError,
        match="descending sort is not supported in nested dtypes",
    ):
        series.search_sorted(values[0])
