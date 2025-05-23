from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.testing import assert_frame_equal, assert_series_equal

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


def test_search_sorted_multivalue() -> None:
    a = pl.Series([5, 7, 12, 14])
    df = pl.DataFrame({"a": a}).lazy()

    # This is the preferred way to do it:
    assert_series_equal(
        a.search_sorted(pl.Series([14, 7])), pl.Series([3, 1], dtype=pl.UInt32())
    )
    assert_frame_equal(
        df.select(pl.col("a").search_sorted(pl.Series([14, 7]))).collect(),
        pl.DataFrame({"a": pl.Series([3, 1], dtype=pl.UInt32())}),
    )

    # Searching for a list is deprecated and will be removed in Polars 2,
    # should only search with pl.Series for multiple values going forward:
    with pytest.deprecated_call():
        assert_series_equal(
            a.search_sorted([14, 7]),  # type: ignore[arg-type]
            pl.Series([3, 1], dtype=pl.UInt32()),
        )
    with pytest.deprecated_call():
        assert_frame_equal(
            df.select(pl.col("a").search_sorted([14, 7])).collect(),
            pl.DataFrame({"a": pl.Series([3, 1], dtype=pl.UInt32())}),
        )


@pytest.mark.parametrize("descending", [False, True])
def test_search_sorted_descending_order(descending: bool) -> None:
    values = sorted([2, 3, 4, 5], reverse=descending)
    series = pl.Series(values)
    df = pl.DataFrame({"series": series}).lazy()
    for value in values:
        expected_index = values.index(value)
        assert series.search_sorted(value, descending=descending) == expected_index
        assert df.select(
            pl.col("series").search_sorted(value, descending=descending)
        ).collect().get_column("series").to_list() == [expected_index]


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

    with pytest.raises(InvalidOperationError, match="cannot cast List type"):
        series.search_sorted([[1]])


def test_search_sorted_list_expr() -> None:
    df = pl.DataFrame({"values": pl.Series([[1], [2], [3]])}).lazy()
    C = pl.col("values")
    assert df.select(C.search_sorted([2])).collect().get_column("values").to_list() == [
        1
    ]
    assert df.select(C.search_sorted(pl.Series([[3], [2]]))).collect().get_column(
        "values"
    ).to_list() == [2, 1]
    # Wrong nesting value:
    with pytest.raises(InvalidOperationError):
        df.select(C.search_sorted([[3], [2]])).collect()

    # Now try with extra nesting:
    df = pl.DataFrame({"values": pl.Series([[[1]], [[2]], [[3]]])}).lazy()
    assert df.select(C.search_sorted([[2]])).collect().get_column(
        "values"
    ).to_list() == [1]
    assert df.select(C.search_sorted(pl.Series([[[3]], [[2]]]))).collect().get_column(
        "values"
    ).to_list() == [2, 1]
    # Wrong nesting values:
    with pytest.raises(InvalidOperationError):
        df.select(C.search_sorted([2])).collect()
    with pytest.raises(InvalidOperationError):
        df.select(C.search_sorted([[[3], [2]]])).collect()


def test_search_sorted_array() -> None:
    dtype = pl.Array(pl.Int64(), 1)
    series = pl.Series([[1], [2], [3]], dtype=dtype)
    assert series.search_sorted([2]) == 1
    assert series.search_sorted(pl.Series([[3], [2]], dtype=dtype)).to_list() == [2, 1]
    assert series.search_sorted(pl.lit([3])).to_list() == [2]
    assert series.search_sorted(pl.lit([3], dtype=dtype)).to_list() == [2]
    with pytest.raises(InvalidOperationError, match="casting from"):
        series.search_sorted([[1]])


def test_search_sorted_struct() -> None:
    v0 = {"a": 0, "b": 0}
    v1 = {"a": 1, "b": 2}
    series = pl.Series([v1, v0]).sort()
    assert series.search_sorted(v0) == 0
    assert series.search_sorted(v1) == 1
    assert series.search_sorted(pl.Series([v1, v0])).to_list() == [1, 0]


def assert_can_find_values(values: list[Any], dtype: PolarsDataType) -> None:
    series = pl.Series(values, dtype=dtype)
    for descending in [True, False]:
        for nulls_last in [True, False]:
            sorted_series = series.sort(descending=descending, nulls_last=nulls_last)
            as_list = sorted_series.to_list()
            for value in values:
                idx = sorted_series.search_sorted(value, "left", descending=descending)
                assert as_list.index(value) == idx
                lookup = sorted_series[idx]
                if isinstance(lookup, pl.Series):
                    lookup = lookup.to_list()
                assert lookup == value


@given(values=st.lists(st.integers(min_value=-10, max_value=10) | st.none()))
def test_nulls_and_ascending(values: list[int | None]) -> None:
    assert_can_find_values(values, pl.Int64())


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
    assert_can_find_values(values, dtype=pl.List(pl.Int64()))


@given(
    values=st.lists(
        st.none()
        | st.lists(
            st.integers(min_value=-10, max_value=10) | st.none(), min_size=3, max_size=3
        )
    )
)
@example(values=[[None, None, None], [0, 0, None]])
def test_search_sorted_array_with_nulls(values: list[list[int | None] | None]) -> None:
    """For all nulls_last options, values can be found in arbitrary arrays."""
    assert_can_find_values(values, dtype=pl.Array(pl.Int64(), 3))


@given(
    values=st.lists(
        st.none()
        | st.fixed_dictionaries(
            {"key": st.integers(min_value=-10, max_value=10) | st.none()}
        )
    )
)
@example(values=[{"key": 0}, {"key": None}])
def test_search_sorted_structs_with_nulls(
    values: list[dict[str, int | None] | None],
) -> None:
    """For all nulls_last options, values can be found in arbitrary structs."""
    assert_can_find_values(values, pl.Struct)
