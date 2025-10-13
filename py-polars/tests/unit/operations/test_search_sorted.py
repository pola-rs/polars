from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.testing import assert_series_equal

if TYPE_CHECKING:
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


@pytest.mark.parametrize("dtype", [pl.Categorical(), pl.Enum(["a", "b", "c"])])
def test_search_sorted_enum_categorical(dtype: PolarsDataType) -> None:
    series = pl.Series(["a", None, "b", "c"], dtype=dtype)
    desc_nulls_last = series.sort(descending=True, nulls_last=True)
    assert desc_nulls_last.search_sorted(None, "left", descending=True) == 3
    assert desc_nulls_last.search_sorted("a", "left", descending=True) == 2

    desc_nulls_first = series.sort(descending=True, nulls_last=False)
    assert desc_nulls_first.search_sorted(None, "left", descending=True) == 0
    assert desc_nulls_first.search_sorted("a", "left", descending=True) == 3

    asc_nulls_last = series.sort(descending=False, nulls_last=True)
    assert asc_nulls_last.search_sorted(None, "left", descending=False) == 3
    assert asc_nulls_last.search_sorted("a", "left", descending=False) == 0

    asc_nulls_first = series.sort(descending=False, nulls_last=False)
    assert asc_nulls_first.search_sorted(None, "left", descending=False) == 0
    assert asc_nulls_first.search_sorted("a", "left", descending=False) == 1


def test_enum_wrong_value() -> None:
    series = pl.Series(["a", "b"], dtype=pl.Enum(["a", "b"]))
    with pytest.raises(
        InvalidOperationError, match="conversion from `str` to `enum` failed"
    ):
        series.search_sorted("c")


@pytest.mark.parametrize("value", [0, 0.1])
def test_categorical_wrong_type_keys_dont_work(value: int | float) -> None:
    series = pl.Series(["a", "c", None, "b"], dtype=pl.Categorical)
    msg = "got invalid or ambiguous dtypes"
    with pytest.raises(InvalidOperationError, match=msg):
        series.search_sorted(value)
    df = pl.DataFrame({"s": series})
    with pytest.raises(InvalidOperationError, match=msg):
        df.select(pl.col("s").search_sorted(value))
