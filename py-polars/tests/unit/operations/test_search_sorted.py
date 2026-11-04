from typing import Literal

import numpy as np
import pytest

import polars as pl
from polars.testing import assert_series_equal


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

    a = pl.Series([1, 1, 4, 4], dtype=pl.Decimal(3))
    b = pl.Series([0, 1, 2, 4, 5], dtype=pl.Decimal(3))
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


def test_search_sorted_list_ambiguous_22058() -> None:
    with pytest.raises(pl.exceptions.InvalidOperationError):
        pl.Series([1, 2, 3]).search_sorted([1, 2])  # type: ignore[call-overload]

    with pytest.raises(pl.exceptions.InvalidOperationError):
        pl.DataFrame({"a": [1, 2, 3]}).select(pl.col("a").search_sorted([1, 2]))


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("side", ["any", "left", "right"])
def test_search_sorted_needle_blocks(
    descending: bool, side: Literal["any", "left", "right"]
) -> None:
    # Sorted and shuffled runs of needles, with nulls and NaN, past the parallel
    # threshold. A haystack in two chunks takes the per-needle search, which
    # serves as the reference.
    rng = np.random.default_rng(0)
    hay = np.concatenate(
        [rng.integers(-50, 50, 1000).astype(float), [np.nan] * 5, [-0.0, 0.0]]
    )
    haystack = pl.Series(hay).sort(descending=descending)
    # One chunk without nulls takes the slice search.
    assert haystack.n_chunks() == 1
    assert not haystack.has_nulls()
    split = pl.concat([haystack[:500], haystack[500:]], rechunk=False)
    assert split.n_chunks() == 2

    n = 200_000
    vals = rng.integers(-60, 60, n).astype(float)
    vals[rng.random(n) < 0.05] = np.nan
    for run in [slice(0, 70_000), slice(130_000, n)]:
        vals[run] = np.sort(vals[run])
        if descending:
            vals[run] = vals[run][::-1]
    needles = pl.Series(vals).set(pl.Series(rng.random(n) < 0.02), None)

    result = haystack.search_sorted(needles, side=side, descending=descending)
    expected = split.search_sorted(needles, side=side, descending=descending)
    assert_series_equal(result, expected)
