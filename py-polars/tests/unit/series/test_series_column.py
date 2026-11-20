"""Tests for a `Series` that is backed by a `Column` (`PySeries` holds a `Column`)."""

from __future__ import annotations

import polars as pl
from polars.testing import assert_series_equal


def test_column_kind_plain_series() -> None:
    assert pl.Series("a", [1, 2, 3])._s._column_kind() == "series"
    assert pl.Series("a", [1], dtype=pl.Int64)._s._column_kind() == "series"
    assert pl.Series("a", [], dtype=pl.Int64)._s._column_kind() == "series"


def test_length_one_series_keeps_flags_and_sortedness() -> None:
    # A length-1 Series reports itself as ascending; it must stay a `Column::Series` (not
    # become a `Column::Scalar`) through the operations that return a new Series.
    s = pl.Series("a", [1])
    assert s.flags == {"SORTED_ASC": True, "SORTED_DESC": False}
    assert s.is_sorted()

    s = s.set_sorted(descending=True)
    assert s.flags == {"SORTED_ASC": True, "SORTED_DESC": False}
    assert s._s._column_kind() == "series"

    # The same flags as a longer Series with the flag set.
    s2 = pl.Series("a", [3, 2, 1]).set_sorted(descending=True)
    assert s2.flags == {"SORTED_ASC": False, "SORTED_DESC": True}
    assert s2.head(1).flags == {"SORTED_ASC": True, "SORTED_DESC": False}
    assert s2.head(1)._s._column_kind() == "series"

    s = pl.Series("a", [1])
    s.append(pl.Series("a", [0]))
    assert s.to_list() == [1, 0]


def test_in_place_mutation() -> None:
    s = pl.Series("a", [1, 2, 3])
    s.append(pl.Series("a", [4]))
    s.extend(pl.Series("a", [5]))
    s._s.rename("b")
    s.scatter(0, 10)
    s.shrink_to_fit(in_place=True)
    s.rechunk(in_place=True)
    assert_series_equal(s, pl.Series("b", [10, 2, 3, 4, 5]))
    assert s.n_chunks() == 1


def test_rolling_map_callback() -> None:
    s = pl.Series("a", [1.0, 2.0, 3.0, 4.0])
    result = s.rolling_map(lambda w: w.sum(), window_size=2)
    assert_series_equal(result, pl.Series("a", [None, 3.0, 5.0, 7.0]))


def test_fold_reduce_cum_fold_callbacks() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    out = df.select(
        fold=pl.fold(pl.lit(0), lambda acc, x: acc + x, exprs=pl.all()),
        reduce=pl.reduce(lambda acc, x: acc + x, exprs=pl.all()),
        cum=pl.cum_fold(pl.lit(0), lambda acc, x: acc + x, exprs=pl.all()),
    )
    assert out["fold"].to_list() == [11, 22, 33]
    assert out["reduce"].to_list() == [11, 22, 33]
    assert out["cum"].to_list() == [
        {"a": 1, "b": 11},
        {"a": 2, "b": 22},
        {"a": 3, "b": 33},
    ]
