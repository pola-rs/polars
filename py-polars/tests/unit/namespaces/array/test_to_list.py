from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_arr_to_list() -> None:
    s = pl.Series("a", [[1, 2], [4, 3]], dtype=pl.Array(inner=pl.Int8, width=2))

    result = s.arr.to_list()

    expected = pl.Series("a", [[1, 2], [4, 3]], dtype=pl.List(pl.Int8))
    assert_series_equal(result, expected)


def test_arr_to_list_lazy() -> None:
    s = pl.Series("a", [[1, 2], [4, 3]], dtype=pl.Array(inner=pl.Int8, width=2))
    lf = s.to_frame().lazy()

    result = lf.select(pl.col("a").arr.to_list())

    s = pl.Series("a", [[1, 2], [4, 3]], dtype=pl.List(pl.Int8))
    expected = s.to_frame().lazy()
    assert_frame_equal(result, expected)


def test_arr_to_list_nested_array_preserved() -> None:
    s = pl.Series(
        "a",
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        dtype=pl.Array(inner=pl.Array(inner=pl.Int8, width=2), width=2),
    )
    lf = s.to_frame().lazy()

    result = lf.select(pl.col("a").arr.to_list())

    s = pl.Series(
        "a",
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
    ).cast(pl.List(pl.Array(inner=pl.Int8, width=2)))
    expected = s.to_frame().lazy()
    assert_frame_equal(result, expected)
