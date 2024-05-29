from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_arr_to_list() -> None:
    s = pl.Series("a", [[1, 2], [4, 3]], dtype=pl.Array(pl.Int8, 2))

    result = s.arr.to_list()

    expected = pl.Series("a", [[1, 2], [4, 3]], dtype=pl.List(pl.Int8))
    assert_series_equal(result, expected)

    # test logical type
    data = {"duration": [[1000, 2000], None]}
    df = pl.DataFrame(
        data,
        schema={
            "duration": pl.Array(pl.Datetime, shape=2),
        },
    ).with_columns(pl.col("duration").arr.to_list())

    expected_df = pl.DataFrame(
        data,
        schema={
            "duration": pl.List(pl.Datetime),
        },
    )
    assert_frame_equal(df, expected_df)


def test_arr_to_list_lazy() -> None:
    s = pl.Series("a", [[1, 2], [4, 3]], dtype=pl.Array(pl.Int8, 2))
    lf = s.to_frame().lazy()

    result = lf.select(pl.col("a").arr.to_list())

    s = pl.Series("a", [[1, 2], [4, 3]], dtype=pl.List(pl.Int8))
    expected = s.to_frame().lazy()
    assert_frame_equal(result, expected)


def test_arr_to_list_nested_array_preserved() -> None:
    s = pl.Series(
        "a",
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        dtype=pl.Array(pl.Array(pl.Int8, 2), 2),
    )
    lf = s.to_frame().lazy()

    result = lf.select(pl.col("a").arr.to_list())

    s = pl.Series(
        "a",
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
    ).cast(pl.List(pl.Array(pl.Int8, 2)))
    expected = s.to_frame().lazy()
    assert_frame_equal(result, expected)
