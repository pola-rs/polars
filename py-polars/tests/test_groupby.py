from __future__ import annotations

from datetime import datetime

import polars as pl


def test_groupby_sorted_empty_dataframe_3680() -> None:
    assert (
        pl.DataFrame(
            [
                pl.Series("key", [], dtype=pl.Categorical),
                pl.Series("val", [], dtype=pl.Float64),
            ]
        )
        .lazy()
        .sort("key")
        .groupby("key")
        .tail(1)
        .collect()
    ).shape == (0, 2)


def test_groupby_custom_agg_empty_list() -> None:
    assert (
        pl.DataFrame(
            [
                pl.Series("key", [], dtype=pl.Categorical),
                pl.Series("val", [], dtype=pl.Float64),
            ]
        )
        .groupby("key")
        .agg(
            [
                pl.col("val").mean().alias("mean"),
                pl.col("val").std().alias("std"),
                pl.col("val").skew().alias("skew"),
                pl.col("val").kurtosis().alias("kurt"),
            ]
        )
    ).dtypes == [pl.Categorical, pl.Float64, pl.Float64, pl.Float64, pl.Float64]


def test_apply_after_take_in_groupby_3869() -> None:
    assert (
        pl.DataFrame(
            {
                "k": list("aaabbb"),
                "t": [1, 2, 3, 4, 5, 6],
                "v": [3, 1, 2, 5, 6, 4],
            }
        )
        .groupby("k", maintain_order=True)
        .agg(
            pl.col("v").take(pl.col("t").arg_max()).sqrt()
        )  # <- fails for sqrt, exp, log, pow, etc.
    ).to_dict(False) == {"k": ["a", "b"], "v": [1.4142135623730951, 2.0]}


def test_groupby_rolling_negative_offset_3914() -> None:
    df = pl.DataFrame(
        {
            "datetime": pl.date_range(datetime(2020, 1, 1), datetime(2020, 1, 5), "1d"),
        }
    )
    assert df.groupby_rolling(index_column="datetime", period="2d", offset="-4d").agg(
        pl.count().alias("count")
    )["count"].to_list() == [0, 0, 1, 2, 2]

    df = pl.DataFrame(
        {
            "ints": range(0, 20),
        }
    )

    assert df.groupby_rolling(index_column="ints", period="2i", offset="-5i",).agg(
        [pl.col("ints").alias("matches")]
    )["matches"].to_list() == [
        [],
        [],
        [],
        [0],
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [12, 13],
        [13, 14],
        [14, 15],
        [15, 16],
    ]
