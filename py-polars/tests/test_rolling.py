from __future__ import annotations

from datetime import datetime

import polars as pl


def test_rolling_kernels_and_groupby_rolling() -> None:
    df = pl.DataFrame(
        {
            "dt": [
                datetime(2021, 1, 1),
                datetime(2021, 1, 2),
                datetime(2021, 1, 4),
                datetime(2021, 1, 5),
                datetime(2021, 1, 7),
            ],
            "values": pl.arange(0, 5, eager=True),
        }
    )
    for period in ["1d", "2d", "3d"]:
        for closed in ["left", "right", "none", "both"]:

            out1 = df.select(
                [
                    pl.col("dt"),
                    pl.col("values")
                    .rolling_sum(period, by="dt", closed=closed)
                    .alias("sum"),
                    pl.col("values")
                    .rolling_var(period, by="dt", closed=closed)
                    .alias("var"),
                    pl.col("values")
                    .rolling_mean(period, by="dt", closed=closed)
                    .alias("mean"),
                    pl.col("values")
                    .rolling_std(period, by="dt", closed=closed)
                    .alias("std"),
                ]
            )

            out2 = df.groupby_rolling("dt", period=period, closed=closed).agg(
                [
                    pl.col("values").sum().alias("sum"),
                    pl.col("values").var().alias("var"),
                    pl.col("values").mean().alias("mean"),
                    pl.col("values").std().alias("std"),
                ]
            )
            pl.testing.assert_frame_equal(out1, out2)


def test_rolling_skew() -> None:
    s = pl.Series([1, 2, 3, 3, 2, 10, 8])
    assert s.rolling_skew(window_size=4, bias=True).to_list() == [
        None,
        None,
        None,
        -0.49338220021815865,
        0.0,
        1.097025449363867,
        0.09770939201338157,
    ]

    assert s.rolling_skew(window_size=4, bias=False).to_list() == [
        None,
        None,
        None,
        -0.8545630383279711,
        0.0,
        1.9001038154942962,
        0.16923763134384154,
    ]
