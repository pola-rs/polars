from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pytest

import polars as pl

if TYPE_CHECKING:
    from polars.internals.type_aliases import ClosedWindow


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
        closed_windows: list[ClosedWindow] = ["left", "right", "none", "both"]
        for closed in closed_windows:

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
    assert s.rolling_skew(window_size=4, bias=True).to_list() == pytest.approx(
        [
            None,
            None,
            None,
            -0.49338220021815865,
            0.0,
            1.097025449363867,
            0.09770939201338157,
        ]
    )

    assert s.rolling_skew(window_size=4, bias=False).to_list() == pytest.approx(
        [
            None,
            None,
            None,
            -0.8545630383279711,
            0.0,
            1.9001038154942962,
            0.16923763134384154,
        ]
    )


def test_rolling_extrema() -> None:
    # sorted data and nulls flags trigger different kernels
    df = (
        pl.DataFrame(
            {
                "col1": pl.arange(0, 7, eager=True),
                "col2": pl.arange(0, 7, eager=True).reverse(),
            }
        )
    ).with_columns(
        [
            pl.when(pl.arange(0, pl.count(), eager=False) < 2)
            .then(None)
            .otherwise(pl.all())
            .suffix("_nulls")
        ]
    )

    assert df.select([pl.all().rolling_min(3)]).to_dict(False) == {
        "col1": [None, None, 0, 1, 2, 3, 4],
        "col2": [None, None, 4, 3, 2, 1, 0],
        "col1_nulls": [None, None, None, None, 2, 3, 4],
        "col2_nulls": [None, None, None, None, 2, 1, 0],
    }

    assert df.select([pl.all().rolling_max(3)]).to_dict(False) == {
        "col1": [None, None, 2, 3, 4, 5, 6],
        "col2": [None, None, 6, 5, 4, 3, 2],
        "col1_nulls": [None, None, None, None, 4, 5, 6],
        "col2_nulls": [None, None, None, None, 4, 3, 2],
    }

    # shuffled data triggers other kernels
    df = df.select([pl.all().shuffle(0)])
    assert df.select([pl.all().rolling_min(3)]).to_dict(False) == {
        "col1": [None, None, 0, 0, 1, 2, 2],
        "col2": [None, None, 0, 2, 1, 1, 1],
        "col1_nulls": [None, None, None, None, None, 2, 2],
        "col2_nulls": [None, None, None, None, None, 1, 1],
    }

    assert df.select([pl.all().rolling_max(3)]).to_dict(False) == {
        "col1": [None, None, 6, 4, 5, 5, 5],
        "col2": [None, None, 6, 6, 5, 4, 4],
        "col1_nulls": [None, None, None, None, None, 5, 5],
        "col2_nulls": [None, None, None, None, None, 4, 4],
    }
