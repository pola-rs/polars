from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from numpy import nan

from polars.exceptions import ComputeError

if sys.version_info >= (3, 9):
    from zoneinfo import ZoneInfo
else:
    # Import from submodule due to typing issue with backports.zoneinfo package:
    # https://github.com/pganssle/zoneinfo/issues/125
    from backports.zoneinfo._zoneinfo import ZoneInfo

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars.type_aliases import ClosedInterval


@pytest.fixture()
def example_df() -> pl.DataFrame:
    return pl.DataFrame(
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


@pytest.mark.parametrize(
    "period",
    ["1d", "2d", "3d", timedelta(days=1), timedelta(days=2), timedelta(days=3)],
)
@pytest.mark.parametrize("closed", ["left", "right", "none", "both"])
def test_rolling_kernels_and_groupby_rolling(
    example_df: pl.DataFrame, period: str | timedelta, closed: ClosedInterval
) -> None:
    out1 = example_df.select(
        [
            pl.col("dt"),
            # this differs from groupby aggregation because the empty window is
            # null here
            # where the sum aggregation of an empty set is 0
            pl.col("values")
            .rolling_sum(period, by="dt", closed=closed)
            .fill_null(0)
            .alias("sum"),
            pl.col("values").rolling_var(period, by="dt", closed=closed).alias("var"),
            pl.col("values").rolling_mean(period, by="dt", closed=closed).alias("mean"),
            pl.col("values").rolling_std(period, by="dt", closed=closed).alias("std"),
        ]
    )
    out2 = (
        example_df.set_sorted("dt")
        .groupby_rolling("dt", period=period, closed=closed)
        .agg(
            [
                pl.col("values").sum().alias("sum"),
                pl.col("values").var().alias("var"),
                pl.col("values").mean().alias("mean"),
                pl.col("values").std().alias("std"),
            ]
        )
    )
    assert_frame_equal(out1, out2)


@pytest.mark.parametrize(
    ("offset", "closed", "expected_values"),
    [
        pytest.param(
            "-1d",
            "left",
            [[1], [1, 2], [2, 3], [3, 4]],
            id="partial lookbehind, left",
        ),
        pytest.param(
            "-1d",
            "right",
            [[1, 2], [2, 3], [3, 4], [4]],
            id="partial lookbehind, right",
        ),
        pytest.param(
            "-1d",
            "both",
            [[1, 2], [1, 2, 3], [2, 3, 4], [3, 4]],
            id="partial lookbehind, both",
        ),
        pytest.param(
            "-1d",
            "none",
            [[1], [2], [3], [4]],
            id="partial lookbehind, none",
        ),
        pytest.param(
            "-2d",
            "left",
            [[], [1], [1, 2], [2, 3]],
            id="full lookbehind, left",
        ),
        pytest.param(
            "-3d",
            "left",
            [[], [], [1], [1, 2]],
            id="full lookbehind, offset > period, left",
        ),
        pytest.param(
            "-3d",
            "right",
            [[], [1], [1, 2], [2, 3]],
            id="full lookbehind, right",
        ),
        pytest.param(
            "-3d",
            "both",
            [[], [1], [1, 2], [1, 2, 3]],
            id="full lookbehind, both",
        ),
        pytest.param(
            "-2d",
            "none",
            [[], [1], [2], [3]],
            id="full lookbehind, none",
        ),
        pytest.param(
            "-3d",
            "none",
            [[], [], [1], [2]],
            id="full lookbehind, offset > period, none",
        ),
    ],
)
def test_rolling_negative_offset(
    offset: str, closed: ClosedInterval, expected_values: list[list[int]]
) -> None:
    df = pl.DataFrame(
        {
            "ts": pl.date_range(
                datetime(2021, 1, 1), datetime(2021, 1, 4), "1d", eager=True
            ),
            "value": [1, 2, 3, 4],
        }
    )
    result = df.groupby_rolling("ts", period="2d", offset=offset, closed=closed).agg(
        pl.col("value")
    )
    expected = pl.DataFrame(
        {
            "ts": pl.date_range(
                datetime(2021, 1, 1), datetime(2021, 1, 4), "1d", eager=True
            ),
            "value": expected_values,
        }
    )
    assert_frame_equal(result, expected)


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


@pytest.mark.parametrize("time_zone", [None, "US/Central"])
@pytest.mark.parametrize(
    ("rolling_fn", "expected_values"),
    [
        ("rolling_mean", [None, 1.0, 2.0, 3.0, 4.0, 5.0]),
        ("rolling_sum", [None, 1, 2, 3, 4, 5]),
        ("rolling_min", [None, 1, 2, 3, 4, 5]),
        ("rolling_max", [None, 1, 2, 3, 4, 5]),
        ("rolling_std", [None, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("rolling_var", [None, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ],
)
def test_rolling_crossing_dst(
    time_zone: str | None, rolling_fn: str, expected_values: list[int | None | float]
) -> None:
    ts = pl.date_range(
        datetime(2021, 11, 5), datetime(2021, 11, 10), "1d", time_zone="UTC", eager=True
    ).dt.replace_time_zone(time_zone)
    df = pl.DataFrame({"ts": ts, "value": [1, 2, 3, 4, 5, 6]})
    result = df.with_columns(
        getattr(pl.col("value"), rolling_fn)("1d", by="ts", closed="left")
    )
    expected = pl.DataFrame({"ts": ts, "value": expected_values})
    assert_frame_equal(result, expected)


def test_rolling_extrema() -> None:
    # sorted data and nulls flags trigger different kernels
    df = (
        pl.DataFrame(
            {
                "col1": pl.int_range(0, 7, eager=True),
                "col2": pl.int_range(0, 7, eager=True).reverse(),
            }
        )
    ).with_columns(
        [
            pl.when(pl.int_range(0, pl.count(), eager=False) < 2)
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
    df = df.select([pl.all().shuffle(0, fixed_seed=True)])
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


def test_rolling_groupby_extrema() -> None:
    # ensure we hit different branches so create

    df = pl.DataFrame(
        {
            "col1": pl.arange(0, 7, eager=True).reverse(),
        }
    ).with_columns(pl.col("col1").reverse().alias("row_nr"))

    assert (
        df.groupby_rolling(
            index_column="row_nr",
            period="3i",
        )
        .agg(
            [
                pl.col("col1").suffix("_list"),
                pl.col("col1").min().suffix("_min"),
                pl.col("col1").max().suffix("_max"),
                pl.col("col1").first().alias("col1_first"),
                pl.col("col1").last().alias("col1_last"),
            ]
        )
        .select(["col1_list", "col1_min", "col1_max", "col1_first", "col1_last"])
    ).to_dict(False) == {
        "col1_list": [
            [6],
            [6, 5],
            [6, 5, 4],
            [5, 4, 3],
            [4, 3, 2],
            [3, 2, 1],
            [2, 1, 0],
        ],
        "col1_min": [6, 5, 4, 3, 2, 1, 0],
        "col1_max": [6, 6, 6, 5, 4, 3, 2],
        "col1_first": [6, 6, 6, 5, 4, 3, 2],
        "col1_last": [6, 5, 4, 3, 2, 1, 0],
    }

    # ascending order

    df = pl.DataFrame(
        {
            "col1": pl.arange(0, 7, eager=True),
        }
    ).with_columns(pl.col("col1").alias("row_nr"))

    assert (
        df.groupby_rolling(
            index_column="row_nr",
            period="3i",
        )
        .agg(
            [
                pl.col("col1").suffix("_list"),
                pl.col("col1").min().suffix("_min"),
                pl.col("col1").max().suffix("_max"),
                pl.col("col1").first().alias("col1_first"),
                pl.col("col1").last().alias("col1_last"),
            ]
        )
        .select(["col1_list", "col1_min", "col1_max", "col1_first", "col1_last"])
    ).to_dict(False) == {
        "col1_list": [
            [0],
            [0, 1],
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
        ],
        "col1_min": [0, 0, 0, 1, 2, 3, 4],
        "col1_max": [0, 1, 2, 3, 4, 5, 6],
        "col1_first": [0, 0, 0, 1, 2, 3, 4],
        "col1_last": [0, 1, 2, 3, 4, 5, 6],
    }

    # shuffled data.
    df = pl.DataFrame(
        {
            "col1": pl.arange(0, 7, eager=True).shuffle(1),
        }
    ).with_columns(pl.col("col1").sort().alias("row_nr"))

    assert (
        df.groupby_rolling(
            index_column="row_nr",
            period="3i",
        )
        .agg(
            [
                pl.col("col1").min().suffix("_min"),
                pl.col("col1").max().suffix("_max"),
                pl.col("col1").suffix("_list"),
            ]
        )
        .select(["col1_list", "col1_min", "col1_max"])
    ).to_dict(False) == {
        "col1_list": [
            [3],
            [3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 2],
            [6, 2, 1],
            [2, 1, 0],
        ],
        "col1_min": [3, 3, 3, 4, 2, 1, 0],
        "col1_max": [3, 4, 5, 6, 6, 6, 2],
    }


def test_rolling_slice_pushdown() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "a", "b"], "c": [1, 3, 5]}).lazy()
    df = (
        df.sort("a")
        .groupby_rolling(
            "a",
            by="b",
            period="2i",
        )
        .agg(
            [
                (pl.col("c") - pl.col("c").shift_and_fill(fill_value=0, periods=1))
                .sum()
                .alias("c")
            ]
        )
    )
    assert df.head(2).collect().to_dict(False) == {
        "b": ["a", "a"],
        "a": [1, 2],
        "c": [1, 3],
    }


def test_groupby_dynamic_slice_pushdown() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "a", "b"], "c": [1, 3, 5]}).lazy()
    df = (
        df.sort("a")
        .groupby_dynamic(
            "a",
            by="b",
            every="2i",
        )
        .agg(
            [
                (pl.col("c") - pl.col("c").shift_and_fill(fill_value=0, periods=1))
                .sum()
                .alias("c")
            ]
        )
    )
    assert df.head(2).collect().to_dict(False) == {
        "b": ["a", "a"],
        "a": [0, 2],
        "c": [1, 3],
    }


def test_overlapping_groups_4628() -> None:
    df = pl.DataFrame(
        {
            "index": [1, 2, 3, 4, 5, 6],
            "val": [10, 20, 40, 70, 110, 160],
        }
    )
    assert (
        df.groupby_rolling(index_column=pl.col("index").set_sorted(), period="3i").agg(
            [
                pl.col("val").diff(n=1).alias("val.diff"),
                (pl.col("val") - pl.col("val").shift(1)).alias("val - val.shift"),
            ]
        )
    ).to_dict(False) == {
        "index": [1, 2, 3, 4, 5, 6],
        "val.diff": [
            [None],
            [None, 10],
            [None, 10, 20],
            [None, 20, 30],
            [None, 30, 40],
            [None, 40, 50],
        ],
        "val - val.shift": [
            [None],
            [None, 10],
            [None, 10, 20],
            [None, 20, 30],
            [None, 30, 40],
            [None, 40, 50],
        ],
    }


def test_rolling_skew_lagging_null_5179() -> None:
    s = pl.Series([None, 3, 4, 1, None, None, None, None, 3, None, 5, 4, 7, 2, 1, None])
    assert s.rolling_skew(3).fill_nan(-1.0).to_list() == [
        None,
        None,
        0.0,
        -0.3818017741606059,
        0.0,
        -1.0,
        None,
        None,
        -1.0,
        -1.0,
        0.0,
        0.0,
        0.38180177416060695,
        0.23906314692954517,
        0.6309038567106234,
        0.0,
    ]


def test_rolling_var_numerical_stability_5197() -> None:
    s = pl.Series([*[1.2] * 4, *[3.3] * 7])
    res = s.to_frame("a").with_columns(pl.col("a").rolling_var(5))[:, 0].to_list()
    assert res[4:] == pytest.approx(
        [
            0.882,
            1.3229999999999997,
            1.3229999999999997,
            0.8819999999999983,
            0.0,
            0.0,
            0.0,
        ]
    )
    assert res[:4] == [None] * 4


@pytest.mark.parametrize(
    ("every", "offset"),
    [
        ("3d", "-1d"),
        (timedelta(days=3), timedelta(days=-1)),
    ],
)
def test_dynamic_groupby_timezone_awareness(
    every: str | timedelta, offset: str | timedelta
) -> None:
    df = pl.DataFrame(
        (
            pl.date_range(
                datetime(2020, 1, 1),
                datetime(2020, 1, 10),
                timedelta(days=1),
                time_unit="ns",
                eager=True,
            )
            .alias("datetime")
            .dt.replace_time_zone("UTC"),
            pl.arange(1, 11, eager=True).alias("value"),
        )
    )

    assert (
        df.groupby_dynamic(
            "datetime",
            every=every,
            offset=offset,
            closed="right",
            include_boundaries=True,
            truncate=False,
        ).agg(pl.col("value").last())
    ).dtypes == [pl.Datetime("ns", "UTC")] * 3 + [pl.Int64]


@pytest.mark.parametrize("tzinfo", [None, ZoneInfo("Asia/Kathmandu")])
def test_groupby_dynamic_startby_5599(tzinfo: ZoneInfo | None) -> None:
    # start by datapoint
    start = datetime(2022, 12, 16, tzinfo=tzinfo)
    stop = datetime(2022, 12, 16, hour=3, tzinfo=tzinfo)
    df = pl.DataFrame({"date": pl.date_range(start, stop, "30m", eager=True)})

    assert df.groupby_dynamic(
        "date",
        every="31m",
        include_boundaries=True,
        truncate=False,
        start_by="datapoint",
    ).agg(pl.count()).to_dict(False) == {
        "_lower_boundary": [
            datetime(2022, 12, 16, 0, 0, tzinfo=tzinfo),
            datetime(2022, 12, 16, 0, 31, tzinfo=tzinfo),
            datetime(2022, 12, 16, 1, 2, tzinfo=tzinfo),
            datetime(2022, 12, 16, 1, 33, tzinfo=tzinfo),
            datetime(2022, 12, 16, 2, 4, tzinfo=tzinfo),
            datetime(2022, 12, 16, 2, 35, tzinfo=tzinfo),
        ],
        "_upper_boundary": [
            datetime(2022, 12, 16, 0, 31, tzinfo=tzinfo),
            datetime(2022, 12, 16, 1, 2, tzinfo=tzinfo),
            datetime(2022, 12, 16, 1, 33, tzinfo=tzinfo),
            datetime(2022, 12, 16, 2, 4, tzinfo=tzinfo),
            datetime(2022, 12, 16, 2, 35, tzinfo=tzinfo),
            datetime(2022, 12, 16, 3, 6, tzinfo=tzinfo),
        ],
        "date": [
            datetime(2022, 12, 16, 0, 0, tzinfo=tzinfo),
            datetime(2022, 12, 16, 1, 0, tzinfo=tzinfo),
            datetime(2022, 12, 16, 1, 30, tzinfo=tzinfo),
            datetime(2022, 12, 16, 2, 0, tzinfo=tzinfo),
            datetime(2022, 12, 16, 2, 30, tzinfo=tzinfo),
            datetime(2022, 12, 16, 3, 0, tzinfo=tzinfo),
        ],
        "count": [2, 1, 1, 1, 1, 1],
    }

    # start by monday
    start = datetime(2022, 1, 1, tzinfo=tzinfo)
    stop = datetime(2022, 1, 12, 7, tzinfo=tzinfo)

    df = pl.DataFrame(
        {"date": pl.date_range(start, stop, "12h", eager=True)}
    ).with_columns(pl.col("date").dt.weekday().alias("day"))

    result = df.groupby_dynamic(
        "date",
        every="1w",
        period="3d",
        include_boundaries=True,
        start_by="monday",
        truncate=False,
    ).agg([pl.count(), pl.col("day").first().alias("data_day")])
    assert result.to_dict(False) == {
        "_lower_boundary": [
            datetime(2022, 1, 3, 0, 0, tzinfo=tzinfo),
            datetime(2022, 1, 10, 0, 0, tzinfo=tzinfo),
        ],
        "_upper_boundary": [
            datetime(2022, 1, 6, 0, 0, tzinfo=tzinfo),
            datetime(2022, 1, 13, 0, 0, tzinfo=tzinfo),
        ],
        "date": [
            datetime(2022, 1, 3, 0, 0, tzinfo=tzinfo),
            datetime(2022, 1, 10, 0, 0, tzinfo=tzinfo),
        ],
        "count": [6, 5],
        "data_day": [1, 1],
    }
    # start by saturday
    result = df.groupby_dynamic(
        "date",
        every="1w",
        period="3d",
        include_boundaries=True,
        start_by="saturday",
        truncate=False,
    ).agg([pl.count(), pl.col("day").first().alias("data_day")])
    assert result.to_dict(False) == {
        "_lower_boundary": [
            datetime(2022, 1, 1, 0, 0, tzinfo=tzinfo),
            datetime(2022, 1, 8, 0, 0, tzinfo=tzinfo),
        ],
        "_upper_boundary": [
            datetime(2022, 1, 4, 0, 0, tzinfo=tzinfo),
            datetime(2022, 1, 11, 0, 0, tzinfo=tzinfo),
        ],
        "date": [
            datetime(2022, 1, 1, 0, 0, tzinfo=tzinfo),
            datetime(2022, 1, 8, 0, 0, tzinfo=tzinfo),
        ],
        "count": [6, 6],
        "data_day": [6, 6],
    }


def test_groupby_dynamic_by_monday_and_offset_5444() -> None:
    df = pl.DataFrame(
        {
            "date": [
                "2022-11-01",
                "2022-11-02",
                "2022-11-05",
                "2022-11-08",
                "2022-11-08",
                "2022-11-09",
                "2022-11-10",
            ],
            "label": ["a", "b", "a", "a", "b", "a", "b"],
            "value": [1, 2, 3, 4, 5, 6, 7],
        }
    ).with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d").set_sorted())

    result = df.groupby_dynamic(
        "date", every="1w", offset="1d", by="label", start_by="monday"
    ).agg(pl.col("value").sum())

    assert result.to_dict(False) == {
        "label": ["a", "a", "b", "b"],
        "date": [
            date(2022, 11, 1),
            date(2022, 11, 8),
            date(2022, 11, 1),
            date(2022, 11, 8),
        ],
        "value": [4, 10, 2, 12],
    }

    # test empty
    result_empty = (
        df.filter(pl.col("date") == date(1, 1, 1))
        .groupby_dynamic("date", every="1w", offset="1d", by="label", start_by="monday")
        .agg(pl.col("value").sum())
    )
    assert result_empty.schema == result.schema


def test_groupby_rolling_iter() -> None:
    df = pl.DataFrame(
        {
            "date": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 5)],
            "a": [1, 2, 2],
            "b": [4, 5, 6],
        }
    ).set_sorted("date")

    # Without 'by' argument
    result1 = [
        (name, data.shape)
        for name, data in df.groupby_rolling(index_column="date", period="2d")
    ]
    expected1 = [
        (date(2020, 1, 1), (1, 3)),
        (date(2020, 1, 2), (2, 3)),
        (date(2020, 1, 5), (1, 3)),
    ]
    assert result1 == expected1

    # With 'by' argument
    result2 = [
        (name, data.shape)
        for name, data in df.groupby_rolling(index_column="date", period="2d", by="a")
    ]
    expected2 = [
        ((1, date(2020, 1, 1)), (1, 3)),
        ((2, date(2020, 1, 2)), (1, 3)),
        ((2, date(2020, 1, 5)), (1, 3)),
    ]
    assert result2 == expected2


def test_groupby_rolling_negative_period() -> None:
    df = pl.DataFrame({"ts": [datetime(2020, 1, 1)], "value": [1]}).with_columns(
        pl.col("ts").set_sorted()
    )
    with pytest.raises(
        ComputeError, match="rolling window period should be strictly positive"
    ):
        df.groupby_rolling("ts", period="-1d", offset="-1d").agg(pl.col("value"))
    with pytest.raises(
        ComputeError, match="rolling window period should be strictly positive"
    ):
        df.lazy().groupby_rolling("ts", period="-1d", offset="-1d").agg(
            pl.col("value")
        ).collect()
    with pytest.raises(ComputeError, match="window size should be strictly positive"):
        df.select(
            pl.col("value").rolling_min(by="ts", window_size="-1d", closed="left")
        )
    with pytest.raises(ComputeError, match="window size should be strictly positive"):
        df.lazy().select(
            pl.col("value").rolling_min(by="ts", window_size="-1d", closed="left")
        ).collect()


def test_rolling_skew_window_offset() -> None:
    assert (pl.arange(0, 20, eager=True) ** 2).rolling_skew(20)[
        -1
    ] == 0.6612545648596286


def test_rolling_kernels_groupby_dynamic_7548() -> None:
    assert pl.DataFrame(
        {"time": pl.arange(0, 4, eager=True), "value": pl.arange(0, 4, eager=True)}
    ).groupby_dynamic("time", every="1i", period="3i").agg(
        pl.col("value"),
        pl.col("value").min().alias("min_value"),
        pl.col("value").max().alias("max_value"),
        pl.col("value").sum().alias("sum_value"),
    ).to_dict(
        False
    ) == {
        "time": [0, 1, 2, 3],
        "value": [[0, 1, 2], [1, 2, 3], [2, 3], [3]],
        "min_value": [0, 1, 2, 3],
        "max_value": [2, 3, 3, 3],
        "sum_value": [3, 6, 5, 3],
    }


def test_rolling_cov_corr() -> None:
    df = pl.DataFrame({"x": [3, 3, 3, 5, 8], "y": [3, 4, 4, 4, 8]})

    res = df.select(
        [
            pl.rolling_cov("x", "y", window_size=3).alias("cov"),
            pl.rolling_corr("x", "y", window_size=3).alias("corr"),
        ]
    ).to_dict(False)
    assert res["cov"][2:] == pytest.approx([0.0, 0.0, 5.333333333333336])
    assert res["corr"][2:] == pytest.approx([nan, nan, 0.9176629354822473], nan_ok=True)
    assert res["cov"][:2] == [None] * 2
    assert res["corr"][:2] == [None] * 2


def test_rolling_window_size_9160() -> None:
    assert pl.Series([1, 5]).rolling_apply(
        lambda x: sum(x), window_size=2, min_periods=1
    ).to_list() == [1, 6]
    assert pl.Series([1]).rolling_apply(
        lambda x: sum(x), window_size=2, min_periods=1
    ).to_list() == [1]


def test_rolling_empty_window_9406() -> None:
    datecol = pl.Series(
        "d",
        [datetime(2019, 1, x) for x in [16, 17, 18, 22, 23]],
        dtype=pl.Datetime(time_unit="us", time_zone=None),
    )
    rawdata = pl.Series("x", [1.1, 1.2, 1.3, 1.15, 1.25], dtype=pl.Float64)
    rmin = pl.Series("x", [None, 1.1, 1.1, None, 1.15], dtype=pl.Float64)
    rmax = pl.Series("x", [None, 1.1, 1.2, None, 1.15], dtype=pl.Float64)
    df = pl.DataFrame([datecol, rawdata])

    assert_frame_equal(
        pl.DataFrame([datecol, rmax]),
        df.select(
            [
                pl.col("d"),
                pl.col("x").rolling_max(by="d", window_size="3d", closed="left"),
            ]
        ),
    )
    assert_frame_equal(
        pl.DataFrame([datecol, rmin]),
        df.select(
            [
                pl.col("d"),
                pl.col("x").rolling_min(by="d", window_size="3d", closed="left"),
            ]
        ),
    )
