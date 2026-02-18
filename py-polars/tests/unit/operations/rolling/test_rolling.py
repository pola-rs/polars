from __future__ import annotations

import random
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given
from numpy import nan

import polars as pl
from polars._utils.convert import parse_as_duration_string
from polars.exceptions import ComputeError, InvalidOperationError
from polars.meta.index_type import get_index_type
from polars.testing import assert_frame_equal, assert_series_equal
from polars.testing.parametric import column, dataframes, series
from polars.testing.parametric.strategies.dtype import _time_units
from tests.unit.conftest import INTEGER_DTYPES, NUMERIC_DTYPES, TEMPORAL_DTYPES

if TYPE_CHECKING:
    from collections.abc import Callable

    from hypothesis.strategies import SearchStrategy

    from polars._typing import (
        ClosedInterval,
        PolarsDataType,
        QuantileMethod,
        RankMethod,
        TimeUnit,
    )


@pytest.fixture
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
def test_rolling_kernels_and_rolling(
    example_df: pl.DataFrame, period: str | timedelta, closed: ClosedInterval
) -> None:
    out1 = example_df.set_sorted("dt").select(
        pl.col("dt"),
        # this differs from group_by aggregation because the empty window is
        # null here
        # where the sum aggregation of an empty set is 0
        pl.col("values")
        .rolling_sum_by("dt", period, closed=closed)
        .fill_null(0)
        .alias("sum"),
        pl.col("values").rolling_var_by("dt", period, closed=closed).alias("var"),
        pl.col("values").rolling_mean_by("dt", period, closed=closed).alias("mean"),
        pl.col("values").rolling_std_by("dt", period, closed=closed).alias("std"),
        pl.col("values")
        .rolling_quantile_by("dt", period, quantile=0.2, closed=closed)
        .alias("quantile"),
    )
    out2 = (
        example_df.set_sorted("dt")
        .rolling("dt", period=period, closed=closed)
        .agg(
            [
                pl.col("values").sum().alias("sum"),
                pl.col("values").var().alias("var"),
                pl.col("values").mean().alias("mean"),
                pl.col("values").std().alias("std"),
                pl.col("values").quantile(quantile=0.2).alias("quantile"),
            ]
        )
    )
    assert_frame_equal(out1, out2)


@pytest.mark.parametrize(
    "period",
    ["1d", "2d", "3d", timedelta(days=1), timedelta(days=2), timedelta(days=3)],
)
@pytest.mark.parametrize("closed", ["right", "both"])
def test_rolling_rank_kernels_and_rolling(
    example_df: pl.DataFrame, period: str | timedelta, closed: ClosedInterval
) -> None:
    out1 = example_df.set_sorted("dt").select(
        pl.col("dt"),
        pl.col("values").rolling_rank_by("dt", period, closed=closed).alias("rank"),
    )
    out2 = (
        example_df.set_sorted("dt")
        .rolling("dt", period=period, closed=closed)
        .agg([pl.col("values").rank().last().alias("rank")])
    )
    assert_frame_equal(out1, out2)


@pytest.mark.parametrize("closed", ["left", "none"])
def test_rolling_rank_needs_closed_right(
    example_df: pl.DataFrame, closed: ClosedInterval
) -> None:
    pat = r"`rolling_rank_by` window needs to be closed on the right side \(i.e., `closed` must be `right` or `both`\)"
    with pytest.raises(InvalidOperationError, match=pat):
        example_df.set_sorted("dt").select(
            pl.col("values").rolling_rank_by("dt", "2d", closed=closed).alias("rank"),
        )


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
            "ts": pl.datetime_range(
                datetime(2021, 1, 1), datetime(2021, 1, 4), "1d", eager=True
            ),
            "value": [1, 2, 3, 4],
        }
    )
    result = df.rolling("ts", period="2d", offset=offset, closed=closed).agg(
        pl.col("value")
    )
    expected = pl.DataFrame(
        {
            "ts": pl.datetime_range(
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


def test_rolling_kurtosis() -> None:
    s = pl.Series([1, 2, 3, 3, 2, 10, 8])
    assert s.rolling_kurtosis(window_size=4, bias=True).to_list() == pytest.approx(
        [
            None,
            None,
            None,
            -1.371900826446281,
            -1.9999999999999991,
            -0.7055324211778693,
            -1.7878967572797346,
        ]
    )
    assert s.rolling_kurtosis(
        window_size=4, bias=True, fisher=False
    ).to_list() == pytest.approx(
        [
            None,
            None,
            None,
            1.628099173553719,
            1.0000000000000009,
            2.2944675788221307,
            1.2121032427202654,
        ]
    )


@pytest.mark.parametrize("time_zone", [None, "America/Chicago"])
@pytest.mark.parametrize(
    ("rolling_fn", "expected_values", "expected_dtype"),
    [
        ("rolling_mean_by", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], pl.Float64),
        ("rolling_sum_by", [1, 2, 3, 4, 5, 6], pl.Int64),
        ("rolling_min_by", [1, 2, 3, 4, 5, 6], pl.Int64),
        ("rolling_max_by", [1, 2, 3, 4, 5, 6], pl.Int64),
        ("rolling_std_by", [None, None, None, None, None, None], pl.Float64),
        ("rolling_var_by", [None, None, None, None, None, None], pl.Float64),
        ("rolling_rank_by", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], pl.Float64),
    ],
)
def test_rolling_crossing_dst(
    time_zone: str | None,
    rolling_fn: str,
    expected_values: list[int | None | float],
    expected_dtype: PolarsDataType,
) -> None:
    ts = pl.datetime_range(
        datetime(2021, 11, 5), datetime(2021, 11, 10), "1d", time_zone="UTC", eager=True
    ).dt.replace_time_zone(time_zone)
    df = pl.DataFrame({"ts": ts, "value": [1, 2, 3, 4, 5, 6]})

    result = df.with_columns(
        getattr(pl.col("value"), rolling_fn)(by="ts", window_size="1d", closed="right")
    )

    expected = pl.DataFrame(
        {"ts": ts, "value": expected_values}, schema_overrides={"value": expected_dtype}
    )
    assert_frame_equal(result, expected)


def test_rolling_by_invalid() -> None:
    df = pl.DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6]}, schema_overrides={"a": pl.Int16}
    ).sort("a")
    msg = "unsupported data type: i16 for temporal/index column, expected UInt64, UInt32, Int64, Int32, Datetime, Date, Duration, or Time"
    with pytest.raises(InvalidOperationError, match=msg):
        df.select(pl.col("b").rolling_min_by("a", "2i"))
    df = pl.DataFrame({"a": [1, 2, 3], "b": [date(2020, 1, 1)] * 3}).sort("b")
    msg = "`window_size` duration may not be a parsed integer"
    with pytest.raises(InvalidOperationError, match=msg):
        df.select(pl.col("a").rolling_min_by("b", "2i"))


def test_rolling_infinity() -> None:
    s = pl.Series("col", ["-inf", "5", "5"]).cast(pl.Float64)
    s = s.rolling_mean(2)
    expected = pl.Series("col", [None, "-inf", "5"]).cast(pl.Float64)
    assert_series_equal(s, expected)


def test_rolling_by_non_temporal_window_size() -> None:
    df = pl.DataFrame(
        {"a": [4, 5, 6], "b": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]}
    ).sort("a", "b")
    msg = "`window_size` duration may not be a parsed integer"
    with pytest.raises(InvalidOperationError, match=msg):
        df.with_columns(pl.col("a").rolling_sum_by("b", "2i", closed="left"))


@pytest.mark.parametrize(
    "dtype",
    [
        pl.UInt8,
        pl.Int64,
        pl.Float32,
        pl.Float64,
        pl.Time,
        pl.Date,
        pl.Datetime("ms"),
        pl.Datetime("us"),
        pl.Datetime("ns"),
        pl.Datetime("ns", "Asia/Kathmandu"),
        pl.Duration("ms"),
        pl.Duration("us"),
        pl.Duration("ns"),
    ],
)
def test_rolling_extrema(dtype: PolarsDataType) -> None:
    # sorted data and nulls flags trigger different kernels
    df = (
        (
            pl.DataFrame(
                {
                    "col1": pl.int_range(0, 7, eager=True),
                    "col2": pl.int_range(0, 7, eager=True).reverse(),
                }
            )
        )
        .with_columns(
            pl.when(pl.int_range(0, pl.len(), eager=False) < 2)
            .then(None)
            .otherwise(pl.all())
            .name.keep()
            .name.suffix("_nulls")
        )
        .cast(dtype)
    )

    expected = {
        "col1": [None, None, 0, 1, 2, 3, 4],
        "col2": [None, None, 4, 3, 2, 1, 0],
        "col1_nulls": [None, None, None, None, 2, 3, 4],
        "col2_nulls": [None, None, None, None, 2, 1, 0],
    }
    result = df.select([pl.all().rolling_min(3)])
    assert result.to_dict(as_series=False) == {
        k: pl.Series(v, dtype=dtype).to_list() for k, v in expected.items()
    }

    expected = {
        "col1": [None, None, 2, 3, 4, 5, 6],
        "col2": [None, None, 6, 5, 4, 3, 2],
        "col1_nulls": [None, None, None, None, 4, 5, 6],
        "col2_nulls": [None, None, None, None, 4, 3, 2],
    }
    result = df.select([pl.all().rolling_max(3)])
    assert result.to_dict(as_series=False) == {
        k: pl.Series(v, dtype=dtype).to_list() for k, v in expected.items()
    }

    # shuffled data triggers other kernels
    df = df.select([pl.all().shuffle(seed=0)])
    expected = {
        "col1": [None, None, 0, 0, 4, 1, 1],
        "col2": [None, None, 1, 1, 0, 0, 0],
        "col1_nulls": [None, None, None, None, 4, None, None],
        "col2_nulls": [None, None, None, None, 0, None, None],
    }
    result = df.select([pl.all().rolling_min(3)])
    assert result.to_dict(as_series=False) == {
        k: pl.Series(v, dtype=dtype).to_list() for k, v in expected.items()
    }
    result = df.select([pl.all().rolling_max(3)])

    expected = {
        "col1": [None, None, 5, 5, 6, 6, 6],
        "col2": [None, None, 6, 6, 2, 5, 5],
        "col1_nulls": [None, None, None, None, 6, None, None],
        "col2_nulls": [None, None, None, None, 2, None, None],
    }
    assert result.to_dict(as_series=False) == {
        k: pl.Series(v, dtype=dtype).to_list() for k, v in expected.items()
    }


@pytest.mark.parametrize(
    "dtype",
    [
        pl.UInt8,
        pl.Int64,
        pl.Float32,
        pl.Float64,
        pl.Time,
        pl.Date,
        pl.Datetime("ms"),
        pl.Datetime("us"),
        pl.Datetime("ns"),
        pl.Datetime("ns", "Asia/Kathmandu"),
        pl.Duration("ms"),
        pl.Duration("us"),
        pl.Duration("ns"),
    ],
)
def test_rolling_group_by_extrema(dtype: PolarsDataType) -> None:
    # ensure we hit different branches so create

    df = pl.DataFrame(
        {
            "col1": pl.arange(0, 7, eager=True).reverse(),
        }
    ).with_columns(
        pl.col("col1").reverse().alias("index"),
        pl.col("col1").cast(dtype),
    )

    expected = {
        "col1_list": pl.Series(
            [
                [6],
                [6, 5],
                [6, 5, 4],
                [5, 4, 3],
                [4, 3, 2],
                [3, 2, 1],
                [2, 1, 0],
            ],
            dtype=pl.List(dtype),
        ).to_list(),
        "col1_min": pl.Series([6, 5, 4, 3, 2, 1, 0], dtype=dtype).to_list(),
        "col1_max": pl.Series([6, 6, 6, 5, 4, 3, 2], dtype=dtype).to_list(),
        "col1_first": pl.Series([6, 6, 6, 5, 4, 3, 2], dtype=dtype).to_list(),
        "col1_last": pl.Series([6, 5, 4, 3, 2, 1, 0], dtype=dtype).to_list(),
    }
    result = (
        df.rolling(
            index_column="index",
            period="3i",
        )
        .agg(
            [
                pl.col("col1").name.suffix("_list"),
                pl.col("col1").min().name.suffix("_min"),
                pl.col("col1").max().name.suffix("_max"),
                pl.col("col1").first().alias("col1_first"),
                pl.col("col1").last().alias("col1_last"),
            ]
        )
        .select(["col1_list", "col1_min", "col1_max", "col1_first", "col1_last"])
    )
    assert result.to_dict(as_series=False) == expected

    # ascending order

    df = pl.DataFrame(
        {
            "col1": pl.arange(0, 7, eager=True),
        }
    ).with_columns(
        pl.col("col1").alias("index"),
        pl.col("col1").cast(dtype),
    )

    result = (
        df.rolling(
            index_column="index",
            period="3i",
        )
        .agg(
            [
                pl.col("col1").name.suffix("_list"),
                pl.col("col1").min().name.suffix("_min"),
                pl.col("col1").max().name.suffix("_max"),
                pl.col("col1").first().alias("col1_first"),
                pl.col("col1").last().alias("col1_last"),
            ]
        )
        .select(["col1_list", "col1_min", "col1_max", "col1_first", "col1_last"])
    )
    expected = {
        "col1_list": pl.Series(
            [
                [0],
                [0, 1],
                [0, 1, 2],
                [1, 2, 3],
                [2, 3, 4],
                [3, 4, 5],
                [4, 5, 6],
            ],
            dtype=pl.List(dtype),
        ).to_list(),
        "col1_min": pl.Series([0, 0, 0, 1, 2, 3, 4], dtype=dtype).to_list(),
        "col1_max": pl.Series([0, 1, 2, 3, 4, 5, 6], dtype=dtype).to_list(),
        "col1_first": pl.Series([0, 0, 0, 1, 2, 3, 4], dtype=dtype).to_list(),
        "col1_last": pl.Series([0, 1, 2, 3, 4, 5, 6], dtype=dtype).to_list(),
    }
    assert result.to_dict(as_series=False) == expected

    # shuffled data.
    df = pl.DataFrame(
        {
            "col1": pl.arange(0, 7, eager=True).shuffle(1),
        }
    ).with_columns(
        pl.col("col1").cast(dtype),
        pl.col("col1").sort().alias("index"),
    )

    result = (
        df.rolling(
            index_column="index",
            period="3i",
        )
        .agg(
            [
                pl.col("col1").min().name.suffix("_min"),
                pl.col("col1").max().name.suffix("_max"),
                pl.col("col1").name.suffix("_list"),
            ]
        )
        .select(["col1_list", "col1_min", "col1_max"])
    )
    expected = {
        "col1_list": pl.Series(
            [
                [4],
                [4, 2],
                [4, 2, 5],
                [2, 5, 1],
                [5, 1, 6],
                [1, 6, 0],
                [6, 0, 3],
            ],
            dtype=pl.List(dtype),
        ).to_list(),
        "col1_min": pl.Series([4, 2, 2, 1, 1, 0, 0], dtype=dtype).to_list(),
        "col1_max": pl.Series([4, 4, 5, 5, 6, 6, 6], dtype=dtype).to_list(),
    }
    assert result.to_dict(as_series=False) == expected


def test_rolling_slice_pushdown() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "a", "b"], "c": [1, 3, 5]}).lazy()
    df = (
        df.sort("a")
        .rolling(
            "a",
            group_by="b",
            period="2i",
        )
        .agg([(pl.col("c") - pl.col("c").shift(fill_value=0)).sum().alias("c")])
    )
    assert df.head(2).collect().to_dict(as_series=False) == {
        "b": ["a", "a"],
        "a": [1, 2],
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
        df.rolling(index_column=pl.col("index").set_sorted(), period="3i").agg(
            [
                pl.col("val").diff(n=1).alias("val.diff"),
                (pl.col("val") - pl.col("val").shift(1)).alias("val - val.shift"),
            ]
        )
    ).to_dict(as_series=False) == {
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
    result = s.rolling_skew(3, min_samples=1).fill_nan(-1.0)
    expected = pl.Series(
        [
            None,
            -1.0,
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
    )
    assert_series_equal(result, expected, check_names=False)


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


def test_rolling_iter() -> None:
    df = pl.DataFrame(
        {
            "date": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 5)],
            "a": [1, 2, 2],
            "b": [4, 5, 6],
        }
    ).set_sorted("date")

    # Without 'by' argument
    result1 = [
        (name[0], data.shape)
        for name, data in df.rolling(index_column="date", period="2d")
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
        for name, data in df.rolling(index_column="date", period="2d", group_by="a")
    ]
    expected2 = [
        ((1, date(2020, 1, 1)), (1, 3)),
        ((2, date(2020, 1, 2)), (1, 3)),
        ((2, date(2020, 1, 5)), (1, 3)),
    ]
    assert result2 == expected2


def test_rolling_negative_period() -> None:
    df = pl.DataFrame({"ts": [datetime(2020, 1, 1)], "value": [1]}).with_columns(
        pl.col("ts").set_sorted()
    )
    with pytest.raises(
        ComputeError, match="rolling window period should be strictly positive"
    ):
        df.rolling("ts", period="-1d", offset="-1d").agg(pl.col("value"))
    with pytest.raises(
        ComputeError, match="rolling window period should be strictly positive"
    ):
        df.lazy().rolling("ts", period="-1d", offset="-1d").agg(
            pl.col("value")
        ).collect()
    with pytest.raises(
        InvalidOperationError, match="`window_size` must be strictly positive"
    ):
        df.select(
            pl.col("value").rolling_min_by("ts", window_size="-1d", closed="left")
        )
    with pytest.raises(
        InvalidOperationError, match="`window_size` must be strictly positive"
    ):
        df.lazy().select(
            pl.col("value").rolling_min_by("ts", window_size="-1d", closed="left")
        ).collect()


def test_rolling_skew_window_offset() -> None:
    assert (pl.arange(0, 20, eager=True) ** 2).rolling_skew(20)[
        -1
    ] == 0.6612545648596286


def test_rolling_cov_corr() -> None:
    df = pl.DataFrame({"x": [3, 3, 3, 5, 8], "y": [3, 4, 4, 4, 8]})

    res = df.select(
        pl.rolling_cov("x", "y", window_size=3).alias("cov"),
        pl.rolling_corr("x", "y", window_size=3).alias("corr"),
    ).to_dict(as_series=False)
    assert res["cov"][2:] == pytest.approx([0.0, 0.0, 5.333333333333336])
    assert res["corr"][2:] == pytest.approx([nan, 0.0, 0.9176629354822473], nan_ok=True)
    assert res["cov"][:2] == [None] * 2
    assert res["corr"][:2] == [None] * 2


def test_rolling_cov_corr_nulls() -> None:
    df1 = pl.DataFrame(
        {"a": [1.06, 1.07, 0.93, 0.78, 0.85], "lag_a": [1.0, 1.06, 1.07, 0.93, 0.78]}
    )
    df2 = pl.DataFrame(
        {
            "a": [1.0, 1.06, 1.07, 0.93, 0.78, 0.85],
            "lag_a": [None, 1.0, 1.06, 1.07, 0.93, 0.78],
        }
    )

    val_1 = df1.select(
        pl.rolling_corr("a", "lag_a", window_size=10, min_samples=5, ddof=1)
    )
    val_2 = df2.select(
        pl.rolling_corr("a", "lag_a", window_size=10, min_samples=5, ddof=1)
    )

    df1_expected = pl.DataFrame({"a": [None, None, None, None, 0.62204709]})
    df2_expected = pl.DataFrame({"a": [None, None, None, None, None, 0.62204709]})

    assert_frame_equal(val_1, df1_expected, abs_tol=0.0000001)
    assert_frame_equal(val_2, df2_expected, abs_tol=0.0000001)

    val_1 = df1.select(
        pl.rolling_cov("a", "lag_a", window_size=10, min_samples=5, ddof=1)
    )
    val_2 = df2.select(
        pl.rolling_cov("a", "lag_a", window_size=10, min_samples=5, ddof=1)
    )

    df1_expected = pl.DataFrame({"a": [None, None, None, None, 0.009445]})
    df2_expected = pl.DataFrame({"a": [None, None, None, None, None, 0.009445]})

    assert_frame_equal(val_1, df1_expected, abs_tol=0.0000001)
    assert_frame_equal(val_2, df2_expected, abs_tol=0.0000001)


@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_rolling_empty_window_9406(time_unit: TimeUnit) -> None:
    datecol = pl.Series(
        "d",
        [datetime(2019, 1, x) for x in [16, 17, 18, 22, 23]],
        dtype=pl.Datetime(time_unit=time_unit, time_zone=None),
    ).set_sorted()
    rawdata = pl.Series("x", [1.1, 1.2, 1.3, 1.15, 1.25], dtype=pl.Float64)
    rmin = pl.Series("x", [None, 1.1, 1.1, None, 1.15], dtype=pl.Float64)
    rmax = pl.Series("x", [None, 1.1, 1.2, None, 1.15], dtype=pl.Float64)
    df = pl.DataFrame([datecol, rawdata])

    assert_frame_equal(
        pl.DataFrame([datecol, rmax]),
        df.select(
            pl.col("d"),
            pl.col("x").rolling_max_by("d", window_size="3d", closed="left"),
        ),
    )
    assert_frame_equal(
        pl.DataFrame([datecol, rmin]),
        df.select(
            pl.col("d"),
            pl.col("x").rolling_min_by("d", window_size="3d", closed="left"),
        ),
    )


def test_rolling_weighted_quantile_10031() -> None:
    assert_series_equal(
        pl.Series([1, 2]).rolling_median(window_size=2, weights=[0, 1]),
        pl.Series([None, 2.0]),
    )

    assert_series_equal(
        pl.Series([1, 2, 3, 5]).rolling_quantile(0.7, "linear", 3, [0.1, 0.3, 0.6]),
        pl.Series([None, None, 2.55, 4.1]),
    )

    assert_series_equal(
        pl.Series([1, 2, 3, 5, 8]).rolling_quantile(
            0.7, "linear", 4, [0.1, 0.2, 0, 0.3]
        ),
        pl.Series([None, None, None, 3.5, 5.5]),
    )


def test_rolling_meta_eq_10101() -> None:
    assert pl.col("A").rolling_sum(10).meta.eq(pl.col("A").rolling_sum(10)) is True


def test_rolling_aggregations_unsorted_raise_10991() -> None:
    df = pl.DataFrame(
        {
            "dt": [datetime(2020, 1, 3), datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "val": [1, 2, 3],
        }
    )
    result = df.with_columns(roll=pl.col("val").rolling_sum_by("dt", "2d"))
    expected = pl.DataFrame(
        {
            "dt": [datetime(2020, 1, 3), datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "val": [1, 2, 3],
            "roll": [4, 2, 5],
        }
    )
    assert_frame_equal(result, expected)
    result = (
        df.with_row_index()
        .sort("dt")
        .with_columns(roll=pl.col("val").rolling_sum_by("dt", "2d"))
        .sort("index")
        .drop("index")
    )
    assert_frame_equal(result, expected)


def test_rolling_aggregations_with_over_11225() -> None:
    start = datetime(2001, 1, 1)

    df_temporal = pl.DataFrame(
        {
            "date": [start + timedelta(days=k) for k in range(5)],
            "group": ["A"] * 2 + ["B"] * 3,
        }
    ).with_row_index()

    df_temporal = df_temporal.sort("group", "date")

    result = df_temporal.with_columns(
        rolling_row_mean=pl.col("index")
        .rolling_mean_by(
            by="date",
            window_size="2d",
            closed="left",
        )
        .over("group")
    )
    expected = pl.DataFrame(
        {
            "index": [0, 1, 2, 3, 4],
            "date": pl.datetime_range(date(2001, 1, 1), date(2001, 1, 5), eager=True),
            "group": ["A", "A", "B", "B", "B"],
            "rolling_row_mean": [None, 0.0, None, 2.0, 2.5],
        },
        schema_overrides={"index": pl.get_index_type()},
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test_rolling_ints(dtype: PolarsDataType) -> None:
    s = pl.Series("a", [1, 2, 3, 2, 1], dtype=dtype)
    assert_series_equal(
        s.rolling_min(2), pl.Series("a", [None, 1, 2, 2, 1], dtype=dtype)
    )
    assert_series_equal(
        s.rolling_max(2), pl.Series("a", [None, 2, 3, 3, 2], dtype=dtype)
    )
    assert_series_equal(
        s.rolling_sum(2),
        pl.Series(
            "a",
            [None, 3, 5, 5, 3],
            dtype=(
                pl.Int64 if dtype in [pl.Int8, pl.UInt8, pl.Int16, pl.UInt16] else dtype
            ),
        ),
    )
    assert_series_equal(s.rolling_mean(2), pl.Series("a", [None, 1.5, 2.5, 2.5, 1.5]))

    assert s.rolling_std(2).to_list()[1] == pytest.approx(0.7071067811865476)
    assert s.rolling_var(2).to_list()[1] == pytest.approx(0.5)
    assert s.rolling_std(2, ddof=0).to_list()[1] == pytest.approx(0.5)
    assert s.rolling_var(2, ddof=0).to_list()[1] == pytest.approx(0.25)

    assert_series_equal(
        s.rolling_median(4), pl.Series("a", [None, None, None, 2, 2], dtype=pl.Float64)
    )
    assert_series_equal(
        s.rolling_quantile(0, "nearest", 3),
        pl.Series("a", [None, None, 1, 2, 1], dtype=pl.Float64),
    )
    assert_series_equal(
        s.rolling_quantile(0, "lower", 3),
        pl.Series("a", [None, None, 1, 2, 1], dtype=pl.Float64),
    )
    assert_series_equal(
        s.rolling_quantile(0, "higher", 3),
        pl.Series("a", [None, None, 1, 2, 1], dtype=pl.Float64),
    )
    assert s.rolling_skew(4).null_count() == 3


def test_rolling_floats() -> None:
    # 3099
    # test if we maintain proper dtype
    for dt in [pl.Float32, pl.Float64]:
        result = pl.Series([1, 2, 3], dtype=dt).rolling_min(2, weights=[0.1, 0.2])
        expected = pl.Series([None, 0.1, 0.2], dtype=dt)
        assert_series_equal(result, expected)

    df = pl.DataFrame({"val": [1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0]})

    for e in [
        pl.col("val").rolling_min(window_size=3),
        pl.col("val").rolling_max(window_size=3),
    ]:
        out = df.with_columns(e).to_series()
        assert out.null_count() == 2
        assert np.isnan(out.to_numpy()).sum() == 5

    expected_values = [None, None, 2.0, 3.0, 5.0, 6.0, 6.0]
    assert (
        df.with_columns(pl.col("val").rolling_median(window_size=3))
        .to_series()
        .to_list()
        == expected_values
    )
    assert (
        df.with_columns(pl.col("val").rolling_quantile(0.5, window_size=3))
        .to_series()
        .to_list()
        == expected_values
    )

    nan = float("nan")
    s = pl.Series("a", [11.0, 2.0, 9.0, nan, 8.0])
    assert_series_equal(
        s.rolling_sum(3),
        pl.Series("a", [None, None, 22.0, nan, nan]),
    )


def test_rolling_std_nulls_min_samples_1_20076() -> None:
    result = pl.Series([1, 2, None, 4]).rolling_std(3, min_samples=1)
    expected = pl.Series(
        [None, 0.7071067811865476, 0.7071067811865476, 1.4142135623730951]
    )
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("bools", "window", "expected"),
    [
        (
            [[True, False, True]],
            2,
            [[None, 1, 1]],
        ),
        (
            [[True, False, True, True, False, False, False, True, True]],
            4,
            [[None, None, None, 3, 2, 2, 1, 1, 2]],
        ),
    ],
)
def test_rolling_eval_boolean_list(
    bools: list[list[bool]], window: int, expected: list[list[int]]
) -> None:
    for accessor, dtype in (
        ("list", pl.List(pl.Boolean)),
        ("arr", pl.Array(pl.Boolean, shape=len(bools[0]))),
    ):
        s = pl.Series(name="bools", values=bools, dtype=dtype)
        res = getattr(s, accessor).eval(pl.element().rolling_sum(window)).to_list()
        assert res == expected


def test_rolling_by_date() -> None:
    df = pl.DataFrame(
        {
            "dt": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
            "val": [1, 2, 3],
        }
    ).sort("dt")

    result = df.with_columns(roll=pl.col("val").rolling_sum_by("dt", "2d"))
    expected = df.with_columns(roll=pl.Series([1, 3, 5]))
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", [pl.Int64, pl.Int32, pl.UInt64, pl.UInt32])
def test_rolling_by_integer(dtype: PolarsDataType) -> None:
    df = (
        pl.DataFrame({"val": [1, 2, 3]})
        .with_row_index()
        .with_columns(pl.col("index").cast(dtype))
    )
    result = df.with_columns(roll=pl.col("val").rolling_sum_by("index", "2i"))
    expected = df.with_columns(roll=pl.Series([1, 3, 5]))
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test_rolling_sum_by_integer(dtype: PolarsDataType) -> None:
    lf = (
        pl.LazyFrame({"a": [1, 2, 3]}, schema={"a": dtype})
        .with_row_index()
        .select(pl.col("a").rolling_sum_by("index", "2i"))
    )
    result = lf.collect()
    expected_dtype = (
        pl.Int64 if dtype in [pl.Int8, pl.UInt8, pl.Int16, pl.UInt16] else dtype
    )
    expected = pl.DataFrame({"a": [1, 3, 5]}, schema={"a": expected_dtype})
    assert_frame_equal(result, expected)
    assert lf.collect_schema() == expected.schema


def test_rolling_nanoseconds_11003() -> None:
    df = pl.DataFrame(
        {
            "dt": [
                "2020-01-01T00:00:00.000000000",
                "2020-01-01T00:00:00.000000100",
                "2020-01-01T00:00:00.000000200",
            ],
            "val": [1, 2, 3],
        }
    )
    df = df.with_columns(pl.col("dt").str.to_datetime(time_unit="ns")).set_sorted("dt")
    result = df.with_columns(pl.col("val").rolling_sum_by("dt", "500ns"))
    expected = df.with_columns(val=pl.Series([1, 3, 6]))
    assert_frame_equal(result, expected)


def test_rolling_by_1mo_saturating_12216() -> None:
    df = pl.DataFrame(
        {
            "date": [
                date(2020, 6, 29),
                date(2020, 6, 30),
                date(2020, 7, 30),
                date(2020, 7, 31),
                date(2020, 8, 1),
            ],
            "val": [1, 2, 3, 4, 5],
        }
    ).set_sorted("date")
    result = df.rolling(index_column="date", period="1mo").agg(vals=pl.col("val"))
    expected = pl.DataFrame(
        {
            "date": [
                date(2020, 6, 29),
                date(2020, 6, 30),
                date(2020, 7, 30),
                date(2020, 7, 31),
                date(2020, 8, 1),
            ],
            "vals": [[1], [1, 2], [3], [3, 4], [3, 4, 5]],
        }
    )
    assert_frame_equal(result, expected)

    # check with `closed='both'` against DuckDB output
    result = df.rolling(index_column="date", period="1mo", closed="both").agg(
        vals=pl.col("val")
    )
    expected = pl.DataFrame(
        {
            "date": [
                date(2020, 6, 29),
                date(2020, 6, 30),
                date(2020, 7, 30),
                date(2020, 7, 31),
                date(2020, 8, 1),
            ],
            "vals": [[1], [1, 2], [2, 3], [2, 3, 4], [3, 4, 5]],
        }
    )
    assert_frame_equal(result, expected)


def test_index_expr_with_literal() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}).sort("a")
    out = df.rolling(index_column=(5 * pl.col("a")).set_sorted(), period="2i").agg(
        pl.col("b")
    )
    expected = pl.DataFrame({"literal": [5, 10, 15], "b": [["a"], ["b"], ["c"]]})
    assert_frame_equal(out, expected)


def test_index_expr_output_name_12244() -> None:
    df = pl.DataFrame({"A": [1, 2, 3]})

    out = df.rolling(pl.int_range(0, pl.len()), period="2i").agg("A")
    assert out.to_dict(as_series=False) == {
        "literal": [0, 1, 2],
        "A": [[1], [1, 2], [2, 3]],
    }


def test_rolling_median() -> None:
    for n in range(10, 25):
        array = np.random.randint(0, 20, n)
        for k in [3, 5, 7]:
            a = pl.Series(array)
            assert_series_equal(
                a.rolling_median(k), pl.from_pandas(a.to_pandas().rolling(k).median())
            )


@pytest.mark.slow
def test_rolling_median_2() -> None:
    np.random.seed(12)
    n = 1000
    df = pl.DataFrame({"x": np.random.normal(0, 1, n)})
    # this can differ because simd sizes and non-associativity of floats.
    assert df.select(
        pl.col("x").rolling_median(window_size=10).sum()
    ).item() == pytest.approx(5.139429061527812)
    assert df.select(
        pl.col("x").rolling_median(window_size=100).sum()
    ).item() == pytest.approx(26.60506093611384)


@pytest.mark.parametrize(
    ("dates", "closed", "expected"),
    [
        (
            [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
            "right",
            [None, 3, 5],
        ),
        (
            [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
            "left",
            [None, None, 3],
        ),
        (
            [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
            "both",
            [None, 3, 6],
        ),
        (
            [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
            "none",
            [None, None, None],
        ),
        (
            [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 4)],
            "right",
            [None, 3, None],
        ),
        (
            [date(2020, 1, 1), date(2020, 1, 3), date(2020, 1, 4)],
            "right",
            [None, None, 5],
        ),
        (
            [date(2020, 1, 1), date(2020, 1, 3), date(2020, 1, 5)],
            "right",
            [None, None, None],
        ),
    ],
)
def test_rolling_min_samples(
    dates: list[date], closed: ClosedInterval, expected: list[int]
) -> None:
    df = pl.DataFrame({"date": dates, "value": [1, 2, 3]}).sort("date")
    result = df.select(
        pl.col("value").rolling_sum_by(
            "date", window_size="2d", min_samples=2, closed=closed
        )
    )["value"]
    assert_series_equal(result, pl.Series("value", expected, pl.Int64))

    # Starting with unsorted data
    result = (
        df.sort("date", descending=True)
        .with_columns(
            pl.col("value").rolling_sum_by(
                "date", window_size="2d", min_samples=2, closed=closed
            )
        )
        .sort("date")["value"]
    )
    assert_series_equal(result, pl.Series("value", expected, pl.Int64))


def test_rolling_returns_scalar_15656() -> None:
    df = pl.DataFrame(
        {
            "a": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
            "b": [4, 5, 6],
            "c": [1, 2, 3],
        }
    )
    result = df.group_by("c").agg(pl.col("b").rolling_mean_by("a", "2d")).sort("c")
    expected = pl.DataFrame({"c": [1, 2, 3], "b": [[4.0], [5.0], [6.0]]})
    assert_frame_equal(result, expected)


def test_rolling_invalid() -> None:
    df = pl.DataFrame(
        {
            "values": [1, 4],
            "times": [datetime(2020, 1, 3), datetime(2020, 1, 1)],
        },
    )
    with pytest.raises(
        InvalidOperationError, match="duration may not be a parsed integer"
    ):
        (
            df.sort("times")
            .rolling("times", period="3000i")
            .agg(pl.col("values").sum().alias("sum"))
        )
    with pytest.raises(
        InvalidOperationError, match="duration must be a parsed integer"
    ):
        (
            df.with_row_index()
            .rolling("index", period="3000d")
            .agg(pl.col("values").sum().alias("sum"))
        )


def test_by_different_length() -> None:
    df = pl.DataFrame({"b": [1]})
    with pytest.raises(InvalidOperationError, match="must be the same length"):
        df.select(
            pl.col("b").rolling_max_by(pl.Series([datetime(2020, 1, 1)] * 2), "1d")
        )


def test_incorrect_nulls_16246() -> None:
    df = pl.concat(
        [
            pl.DataFrame({"a": [datetime(2020, 1, 1)], "b": [1]}),
            pl.DataFrame({"a": [datetime(2021, 1, 1)], "b": [1]}),
        ],
        rechunk=False,
    )
    result = df.select(pl.col("b").rolling_max_by("a", "1d"))
    expected = pl.DataFrame({"b": [1, 1]})
    assert_frame_equal(result, expected)


def test_rolling_with_dst() -> None:
    df = pl.DataFrame(
        {"a": [datetime(2020, 10, 26, 1), datetime(2020, 10, 26)], "b": [1, 2]}
    ).with_columns(pl.col("a").dt.replace_time_zone("Europe/London"))
    result = df.select(pl.col("b").rolling_sum_by("a", "1d"))
    expected = pl.DataFrame({"b": [3, 2]})
    assert_frame_equal(result, expected)

    result = df.sort("a").select(pl.col("b").rolling_sum_by("a", "1d"))
    expected = pl.DataFrame({"b": [2, 3]})
    assert_frame_equal(result, expected)


def interval_defs() -> SearchStrategy[ClosedInterval]:
    closed: list[ClosedInterval] = ["left", "right", "both", "none"]
    return st.sampled_from(closed)


@given(
    period=st.timedeltas(
        min_value=timedelta(microseconds=0), max_value=timedelta(days=1000)
    ).map(parse_as_duration_string),
    offset=st.timedeltas(
        min_value=timedelta(days=-1000), max_value=timedelta(days=1000)
    ).map(parse_as_duration_string),
    closed=interval_defs(),
    data=st.data(),
    time_unit=_time_units(),
)
def test_rolling_parametric(
    period: str,
    offset: str,
    closed: ClosedInterval,
    data: st.DataObject,
    time_unit: TimeUnit,
) -> None:
    assume(period != "")
    dataframe = data.draw(
        dataframes(
            [
                column(
                    "ts",
                    strategy=st.datetimes(
                        min_value=datetime(2000, 1, 1),
                        max_value=datetime(2001, 1, 1),
                    ),
                    dtype=pl.Datetime(time_unit),
                ),
                column(
                    "value",
                    strategy=st.integers(min_value=-100, max_value=100),
                    dtype=pl.Int64,
                ),
            ],
            min_size=1,
        )
    )
    df = dataframe.sort("ts")
    result = df.rolling("ts", period=period, offset=offset, closed=closed).agg(
        pl.col("value")
    )

    expected_dict: dict[str, list[object]] = {"ts": [], "value": []}
    for ts, _ in df.iter_rows():
        window = df.filter(
            pl.col("ts").is_between(
                pl.lit(ts, dtype=pl.Datetime(time_unit)).dt.offset_by(offset),
                pl.lit(ts, dtype=pl.Datetime(time_unit))
                .dt.offset_by(offset)
                .dt.offset_by(period),
                closed=closed,
            )
        )
        value = window["value"].to_list()
        expected_dict["ts"].append(ts)
        expected_dict["value"].append(value)
    expected = pl.DataFrame(expected_dict).select(
        pl.col("ts").cast(pl.Datetime(time_unit)),
        pl.col("value").cast(pl.List(pl.Int64)),
    )
    assert_frame_equal(result, expected)


@given(
    window_size=st.timedeltas(
        min_value=timedelta(microseconds=0), max_value=timedelta(days=2)
    ).map(parse_as_duration_string),
    closed=interval_defs(),
    data=st.data(),
    time_unit=_time_units(),
    aggregation=st.sampled_from(
        [
            "min",
            "max",
            "mean",
            "sum",
            "std",
            "var",
            "median",
        ]
    ),
)
def test_rolling_aggs(
    window_size: str,
    closed: ClosedInterval,
    data: st.DataObject,
    time_unit: TimeUnit,
    aggregation: str,
) -> None:
    assume(window_size != "")

    # Testing logic can be faulty when window is more precise than time unit
    # https://github.com/pola-rs/polars/issues/11754
    assume(not (time_unit == "ms" and "us" in window_size))

    dataframe = data.draw(
        dataframes(
            [
                column(
                    "ts",
                    strategy=st.datetimes(
                        min_value=datetime(2000, 1, 1),
                        max_value=datetime(2001, 1, 1),
                    ),
                    dtype=pl.Datetime(time_unit),
                ),
                column(
                    "value",
                    strategy=st.integers(min_value=-100, max_value=100),
                    dtype=pl.Int64,
                ),
            ],
        )
    )
    df = dataframe.sort("ts")
    func = f"rolling_{aggregation}_by"
    result = df.with_columns(
        getattr(pl.col("value"), func)("ts", window_size=window_size, closed=closed)
    )
    result_from_unsorted = dataframe.with_columns(
        getattr(pl.col("value"), func)("ts", window_size=window_size, closed=closed)
    ).sort("ts")

    expected_dict: dict[str, list[object]] = {"ts": [], "value": []}
    for ts, _ in df.iter_rows():
        window = df.filter(
            pl.col("ts").is_between(
                pl.lit(ts, dtype=pl.Datetime(time_unit)).dt.offset_by(
                    f"-{window_size}"
                ),
                pl.lit(ts, dtype=pl.Datetime(time_unit)),
                closed=closed,
            )
        )
        expected_dict["ts"].append(ts)
        if window.is_empty():
            expected_dict["value"].append(None)
        else:
            value = getattr(window["value"], aggregation)()
            expected_dict["value"].append(value)
    expected = pl.DataFrame(expected_dict).select(
        pl.col("ts").cast(pl.Datetime(time_unit)),
        pl.col("value").cast(result["value"].dtype),
    )
    assert_frame_equal(result, expected)
    assert_frame_equal(result_from_unsorted, expected)


def test_window_size_validation() -> None:
    df = pl.DataFrame({"x": [1.0]})

    with pytest.raises(OverflowError, match=r"can't convert negative int to unsigned"):
        df.with_columns(trailing_min=pl.col("x").rolling_min(window_size=-3))


def test_rolling_empty_21032() -> None:
    df = pl.DataFrame(schema={"a": pl.Datetime("ms"), "b": pl.Int64()})

    result = df.rolling(index_column="a", period=timedelta(days=2)).agg(
        pl.col("b").sum()
    )
    assert_frame_equal(result, df)

    result = df.rolling(
        index_column="a", period=timedelta(days=2), offset=timedelta(days=3)
    ).agg(pl.col("b").sum())
    assert_frame_equal(result, df)


def test_rolling_offset_agg_15122() -> None:
    df = pl.DataFrame({"a": [1, 1, 1, 2, 2, 2], "b": [1, 2, 3, 1, 2, 3]})

    result = df.rolling(index_column="b", period="1i", offset="0i", group_by="a").agg(
        window=pl.col("b")
    )
    expected = df.with_columns(window=pl.Series([[2], [3], [], [2], [3], []]))
    assert_frame_equal(result, expected)

    result = df.rolling(index_column="b", period="1i", offset="1i", group_by="a").agg(
        window=pl.col("b")
    )
    expected = df.with_columns(window=pl.Series([[3], [], [], [3], [], []]))
    assert_frame_equal(result, expected)


def test_rolling_sum_stability_11146() -> None:
    data_frame = pl.DataFrame(
        {
            "value": [
                0.0,
                290.57,
                107.0,
                172.0,
                124.25,
                304.0,
                379.5,
                347.35,
                1516.41,
                386.12,
                226.5,
                294.62,
                125.5,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        }
    )
    assert (
        data_frame.with_columns(
            pl.col("value").rolling_mean(window_size=8, min_samples=1).alias("test_col")
        )["test_col"][-1]
        == 0.0
    )


def test_rolling() -> None:
    df = pl.DataFrame(
        {
            "n": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10],
            "col1": ["A", "B"] * 11,
        }
    )

    assert df.rolling("n", period="1i", group_by="col1").agg().to_dict(
        as_series=False
    ) == {
        "col1": [
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
            "B",
            "B",
            "B",
            "B",
            "B",
            "B",
            "B",
            "B",
        ],
        "n": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }


@pytest.mark.parametrize(
    "method",
    ["nearest", "higher", "lower", "midpoint", "linear", "equiprobable"],
)
def test_rolling_quantile_with_nulls_22781(method: QuantileMethod) -> None:
    lf = pl.LazyFrame(
        {
            "index": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            "a": [None, None, 1.0, None, None, 1.0, 1.0, None, None],
        }
    )
    out = (
        lf.rolling("index", period="2i")
        .agg(pl.col("a").quantile(0.5, interpolation=method))
        .collect()
    )
    expected = pl.Series("a", [None, None, 1.0, 1.0, None, 1.0, 1.0, 1.0, None])
    assert_series_equal(out["a"], expected)


def test_rolling_quantile_nearest_23392() -> None:
    base = range(11)
    s = pl.Series(base)

    shuffle_base = list(base)
    random.shuffle(shuffle_base)
    s_shuffled = pl.Series(shuffle_base)

    for q in np.arange(0, 1.0, 0.02, dtype=float):
        out = s.rolling_quantile(q, interpolation="nearest", window_size=11)

        # explicit:
        expected = pl.Series([None] * 10 + [float(round(q * 10.0))])
        assert_series_equal(out, expected)

        # equivalence:
        equiv = s.quantile(q, interpolation="nearest")
        assert out.last() == equiv

        # shuffled:
        out = s_shuffled.rolling_quantile(q, interpolation="nearest", window_size=11)
        assert_series_equal(out, expected)


def test_rolling_quantile_temporals() -> None:
    tz = ZoneInfo("Asia/Tokyo")
    dt = pl.Datetime("ms", "Asia/Tokyo")
    # We use ms to verify that the correct time unit is propagating.
    lf = pl.LazyFrame(
        {
            "date": [date(2025, 1, x) for x in range(1, 6)],
            "datetime": [datetime(2025, 1, x) for x in range(1, 6)],
            "datetime_tu_tz": pl.Series(
                [datetime(2025, 1, x, tzinfo=tz) for x in range(1, 6)], dtype=dt
            ),
            "duration": pl.Series(
                [timedelta(hours=x) for x in range(1, 6)], dtype=pl.Duration("ms")
            ),
            "time": [time(hour=x) for x in range(1, 6)],
        }
    )
    result = lf.select(
        rolling_date=pl.col("date").rolling_quantile(
            quantile=0.5, window_size=4, interpolation="linear"
        ),
        rolling_datetime=pl.col("datetime").rolling_quantile(
            quantile=0.5, window_size=4, interpolation="linear"
        ),
        rolling_datetime_tu_tz=pl.col("datetime_tu_tz").rolling_quantile(
            quantile=0.5, window_size=4, interpolation="linear"
        ),
        rolling_duration=pl.col("duration").rolling_quantile(
            quantile=0.5, window_size=4, interpolation="linear"
        ),
        rolling_time=pl.col("time").rolling_quantile(
            quantile=0.5, window_size=4, interpolation="linear"
        ),
    )
    expected = pl.DataFrame(
        {
            "rolling_date": pl.Series(
                [None, None, None, datetime(2025, 1, 2, 12), datetime(2025, 1, 3, 12)],
                dtype=pl.Datetime,
            ),
            "rolling_datetime": pl.Series(
                [None, None, None, datetime(2025, 1, 2, 12), datetime(2025, 1, 3, 12)]
            ),
            "rolling_datetime_tu_tz": pl.Series(
                [
                    None,
                    None,
                    None,
                    datetime(2025, 1, 2, 12, tzinfo=tz),
                    datetime(2025, 1, 3, 12, tzinfo=tz),
                ],
                dtype=dt,
            ),
            "rolling_duration": pl.Series(
                [None, None, None, timedelta(hours=2.5), timedelta(hours=3.5)],
                dtype=pl.Duration("ms"),
            ),
            "rolling_time": [
                None,
                None,
                None,
                time(hour=2, minute=30),
                time(hour=3, minute=30),
            ],
        }
    )
    assert result.collect_schema() == pl.Schema(
        {  # type: ignore[arg-type]
            "rolling_date": pl.Datetime("us"),
            "rolling_datetime": pl.Datetime("us"),
            "rolling_datetime_tu_tz": dt,
            "rolling_duration": pl.Duration("ms"),
            "rolling_time": pl.Time,
        }
    )
    assert_frame_equal(result.collect(), expected)


def test_rolling_agg_quantile_temporal() -> None:
    tz = ZoneInfo("Asia/Tokyo")
    dt = pl.Datetime("ms", "Asia/Tokyo")
    # We use ms to verify that the correct time unit is propagating.
    lf = pl.LazyFrame(
        {
            "index": [1, 2, 3, 4, 5],
            "int": [1, 2, 3, 4, 5],
            "date": [date(2025, 1, x) for x in range(1, 6)],
            "datetime": [datetime(2025, 1, x) for x in range(1, 6)],
            "datetime_tu_tz": pl.Series(
                [datetime(2025, 1, x, tzinfo=tz) for x in range(1, 6)], dtype=dt
            ),
            "duration": pl.Series(
                [timedelta(hours=x) for x in range(1, 6)], dtype=pl.Duration("ms")
            ),
            "time": [time(hour=x) for x in range(1, 6)],
        }
    )

    # Using rolling.agg()
    result1 = lf.rolling("index", period="4i").agg(
        rolling_int=pl.col("int").quantile(0.5, "linear"),
        rolling_date=pl.col("date").quantile(0.5, "linear"),
        rolling_datetime=pl.col("datetime").quantile(0.5, "linear"),
        rolling_datetime_tu_tz=pl.col("datetime_tu_tz").quantile(0.5, "linear"),
        rolling_duration=pl.col("duration").quantile(0.5, "linear"),
        rolling_time=pl.col("time").quantile(0.5, "linear"),
    )
    # Using rolling_quantile_by()
    result2 = lf.select(
        "index",
        rolling_int=pl.col("int").rolling_quantile_by(
            "index", window_size="4i", quantile=0.5, interpolation="linear"
        ),
        rolling_date=pl.col("date").rolling_quantile_by(
            "index", window_size="4i", quantile=0.5, interpolation="linear"
        ),
        rolling_datetime=pl.col("datetime").rolling_quantile_by(
            "index", window_size="4i", quantile=0.5, interpolation="linear"
        ),
        rolling_datetime_tu_tz=pl.col("datetime_tu_tz").rolling_quantile_by(
            "index", window_size="4i", quantile=0.5, interpolation="linear"
        ),
        rolling_duration=pl.col("duration").rolling_quantile_by(
            "index", window_size="4i", quantile=0.5, interpolation="linear"
        ),
        rolling_time=pl.col("time").rolling_quantile_by(
            "index", window_size="4i", quantile=0.5, interpolation="linear"
        ),
    )
    expected = pl.DataFrame(
        {
            "index": [1, 2, 3, 4, 5],
            "rolling_int": [1.0, 1.5, 2.0, 2.5, 3.5],
            "rolling_date": pl.Series(
                [
                    datetime(2025, 1, 1),
                    datetime(2025, 1, 1, 12),
                    datetime(2025, 1, 2),
                    datetime(2025, 1, 2, 12),
                    datetime(2025, 1, 3, 12),
                ]
            ),
            "rolling_datetime": pl.Series(
                [
                    datetime(2025, 1, 1),
                    datetime(2025, 1, 1, 12),
                    datetime(2025, 1, 2),
                    datetime(2025, 1, 2, 12),
                    datetime(2025, 1, 3, 12),
                ]
            ),
            "rolling_datetime_tu_tz": pl.Series(
                [
                    datetime(2025, 1, 1, tzinfo=tz),
                    datetime(2025, 1, 1, 12, tzinfo=tz),
                    datetime(2025, 1, 2, tzinfo=tz),
                    datetime(2025, 1, 2, 12, tzinfo=tz),
                    datetime(2025, 1, 3, 12, tzinfo=tz),
                ],
                dtype=dt,
            ),
            "rolling_duration": pl.Series(
                [
                    timedelta(hours=1),
                    timedelta(hours=1.5),
                    timedelta(hours=2),
                    timedelta(hours=2.5),
                    timedelta(hours=3.5),
                ],
                dtype=pl.Duration("ms"),
            ),
            "rolling_time": [
                time(hour=1),
                time(hour=1, minute=30),
                time(hour=2),
                time(hour=2, minute=30),
                time(hour=3, minute=30),
            ],
        }
    )
    expected_schema = pl.Schema(
        {  # type: ignore[arg-type]
            "index": pl.Int64,
            "rolling_int": pl.Float64,
            "rolling_date": pl.Datetime("us"),
            "rolling_datetime": pl.Datetime("us"),
            "rolling_datetime_tu_tz": dt,
            "rolling_duration": pl.Duration("ms"),
            "rolling_time": pl.Time,
        }
    )
    assert result1.collect_schema() == expected_schema
    assert result2.collect_schema() == expected_schema
    assert_frame_equal(result1.collect(), expected)
    assert_frame_equal(result2.collect(), expected)


def test_rolling_quantile_nearest_kernel_23392() -> None:
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
    # values (period="3d", quantile=0.7) are chosen to trigger index rounding
    out = (
        df.set_sorted("dt")
        .rolling("dt", period="3d", closed="both")
        .agg([pl.col("values").quantile(quantile=0.7).alias("quantile")])
        .select("quantile")
    )
    expected = pl.DataFrame({"quantile": [0.0, 1.0, 1.0, 2.0, 3.0]})
    assert_frame_equal(out, expected)


def test_rolling_quantile_nearest_with_nulls_23932() -> None:
    lf = pl.LazyFrame(
        {
            "index": [0, 1, 2, 3, 4, 5, 6],
            "a": [None, None, 1.0, 2.0, 3.0, None, None],
        }
    )
    # values (period="3i", quantile=0.7) are chosen to trigger index rounding
    out = (
        lf.rolling("index", period="3i")
        .agg(pl.col("a").quantile(0.7, interpolation="nearest"))
        .collect()
    )
    expected = pl.Series("a", [None, None, 1.0, 2.0, 2.0, 3.0, 3.0])
    assert_series_equal(out["a"], expected)


def test_wtd_min_periods_less_window() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]}).with_columns(
        pl.col("a")
        .rolling_mean(
            window_size=3, weights=[0.25, 0.5, 0.25], min_samples=2, center=True
        )
        .alias("kernel_mean")
    )

    expected = pl.DataFrame(
        {"a": [1, 2, 3, 4, 5], "kernel_mean": [1.333333, 2, 3, 4, 4.666667]}
    )

    assert_frame_equal(df, expected)

    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]}).with_columns(
        pl.col("a")
        .rolling_sum(
            window_size=3, weights=[0.25, 0.5, 0.25], min_samples=2, center=True
        )
        .alias("kernel_sum")
    )
    expected = pl.DataFrame(
        {"a": [1, 2, 3, 4, 5], "kernel_sum": [1.0, 2.0, 3.0, 4.0, 3.5]}
    )

    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]}).with_columns(
        pl.col("a")
        .rolling_mean(
            window_size=3, weights=[0.2, 0.3, 0.5], min_samples=2, center=False
        )
        .alias("kernel_mean")
    )

    expected = pl.DataFrame(
        {"a": [1, 2, 3, 4, 5], "kernel_mean": [None, 1.625, 2.3, 3.3, 4.3]}
    )

    assert_frame_equal(df, expected)

    df = pl.DataFrame({"a": [1, 2]}).with_columns(
        pl.col("a")
        .rolling_mean(
            window_size=3, weights=[0.25, 0.5, 0.25], min_samples=2, center=True
        )
        .alias("kernel_mean")
    )

    # Handle edge case where the window size is larger than the number of elements
    expected = pl.DataFrame({"a": [1, 2], "kernel_mean": [1.333333, 1.666667]})
    assert_frame_equal(df, expected)

    df = pl.DataFrame({"a": [1, 2]}).with_columns(
        pl.col("a")
        .rolling_mean(
            window_size=3, weights=[0.25, 0.25, 0.5], min_samples=1, center=False
        )
        .alias("kernel_mean")
    )

    expected = pl.DataFrame({"a": [1, 2], "kernel_mean": [1.0, 2 * 2 / 3 + 1 * 1 / 3]})

    df = pl.DataFrame({"a": [1]}).with_columns(
        pl.col("a")
        .rolling_sum(
            6, center=True, min_samples=0, weights=[1, 10, 100, 1000, 10_000, 100_000]
        )
        .alias("kernel_sum")
    )
    expected = pl.DataFrame({"a": [1], "kernel_sum": [1000.0]})
    assert_frame_equal(df, expected)


def test_rolling_median_23480() -> None:
    vals = [None] * 17 + [3262645.8, 856191.4, 1635379.0, 34707156.0]
    evals = [None] * 19 + [1635379.0, (3262645.8 + 1635379.0) / 2]
    out = pl.DataFrame({"a": vals}).select(
        r15=pl.col("a").rolling_median(15, min_samples=3),
        r17=pl.col("a").rolling_median(17, min_samples=3),
    )
    expected = pl.DataFrame({"r15": evals, "r17": evals})
    assert_frame_equal(out, expected)


@pytest.mark.slow
@pytest.mark.parametrize("with_nulls", [True, False])
def test_rolling_sum_non_finite_23115(with_nulls: bool) -> None:
    values: list[float | None] = [
        0.0,
        float("nan"),
        float("inf"),
        -float("inf"),
        42.0,
        -3.0,
    ]
    if with_nulls:
        values.append(None)
    data = random.choices(values, k=1000)
    naive = [
        (
            sum(0 if x is None else x for x in data[max(0, i + 1 - 4) : i + 1])
            if sum(x is not None for x in data[max(0, i + 1 - 4) : i + 1]) >= 2
            else None
        )
        for i in range(1000)
    ]
    assert_series_equal(pl.Series(data).rolling_sum(4, min_samples=2), pl.Series(naive))


@pytest.mark.parametrize(
    ("method", "out_dtype"),
    [
        ("average", pl.Float64),
        ("min", get_index_type()),
        ("max", get_index_type()),
        ("dense", get_index_type()),
    ],
)
@given(
    s=series(
        name="a",
        allowed_dtypes=NUMERIC_DTYPES + TEMPORAL_DTYPES + [pl.Boolean],
        min_size=1,
        max_size=50,
    ),
    window_size=st.integers(1, 50),
)
def test_rolling_rank(
    s: pl.Series,
    window_size: int,
    method: RankMethod,
    out_dtype: pl.DataType,
) -> None:
    df = pl.DataFrame({"a": s})
    expected = (
        df.with_row_index()
        .with_columns(
            a=pl.col("a")
            .rank(method=method)
            .rolling(index_column="index", period=f"{window_size}i")
            .list.last()
            .cast(out_dtype)
        )
        .drop("index")
    )
    actual = df.lazy().select(
        pl.col("a").rolling_rank(
            window_size=window_size, method=method, seed=0, min_samples=1
        )
    )
    assert actual.collect_schema() == actual.collect().schema
    assert_frame_equal(actual.collect(), expected)


@pytest.mark.parametrize("center", [False, True])
@given(
    s=series(
        name="a",
        allowed_dtypes=NUMERIC_DTYPES + TEMPORAL_DTYPES + [pl.Boolean],
        min_size=1,
        max_size=50,
    ),
    window_size=st.integers(1, 50),
)
def test_rolling_rank_method_random(
    s: pl.Series, window_size: int, center: bool
) -> None:
    df = pl.DataFrame({"a": s})
    actual = df.lazy().with_columns(
        lo=pl.col("a").rolling_rank(
            window_size=window_size, method="min", center=center
        ),
        hi=pl.col("a").rolling_rank(
            window_size=window_size, method="max", center=center
        ),
        random=pl.col("a").rolling_rank(
            window_size=window_size,
            method="random",
            center=center,
        ),
    )

    assert actual.collect_schema() == actual.collect().schema, (
        f"expected {actual.collect_schema()}, got {actual.collect().schema}"
    )
    assert (
        actual.select(
            (
                (pl.col("lo") <= pl.col("random")) & (pl.col("random") <= pl.col("hi"))
            ).all()
        )
        .collect()
        .item()
    )


@pytest.mark.parametrize("op", [pl.Expr.rolling_mean, pl.Expr.rolling_median])
def test_rolling_mean_median_temporals(op: Callable[..., pl.Expr]) -> None:
    tz = ZoneInfo("Asia/Tokyo")
    # We use ms to verify that the correct time unit is propagating.
    dt = pl.Datetime("ms", "Asia/Tokyo")
    lf = pl.LazyFrame(
        {
            "int": [1, 2, 3, 4, 5],
            "date": [date(2025, 1, x) for x in range(1, 6)],
            "datetime": [datetime(2025, 1, x) for x in range(1, 6)],
            "datetime_tu_tz": pl.Series(
                [datetime(2025, 1, x, tzinfo=tz) for x in range(1, 6)], dtype=dt
            ),
            "duration": pl.Series(
                [timedelta(hours=x) for x in range(1, 6)], dtype=pl.Duration("ms")
            ),
            "time": [time(hour=x) for x in range(1, 6)],
        }
    )
    result = lf.select(
        rolling_date=op(pl.col("date"), window_size=4),
        rolling_datetime=op(pl.col("datetime"), window_size=4),
        rolling_datetime_tu_tz=op(pl.col("datetime_tu_tz"), window_size=4),
        rolling_duration=op(pl.col("duration"), window_size=4),
        rolling_time=op(pl.col("time"), window_size=4),
    )
    expected = pl.DataFrame(
        {
            "rolling_date": pl.Series(
                [None, None, None, datetime(2025, 1, 2, 12), datetime(2025, 1, 3, 12)],
                dtype=pl.Datetime,
            ),
            "rolling_datetime": pl.Series(
                [None, None, None, datetime(2025, 1, 2, 12), datetime(2025, 1, 3, 12)]
            ),
            "rolling_datetime_tu_tz": pl.Series(
                [
                    None,
                    None,
                    None,
                    datetime(2025, 1, 2, 12, tzinfo=tz),
                    datetime(2025, 1, 3, 12, tzinfo=tz),
                ],
                dtype=dt,
            ),
            "rolling_duration": pl.Series(
                [None, None, None, timedelta(hours=2.5), timedelta(hours=3.5)],
                dtype=pl.Duration("ms"),
            ),
            "rolling_time": [
                None,
                None,
                None,
                time(hour=2, minute=30),
                time(hour=3, minute=30),
            ],
        }
    )
    assert result.collect_schema() == pl.Schema(
        {  # type: ignore[arg-type]
            "rolling_date": pl.Datetime("us"),
            "rolling_datetime": pl.Datetime("us"),
            "rolling_datetime_tu_tz": dt,
            "rolling_duration": pl.Duration("ms"),
            "rolling_time": pl.Time,
        }
    )
    assert_frame_equal(result.collect(), expected)


@pytest.mark.parametrize(
    "op",
    [
        (pl.Expr.mean, pl.Expr.rolling_mean_by),
        (pl.Expr.median, pl.Expr.rolling_median_by),
    ],
)
def test_rolling_agg_mean_median_temporal(
    op: tuple[Callable[..., pl.Expr], Callable[..., pl.Expr]],
) -> None:
    tz = ZoneInfo("Asia/Tokyo")
    # We use ms to verify that the correct time unit is propagating.
    dt = pl.Datetime("ms", "Asia/Tokyo")
    lf = pl.LazyFrame(
        {
            "index": [1, 2, 3, 4, 5],
            "int": [1, 2, 3, 4, 5],
            "date": [date(2025, 1, x) for x in range(1, 6)],
            "datetime": [datetime(2025, 1, x) for x in range(1, 6)],
            "datetime_tu_tz": pl.Series(
                [datetime(2025, 1, x, tzinfo=tz) for x in range(1, 6)], dtype=dt
            ),
            "duration": pl.Series(
                [timedelta(hours=x) for x in range(1, 6)], dtype=pl.Duration("ms")
            ),
            "time": [time(hour=x) for x in range(1, 6)],
        }
    )

    # Using rolling.agg()
    result1 = lf.rolling("index", period="4i").agg(
        rolling_int=op[0](pl.col("int")),
        rolling_date=op[0](pl.col("date")),
        rolling_datetime=op[0](pl.col("datetime")),
        rolling_datetime_tu_tz=op[0](pl.col("datetime_tu_tz")),
        rolling_duration=op[0](pl.col("duration")),
        rolling_time=op[0](pl.col("time")),
    )
    # Using rolling_quantile_by()
    result2 = lf.select(
        "index",
        rolling_int=op[1](pl.col("int"), "index", window_size="4i"),
        rolling_date=op[1](pl.col("date"), "index", window_size="4i"),
        rolling_datetime=op[1](pl.col("datetime"), "index", window_size="4i"),
        rolling_datetime_tu_tz=op[1](
            pl.col("datetime_tu_tz"), "index", window_size="4i"
        ),
        rolling_duration=op[1](pl.col("duration"), "index", window_size="4i"),
        rolling_time=op[1](pl.col("time"), "index", window_size="4i"),
    )
    expected = pl.DataFrame(
        {
            "index": [1, 2, 3, 4, 5],
            "rolling_int": [1.0, 1.5, 2.0, 2.5, 3.5],
            "rolling_date": pl.Series(
                [
                    datetime(2025, 1, 1),
                    datetime(2025, 1, 1, 12),
                    datetime(2025, 1, 2),
                    datetime(2025, 1, 2, 12),
                    datetime(2025, 1, 3, 12),
                ]
            ),
            "rolling_datetime": pl.Series(
                [
                    datetime(2025, 1, 1),
                    datetime(2025, 1, 1, 12),
                    datetime(2025, 1, 2),
                    datetime(2025, 1, 2, 12),
                    datetime(2025, 1, 3, 12),
                ]
            ),
            "rolling_datetime_tu_tz": pl.Series(
                [
                    datetime(2025, 1, 1, tzinfo=tz),
                    datetime(2025, 1, 1, 12, tzinfo=tz),
                    datetime(2025, 1, 2, tzinfo=tz),
                    datetime(2025, 1, 2, 12, tzinfo=tz),
                    datetime(2025, 1, 3, 12, tzinfo=tz),
                ],
                dtype=dt,
            ),
            "rolling_duration": pl.Series(
                [
                    timedelta(hours=1),
                    timedelta(hours=1.5),
                    timedelta(hours=2),
                    timedelta(hours=2.5),
                    timedelta(hours=3.5),
                ],
                dtype=pl.Duration("ms"),
            ),
            "rolling_time": [
                time(hour=1),
                time(hour=1, minute=30),
                time(hour=2),
                time(hour=2, minute=30),
                time(hour=3, minute=30),
            ],
        }
    )
    expected_schema = pl.Schema(
        {  # type: ignore[arg-type]
            "index": pl.Int64,
            "rolling_int": pl.Float64,
            "rolling_date": pl.Datetime("us"),
            "rolling_datetime": pl.Datetime("us"),
            "rolling_datetime_tu_tz": dt,
            "rolling_duration": pl.Duration("ms"),
            "rolling_time": pl.Time,
        }
    )
    assert result1.collect_schema() == expected_schema
    assert result2.collect_schema() == expected_schema
    assert_frame_equal(result1.collect(), expected)
    assert_frame_equal(result2.collect(), expected)


@pytest.mark.parametrize(
    ("df", "expected"),
    [
        (
            pl.DataFrame(
                {"a": [1, 2, 3, 4], "offset": [0, 0, 0, 0], "len": [3, 1, 2, 1]}
            ),
            pl.DataFrame({"a": [6, 2, 7, 4]}),
        ),
        (
            pl.DataFrame(
                {
                    "a": [1, 2, 3, 4, 5, 6],
                    "offset": [0, 0, 2, 0, 0, 0],
                    "len": [3, 1, 3, 3, 1, 1],
                }
            ),
            pl.DataFrame({"a": [6, 2, 11, 15, 5, 6]}),
        ),
        (
            pl.DataFrame(
                {"a": [1, 2, 3, None], "offset": [0, 0, 0, 0], "len": [3, 1, 2, 1]}
            ),
            pl.DataFrame({"a": [6, 2, 3, 0]}),
        ),
        (
            pl.DataFrame(
                {
                    "a": [1, 2, 3, 4, 5, None],
                    "offset": [0, 0, 2, 0, 0, 0],
                    "len": [3, 1, 3, 3, 1, 1],
                }
            ),
            pl.DataFrame({"a": [6, 2, 5, 9, 5, 0]}),
        ),
    ],
)
def test_rolling_agg_sum_varying_slice_25434(
    df: pl.DataFrame, expected: pl.DataFrame
) -> None:
    out = df.with_row_index().select(
        pl.col("a")
        .slice(pl.col("offset").first(), pl.col("len").first())
        .sum()
        .rolling("index", period=f"{df.height}i", offset="0i", closed="left")
    )
    assert_frame_equal(out, expected)


@pytest.mark.parametrize("with_nulls", [True, False])
def test_rolling_agg_sum_varying_slice_fuzz(with_nulls: bool) -> None:
    n = 1000
    max_rand = 10

    def opt_null(n: int) -> int | None:
        return None if random.randint(0, max_rand) == max_rand and with_nulls else n

    df = pl.DataFrame(
        {
            "a": [opt_null(i) for i in range(n)],
            "offset": [random.randint(0, max_rand) for _ in range(n)],
            "length": [random.randint(0, max_rand) for _ in range(n)],
        }
    )

    out = df.with_row_index().select(
        pl.col("a")
        .slice(pl.col("offset").first(), pl.col("length").first())
        .sum()
        .rolling("index", period=f"{df.height}i", offset="0i", closed="left")
    )

    out = out.select(pl.col("a").fill_null(0))
    df = df.with_columns(pl.col("a").fill_null(0))

    (a, offset, length) = (
        df["a"].to_list(),
        df["offset"].to_list(),
        df["length"].to_list(),
    )
    expected = [sum(a[i + offset[i] : i + offset[i] + length[i]]) for i in range(n)]
    assert_frame_equal(out, pl.DataFrame({"a": expected}))


def test_rolling_midpoint_25793() -> None:
    df = pl.DataFrame({"i": [1, 2, 3, 4], "x": [1, 2, 3, 4]})

    out = df.select(
        pl.col.x.quantile(0.5, interpolation="midpoint").rolling("i", period="4i")
    )
    expected = pl.DataFrame({"x": [1.0, 1.5, 2.0, 2.5]})
    assert_frame_equal(out, expected)

    out = df.select(
        pl.col.x.cumulative_eval(pl.element().quantile(0.5, interpolation="midpoint"))
    )
    assert_frame_equal(out, expected)


def test_rolling_rank_closed_left_26147() -> None:
    df = pl.DataFrame(
        {
            "date": [datetime(2025, 1, 1), datetime(2025, 1, 1)],
            "x": [0, 1],
            "x_flipped": [1, 0],
        }
    )
    actual = df.with_columns(
        x_ranked=pl.col("x").rolling_rank_by("date", "2d"),
        x_flipped_ranked=pl.col("x_flipped").rolling_rank_by("date", "2d"),
    )
    expected = df.with_columns(
        x_ranked=pl.Series([1.0, 2.0]),
        x_flipped_ranked=pl.Series([2.0, 1.0]),
    )
    assert_frame_equal(actual, expected)
