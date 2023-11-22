from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy import nan

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from polars.type_aliases import ClosedInterval, TimeUnit


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
def test_rolling_kernels_and_rolling(
    example_df: pl.DataFrame, period: str | timedelta, closed: ClosedInterval
) -> None:
    out1 = example_df.set_sorted("dt").select(
        [
            pl.col("dt"),
            # this differs from group_by aggregation because the empty window is
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
        .rolling("dt", period=period, closed=closed)
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
    ts = pl.datetime_range(
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
            .name.suffix("_nulls")
        ]
    )

    assert df.select([pl.all().rolling_min(3)]).to_dict(as_series=False) == {
        "col1": [None, None, 0, 1, 2, 3, 4],
        "col2": [None, None, 4, 3, 2, 1, 0],
        "col1_nulls": [None, None, None, None, 2, 3, 4],
        "col2_nulls": [None, None, None, None, 2, 1, 0],
    }

    assert df.select([pl.all().rolling_max(3)]).to_dict(as_series=False) == {
        "col1": [None, None, 2, 3, 4, 5, 6],
        "col2": [None, None, 6, 5, 4, 3, 2],
        "col1_nulls": [None, None, None, None, 4, 5, 6],
        "col2_nulls": [None, None, None, None, 4, 3, 2],
    }

    # shuffled data triggers other kernels
    df = df.select([pl.all().shuffle(0)])
    assert df.select([pl.all().rolling_min(3)]).to_dict(as_series=False) == {
        "col1": [None, None, 0, 0, 1, 2, 2],
        "col2": [None, None, 0, 2, 1, 1, 1],
        "col1_nulls": [None, None, None, None, None, 2, 2],
        "col2_nulls": [None, None, None, None, None, 1, 1],
    }

    assert df.select([pl.all().rolling_max(3)]).to_dict(as_series=False) == {
        "col1": [None, None, 6, 4, 5, 5, 5],
        "col2": [None, None, 6, 6, 5, 4, 4],
        "col1_nulls": [None, None, None, None, None, 5, 5],
        "col2_nulls": [None, None, None, None, None, 4, 4],
    }


def test_rolling_group_by_extrema() -> None:
    # ensure we hit different branches so create

    df = pl.DataFrame(
        {
            "col1": pl.arange(0, 7, eager=True).reverse(),
        }
    ).with_columns(pl.col("col1").reverse().alias("row_nr"))

    assert (
        df.rolling(
            index_column="row_nr",
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
    ).to_dict(as_series=False) == {
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
        df.rolling(
            index_column="row_nr",
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
    ).to_dict(as_series=False) == {
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
        df.rolling(
            index_column="row_nr",
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
    ).to_dict(as_series=False) == {
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
        .rolling(
            "a",
            by="b",
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
        (name, data.shape)
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
        for name, data in df.rolling(index_column="date", period="2d", by="a")
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


def test_rolling_cov_corr() -> None:
    df = pl.DataFrame({"x": [3, 3, 3, 5, 8], "y": [3, 4, 4, 4, 8]})

    res = df.select(
        [
            pl.rolling_cov("x", "y", window_size=3).alias("cov"),
            pl.rolling_corr("x", "y", window_size=3).alias("corr"),
        ]
    ).to_dict(as_series=False)
    assert res["cov"][2:] == pytest.approx([0.0, 0.0, 5.333333333333336])
    assert res["corr"][2:] == pytest.approx([nan, nan, 0.9176629354822473], nan_ok=True)
    assert res["cov"][:2] == [None] * 2
    assert res["corr"][:2] == [None] * 2


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
    with pytest.warns(
        UserWarning, match="Series is not known to be sorted by `by` column."
    ):
        _ = df.with_columns(
            roll=pl.col("val").rolling_sum("2d", by="dt", closed="right")
        )


def test_rolling_aggregations_with_over_11225() -> None:
    start = datetime(2001, 1, 1)

    df_temporal = pl.DataFrame(
        {
            "date": [start + timedelta(days=k) for k in range(5)],
            "group": ["A"] * 2 + ["B"] * 3,
        }
    ).with_row_count()

    df_temporal = df_temporal.sort("group", "date")

    result = df_temporal.with_columns(
        rolling_row_mean=pl.col("row_nr")
        .rolling_mean(
            window_size="2d",
            by="date",
            closed="left",
            warn_if_unsorted=False,
        )
        .over("group")
    )
    expected = pl.DataFrame(
        {
            "row_nr": [0, 1, 2, 3, 4],
            "date": pl.datetime_range(date(2001, 1, 1), date(2001, 1, 5), eager=True),
            "group": ["A", "A", "B", "B", "B"],
            "rolling_row_mean": [None, 0.0, None, 2.0, 2.5],
        },
        schema_overrides={"row_nr": pl.UInt32},
    )
    assert_frame_equal(result, expected)


def test_rolling() -> None:
    a = pl.Series("a", [1, 2, 3, 2, 1])
    assert_series_equal(a.rolling_min(2), pl.Series("a", [None, 1, 2, 2, 1]))
    assert_series_equal(a.rolling_max(2), pl.Series("a", [None, 2, 3, 3, 2]))
    assert_series_equal(a.rolling_sum(2), pl.Series("a", [None, 3, 5, 5, 3]))
    assert_series_equal(a.rolling_mean(2), pl.Series("a", [None, 1.5, 2.5, 2.5, 1.5]))

    assert a.rolling_std(2).to_list()[1] == pytest.approx(0.7071067811865476)
    assert a.rolling_var(2).to_list()[1] == pytest.approx(0.5)
    assert a.rolling_std(2, ddof=0).to_list()[1] == pytest.approx(0.5)
    assert a.rolling_var(2, ddof=0).to_list()[1] == pytest.approx(0.25)

    assert_series_equal(
        a.rolling_median(4), pl.Series("a", [None, None, None, 2, 2], dtype=pl.Float64)
    )
    assert_series_equal(
        a.rolling_quantile(0, "nearest", 3),
        pl.Series("a", [None, None, 1, 2, 1], dtype=pl.Float64),
    )
    assert_series_equal(
        a.rolling_quantile(0, "lower", 3),
        pl.Series("a", [None, None, 1, 2, 1], dtype=pl.Float64),
    )
    assert_series_equal(
        a.rolling_quantile(0, "higher", 3),
        pl.Series("a", [None, None, 1, 2, 1], dtype=pl.Float64),
    )
    assert a.rolling_skew(4).null_count() == 3

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
    a = pl.Series("a", [11.0, 2.0, 9.0, nan, 8.0])
    assert_series_equal(
        a.rolling_sum(3),
        pl.Series("a", [None, None, 22.0, nan, nan]),
    )


def test_rolling_by_date() -> None:
    df = pl.DataFrame(
        {
            "dt": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
            "val": [1, 2, 3],
        }
    ).sort("dt")

    result = df.with_columns(
        roll=pl.col("val").rolling_sum("2d", by="dt", closed="right")
    )
    expected = df.with_columns(roll=pl.Series([1, 3, 5]))
    assert_frame_equal(result, expected)


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
    result = df.with_columns(
        pl.col("val").rolling_sum("500ns", by="dt", closed="right")
    )
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
    with pytest.deprecated_call(match="The '_saturating' suffix is deprecated"):
        result = df.rolling(index_column="date", period="1mo_saturating").agg(
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

    # pl.int_range's output name is: `int`.
    out = df.rolling(pl.int_range(0, pl.count()), period="2i").agg("A")
    assert out.to_dict(as_series=False) == {
        "int": [0, 1, 2],
        "A": [[1], [1, 2], [2, 3]],
    }
