from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from decimal import Decimal as D
from zoneinfo import ZoneInfo

import pytest

import polars as pl
import polars.selectors as cs
from polars.testing import assert_frame_equal


def test_cum_sum() -> None:
    df = pl.DataFrame(
        {
            "int8": pl.Series([1, 2, 3, 4], dtype=pl.Int8),
            "int64": [1, 2, 3, 4],
            "uint64": pl.Series([1, 2, 3, 4], dtype=pl.UInt64),
            "float32": pl.Series([1.0, 2.0, 3.0, 4.0], dtype=pl.Float32),
            "float64": [1.0, 2.0, 3.0, 4.0],
            "bool": [True, False, True, True],
            "negative": [-1, 2, -3, 4],
            "i64_nulls": [1, None, 3, None],
            "decimal": [D("1.5"), D("2.5"), D("3.0"), D("4.0")],
            "duration": [
                timedelta(seconds=1),
                timedelta(seconds=2),
                timedelta(seconds=3),
                timedelta(seconds=4),
            ],
        }
    )
    res = df.select(pl.all().cum_sum())

    assert res.to_dict(as_series=False) == {
        "int8": [1, 3, 6, 10],
        "int64": [1, 3, 6, 10],
        "uint64": [1, 3, 6, 10],
        "float32": [1.0, 3.0, 6.0, 10.0],
        "float64": [1.0, 3.0, 6.0, 10.0],
        "bool": [1, 1, 2, 3],
        "negative": [-1, 1, -2, 2],
        "i64_nulls": [1, None, 4, None],
        "decimal": [D("1.5"), D("4.0"), D("7.0"), D("11.0")],
        "duration": [
            timedelta(seconds=1),
            timedelta(seconds=3),
            timedelta(seconds=6),
            timedelta(seconds=10),
        ],
    }
    assert res.schema == pl.Schema(
        {
            "int8": pl.Int64(),
            "int64": pl.Int64(),
            "uint64": pl.UInt64(),
            "float32": pl.Float32(),
            "float64": pl.Float64(),
            "bool": pl.UInt32(),
            "negative": pl.Int64(),
            "i64_nulls": pl.Int64(),
            "decimal": pl.Decimal(precision=38, scale=1),
            "duration": pl.Duration("us"),
        }
    )


def test_cum_min_max() -> None:
    df = pl.DataFrame(
        {
            "int": [3, 1, 4, 1, 5, 9, 2, 6],
            "float32": [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0],
            "negative": [-1, -5, 3, -2, 0, -3, 1, -4],
            "with_nulls": [5, None, 3, None, 1, None, 4, None],
            "bool": [None, True, True, None, False, None, True, False],
        },
        schema_overrides={"float32": pl.Float32},
    )
    res_min = df.select(pl.all().cum_min())
    res_max = df.select(pl.all().cum_max())

    assert res_min.to_dict(as_series=False) == {
        "int": [3, 1, 1, 1, 1, 1, 1, 1],
        "float32": [3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "negative": [-1, -5, -5, -5, -5, -5, -5, -5],
        "with_nulls": [5, None, 3, None, 1, None, 1, None],
        "bool": [None, True, True, None, False, None, False, False],
    }
    assert res_min["float32"].dtype == pl.Float32
    assert res_max.to_dict(as_series=False) == {
        "int": [3, 3, 4, 4, 5, 9, 9, 9],
        "float32": [3.0, 3.0, 4.0, 4.0, 5.0, 9.0, 9.0, 9.0],
        "negative": [-1, -1, 3, 3, 3, 3, 3, 3],
        "with_nulls": [5, None, 5, None, 5, None, 5, None],
        "bool": [None, True, True, None, True, None, True, True],
    }
    assert res_max["float32"].dtype == pl.Float32


def test_cumulative_nan_and_inf() -> None:
    nan = float("nan")
    inf = float("inf")

    # cum_min/cum_max with NaN and infinity
    df = pl.DataFrame(
        {
            "with_nan": [1.0, nan, 3.0],
            "inf_first": [inf, 1.0, 2.0],
            "neg_inf_first": [-inf, 1.0, 2.0],
        }
    )
    res_min = df.select(pl.all().cum_min())
    res_max = df.select(pl.all().cum_max())

    assert res_min.to_dict(as_series=False) == {
        "with_nan": [1.0, 1.0, 1.0],
        "inf_first": [inf, 1.0, 1.0],
        "neg_inf_first": [-inf, -inf, -inf],
    }
    assert res_max.to_dict(as_series=False) == {
        "with_nan": [1.0, 1.0, 3.0],
        "inf_first": [inf, inf, inf],
        "neg_inf_first": [-inf, 1.0, 2.0],
    }

    # cum_sum/cum_prod with infinity
    df_inf = pl.DataFrame(
        {
            "sum_inf": [1.0, inf, 2.0],
            "prod_inf": [1.0, inf, 2.0],
            "prod_zero_inf": [0.0, inf, 1.0],
        }
    )
    res = df_inf.select(
        pl.col("sum_inf").cum_sum().alias("sum"),
        pl.col("prod_inf").cum_prod().alias("prod"),
        pl.col("prod_zero_inf").cum_prod().alias("prod_nan"),
    )
    assert res["sum"].to_list() == [1.0, inf, inf]
    assert res["prod"].to_list() == [1.0, inf, inf]
    assert res["prod_nan"][0] == 0.0
    assert math.isnan(res["prod_nan"][1])
    assert math.isnan(res["prod_nan"][2])

    # cum_mean with NaN and infinity
    res_mean = pl.DataFrame(
        {
            "with_nan": [1.0, nan, 3.0],
            "with_inf": [1.0, inf, 2.0],
            "inf_to_nan": [inf, -inf, 1.0],
        }
    ).select(pl.all().cum_mean())

    assert res_mean["with_nan"][0] == 1.0
    assert math.isnan(res_mean["with_nan"][1])
    assert math.isnan(res_mean["with_nan"][2])

    # inf propagates
    assert res_mean["with_inf"].to_list() == [1.0, inf, inf]

    # inf + (-inf) = NaN, which then propagates
    assert res_mean["inf_to_nan"][0] == inf
    assert math.isnan(res_mean["inf_to_nan"][1])
    assert math.isnan(res_mean["inf_to_nan"][2])


def test_cum_prod() -> None:
    df = pl.DataFrame(
        {
            "int": [1, 2, 3, 4],
            "float32": pl.Series([1.0, 2.0, 3.0, 4.0], dtype=pl.Float32),
            "bool": [True, True, False, True],
            "negative": [-1, 2, -3, 4],
            "with_nulls": [2, None, 3, None],
        }
    )
    res = df.select(pl.all().cum_prod())

    assert res.to_dict(as_series=False) == {
        "int": [1, 2, 6, 24],
        "float32": [1.0, 2.0, 6.0, 24.0],
        "bool": [1, 1, 0, 0],
        "negative": [-1, -2, 6, 24],
        "with_nulls": [2, None, 6, None],
    }
    assert res["float32"].dtype == pl.Float32
    assert res["bool"].dtype == pl.Int64


def test_cum_count() -> None:
    df = pl.DataFrame(
        {
            "no_nulls": [1, 2, 3, 4],
            "with_nulls": ["a", None, "b", None],
            "all_nulls": pl.Series([None, None, None, None], dtype=pl.Int64),
        }
    )
    res = df.select(pl.all().cum_count())

    assert res.to_dict(as_series=False) == {
        "no_nulls": [1, 2, 3, 4],
        "with_nulls": [1, 1, 2, 2],
        "all_nulls": [0, 0, 0, 0],
    }
    assert res["no_nulls"].dtype == pl.UInt32


def test_cum_mean() -> None:
    df = pl.DataFrame(
        {
            "int": [1, 2, 3, 4],
            "float32": pl.Series([1.0, 2.0, 3.0, 4.0], dtype=pl.Float32),
            "float64": [1.0, 2.0, 3.0, 4.0],
            "bool": [True, False, True, True],
            "negative": [-4, 2, -2, 4],
            "with_nulls": [1.0, None, 3.0, None],
            "decimal": [D("1.00"), D("2.00"), D("3.00"), D("4.00")],
            "decimal_nulls": [D("2.00"), None, D("4.00"), None],
            "duration": [
                timedelta(seconds=2),
                timedelta(seconds=4),
                timedelta(seconds=6),
                timedelta(seconds=8),
            ],
            "duration_nulls": [timedelta(seconds=2), None, timedelta(seconds=4), None],
        },
        schema_overrides={"decimal": pl.Decimal(precision=20, scale=8)},
    )
    res = df.select(pl.all().cum_mean())

    assert res.to_dict(as_series=False) == {
        "int": [1.0, 1.5, 2.0, 2.5],
        "float32": [1.0, 1.5, 2.0, 2.5],
        "float64": [1.0, 1.5, 2.0, 2.5],
        "bool": [1.0, 0.5, 2 / 3, 0.75],
        "negative": [-4.0, -1.0, -4 / 3, 0.0],
        "with_nulls": [1.0, None, 2.0, None],
        "decimal": [D("1.00"), D("1.50"), D("2.00"), D("2.50")],
        "decimal_nulls": [D("2.00"), None, D("3.00"), None],
        "duration": [
            timedelta(seconds=2),
            timedelta(seconds=3),
            timedelta(seconds=4),
            timedelta(seconds=5),
        ],
        "duration_nulls": [timedelta(seconds=2), None, timedelta(seconds=3), None],
    }
    assert res.schema == pl.Schema(
        {
            "int": pl.Float64(),
            "float32": pl.Float32(),
            "float64": pl.Float64(),
            "bool": pl.Float64(),
            "negative": pl.Float64(),
            "with_nulls": pl.Float64(),
            "decimal": pl.Decimal(precision=20, scale=8),
            "decimal_nulls": pl.Decimal(precision=None, scale=2),
            "duration": pl.Duration("us"),
            "duration_nulls": pl.Duration("us"),
        }
    )


def test_cumulative_temporal_types() -> None:
    TK = ZoneInfo(key="Asia/Tokyo")
    res = pl.DataFrame(
        {
            "date": [
                date(1960, 4, 10),
                date(1960, 2, 28),
                date(1960, 8, 12),
                date(1960, 8, 25),
            ],
            "datetime": [
                datetime(2077, 1, 10, 10, 0),
                datetime(2077, 4, 14, 8, 15),
                datetime(2077, 2, 18, 12, 30),
                datetime(2077, 3, 22, 9, 45),
            ],
            "duration": [
                timedelta(hours=2, minutes=45),
                timedelta(hours=1),
                timedelta(hours=3, seconds=1234),
                timedelta(hours=4, microseconds=500001),
            ],
        },
        schema_overrides={
            "datetime": pl.Datetime("ms", time_zone=TK),
            "duration": pl.Duration("ns"),
        },
    ).select(
        cs.all().cum_min().name.suffix("_cum_min"),
        cs.all().cum_max().name.suffix("_cum_max"),
        cs.all().cum_mean().name.suffix("_cum_mean"),
    )

    assert res.schema == pl.Schema(
        {
            "date_cum_min": pl.Date(),
            "datetime_cum_min": pl.Datetime(time_unit="ms", time_zone=TK),
            "duration_cum_min": pl.Duration(time_unit="ns"),
            "date_cum_max": pl.Date(),
            "datetime_cum_max": pl.Datetime(time_unit="ms", time_zone=TK),
            "duration_cum_max": pl.Duration(time_unit="ns"),
            "date_cum_mean": pl.Datetime("us"),
            "datetime_cum_mean": pl.Datetime(time_unit="ms", time_zone=TK),
            "duration_cum_mean": pl.Duration(time_unit="ns"),
        }
    )
    assert res.to_dict(as_series=False) == {
        "date_cum_min": [
            date(1960, 4, 10),
            date(1960, 2, 28),
            date(1960, 2, 28),
            date(1960, 2, 28),
        ],
        "datetime_cum_min": [
            datetime(2077, 1, 10, 19, 0, tzinfo=TK),
            datetime(2077, 1, 10, 19, 0, tzinfo=TK),
            datetime(2077, 1, 10, 19, 0, tzinfo=TK),
            datetime(2077, 1, 10, 19, 0, tzinfo=TK),
        ],
        "duration_cum_min": [
            timedelta(seconds=9900),
            timedelta(seconds=3600),
            timedelta(seconds=3600),
            timedelta(seconds=3600),
        ],
        "date_cum_max": [
            date(1960, 4, 10),
            date(1960, 4, 10),
            date(1960, 8, 12),
            date(1960, 8, 25),
        ],
        "datetime_cum_max": [
            datetime(2077, 1, 10, 19, 0, tzinfo=TK),
            datetime(2077, 4, 14, 17, 15, tzinfo=TK),
            datetime(2077, 4, 14, 17, 15, tzinfo=TK),
            datetime(2077, 4, 14, 17, 15, tzinfo=TK),
        ],
        "duration_cum_max": [
            timedelta(seconds=9900),
            timedelta(seconds=9900),
            timedelta(seconds=12034),
            timedelta(seconds=14400, microseconds=500001),
        ],
        "date_cum_mean": [
            datetime(1960, 4, 10, 0, 0),
            datetime(1960, 3, 20, 0, 0),
            datetime(1960, 5, 7, 8, 0),
            datetime(1960, 6, 3, 18, 0),
        ],
        "datetime_cum_mean": [
            datetime(2077, 1, 10, 19, 0, tzinfo=TK),
            datetime(2077, 2, 26, 18, 7, 30, tzinfo=TK),
            datetime(2077, 2, 24, 3, 15, tzinfo=TK),
            datetime(2077, 3, 2, 19, 7, 30, tzinfo=TK),
        ],
        "duration_cum_mean": [
            timedelta(seconds=9900),
            timedelta(seconds=6750),
            timedelta(seconds=8511, microseconds=333333),
            timedelta(seconds=9983, microseconds=625000),
        ],
    }


def test_cumulative_reverse() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4], "with_nulls": [1, None, 3, None]})
    res = df.select(
        pl.col("a").cum_sum(reverse=True).alias("sum"),
        pl.col("a").cum_min(reverse=True).alias("min"),
        pl.col("a").cum_max(reverse=True).alias("max"),
        pl.col("a").cum_prod(reverse=True).alias("prod"),
        pl.col("a").cum_count(reverse=True).alias("count"),
        pl.col("a").cum_mean(reverse=True).alias("mean"),
        pl.col("with_nulls").cum_sum(reverse=True).alias("sum_nulls"),
        pl.col("with_nulls").cum_count(reverse=True).alias("count_nulls"),
    )

    assert res.to_dict(as_series=False) == {
        "sum": [10, 9, 7, 4],
        "min": [1, 2, 3, 4],
        "max": [4, 4, 4, 4],
        "prod": [24, 24, 12, 4],
        "count": [4, 3, 2, 1],
        "mean": [2.5, 3.0, 3.5, 4.0],
        "sum_nulls": [4, None, 3, None],
        "count_nulls": [2, 1, 1, 0],
    }


def test_cumulative_misc_edge_cases() -> None:
    # empty frame
    empty = pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)})
    res_empty = empty.select(
        pl.col("a").cum_sum().alias("sum"),
        pl.col("a").cum_min().alias("min"),
        pl.col("a").cum_max().alias("max"),
        pl.col("a").cum_prod().alias("prod"),
        pl.col("a").cum_count().alias("count"),
        pl.col("a").cum_mean().alias("mean"),
    )
    assert res_empty.height == 0
    assert res_empty.to_dict(as_series=False) == {
        "sum": [],
        "min": [],
        "max": [],
        "prod": [],
        "count": [],
        "mean": [],
    }

    # single element
    single = pl.DataFrame({"a": [42], "null": pl.Series([None], dtype=pl.Int64)})
    res_single = single.select(
        pl.col("a").cum_sum().alias("sum"),
        pl.col("a").cum_min().alias("min"),
        pl.col("a").cum_max().alias("max"),
        pl.col("a").cum_prod().alias("prod"),
        pl.col("a").cum_count().alias("count"),
        pl.col("a").cum_mean().alias("mean"),
        pl.col("null").cum_sum().alias("null_sum"),
        pl.col("null").cum_count().alias("null_count"),
    )
    assert res_single.to_dict(as_series=False) == {
        "sum": [42],
        "min": [42],
        "max": [42],
        "prod": [42],
        "count": [1],
        "mean": [42.0],
        "null_sum": [None],
        "null_count": [0],
    }

    # all nulls
    all_nulls = pl.DataFrame({"a": pl.Series([None, None, None], dtype=pl.Int64)})
    res_nulls = all_nulls.select(
        pl.col("a").cum_sum().alias("sum"),
        pl.col("a").cum_count().alias("count"),
    )
    assert res_nulls.to_dict(as_series=False) == {
        "sum": [None, None, None],
        "count": [0, 0, 0],
    }


def test_cumulative_over_groups() -> None:
    df = pl.DataFrame(
        {
            "grp": ["a", "a", "b", "b", "a"],
            "value": [1, 2, 10, 20, 3],
        }
    )
    res = df.with_columns(
        pl.col("value").cum_sum().over("grp").alias("cum_sum"),
        pl.col("value").cum_mean().over("grp").alias("cum_mean"),
        pl.col("value").cum_min().over("grp").alias("cum_min"),
        pl.col("value").cum_max().over("grp").alias("cum_max"),
    )
    assert res.to_dict(as_series=False) == {
        "grp": ["a", "a", "b", "b", "a"],
        "value": [1, 2, 10, 20, 3],
        "cum_sum": [1, 3, 10, 30, 6],
        "cum_mean": [1.0, 1.5, 10.0, 15.0, 2.0],
        "cum_min": [1, 1, 10, 10, 1],
        "cum_max": [1, 2, 10, 20, 3],
    }


def test_cum_mean_kahan() -> None:
    # naive f64 summation would fail this test due to catastrophic cancellation :p
    data = [1e16] + [1.0] * 1000 + [-1e16]
    res = pl.Series("n", data, dtype=pl.Float64).cum_mean()
    assert res[-1] == pytest.approx(1000.0 / 1002, rel=1e-12)


@pytest.mark.parametrize("no_optimization", [False, True])
def test_cumulative_streaming_vs_memory(no_optimization: bool) -> None:
    opts = pl.QueryOptFlags()
    if no_optimization:
        opts.none()

    dt = datetime(2026, 1, 31, 12, 0, 0)
    lf = (
        pl.LazyFrame(
            data={"a": range(-101, 101, 3)},
            schema_overrides={"a": pl.Int32},
        )
        .with_columns(
            b=pl.lit(dt).dt.add_business_days(pl.col("a"), roll="forward"),
            c=pl.lit(dt) + pl.col("a").cast(pl.Duration),
            d=pl.col("a").shift(1) / 2.3456789,
            dec=pl.col("a").cast(pl.Decimal(precision=20, scale=8)) / 100,
        )
        .select(
            cs.all().cum_min().name.suffix("_cum_min"),
            cs.all().cum_max().name.suffix("_cum_max"),
            cs.all().cum_mean().name.suffix("_cum_mean"),
            cs.numeric().cum_sum().name.suffix("_cum_sum"),
        )
    )
    res_streaming = lf.collect(engine="streaming", optimizations=opts)
    res_in_memory = lf.collect(engine="in-memory", optimizations=opts)
    assert_frame_equal(left=res_streaming, right=res_in_memory)
