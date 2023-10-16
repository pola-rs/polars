from __future__ import annotations

import datetime as dt
from datetime import timedelta
from typing import TYPE_CHECKING

import hypothesis.strategies as st
from hypothesis import assume, given

import polars as pl
from polars.testing import assert_frame_equal
from polars.testing.parametric.primitives import column, dataframes
from polars.testing.parametric.strategies import strategy_closed, strategy_time_unit
from polars.utils.convert import _timedelta_to_pl_duration

if TYPE_CHECKING:
    from polars.type_aliases import ClosedInterval, TimeUnit


@given(
    period=st.timedeltas(
        min_value=timedelta(microseconds=0), max_value=timedelta(days=1000)
    ).map(_timedelta_to_pl_duration),
    offset=st.timedeltas(
        min_value=timedelta(microseconds=0), max_value=timedelta(days=1000)
    ).map(_timedelta_to_pl_duration),
    closed=strategy_closed,
    data=st.data(),
    time_unit=strategy_time_unit,
)
def test_rolling(
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
                        min_value=dt.datetime(2000, 1, 1),
                        max_value=dt.datetime(2001, 1, 1),
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
    ).map(_timedelta_to_pl_duration),
    closed=strategy_closed,
    data=st.data(),
    time_unit=strategy_time_unit,
    aggregation=st.sampled_from(
        [
            "min",
            "max",
            "mean",
            "sum",
            #  "std", blocked by https://github.com/pola-rs/polars/issues/11140
            #  "var", blocked by https://github.com/pola-rs/polars/issues/11140
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
    dataframe = data.draw(
        dataframes(
            [
                column(
                    "ts",
                    strategy=st.datetimes(
                        min_value=dt.datetime(2000, 1, 1),
                        max_value=dt.datetime(2001, 1, 1),
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
    func = f"rolling_{aggregation}"
    result = df.with_columns(
        getattr(pl.col("value"), func)(window_size=window_size, by="ts", closed=closed)
    )

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
