from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import hypothesis.strategies as st
from hypothesis import assume, given, reject

import polars as pl
from polars.testing import assert_frame_equal
from polars.testing.parametric.primitives import column, dataframes
from polars.testing.parametric.strategies import strategy_closed, strategy_time_unit
from polars.utils.convert import _timedelta_to_pl_duration

if TYPE_CHECKING:
    from polars.type_aliases import ClosedInterval, TimeUnit


@given(
    period=st.timedeltas(min_value=timedelta(microseconds=0)).map(
        _timedelta_to_pl_duration
    ),
    offset=st.timedeltas().map(_timedelta_to_pl_duration),
    closed=strategy_closed,
    data=st.data(),
    time_unit=strategy_time_unit,
)
def test_groupby_rolling(
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
                column("ts", dtype=pl.Datetime(time_unit)),
                column("value", dtype=pl.Int64),
            ],
        )
    )
    df = dataframe.sort("ts").unique("ts")
    try:
        result = df.groupby_rolling(
            "ts", period=period, offset=offset, closed=closed
        ).agg(pl.col("value"))
    except pl.exceptions.PolarsPanicError as exc:
        assert any(  # noqa: PT017
            msg in str(exc)
            for msg in (
                "attempt to multiply with overflow",
                "attempt to add with overflow",
            )
        )
        reject()

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
