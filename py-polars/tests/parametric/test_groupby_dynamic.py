# --------------------------------------------------------
# Validate groupby_dynamic behaviour with parametric tests
# --------------------------------------------------------
#
# Note: there are three tests here which look very similar.
# The reason they are split into different tests is due to
# some upstream bugs in pandas/pyarrow which would cause some tests
# to fail if they were all in the same test function. Ideally,
# once upstream is fixed, these tests should be merged into one.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sys

    from polars.type_aliases import StartBy

    if sys.version_info >= (3, 8):
        from typing import Literal
    else:
        from typing_extensions import Literal

import pandas as pd
import pytz
from hypothesis import given, reject
from hypothesis import strategies as st

import polars as pl
from polars.testing.parametric.primitives import series
from polars.testing.parametric.strategies import strategy_time_zone_aware_series


def _compare_polars_and_pandas(
    *,
    data: st.DataObject,
    time_series: pl.Series,
    pl_every: str,
    pd_alias: str,
    pl_startby: StartBy,
    closed: Literal["left", "right"],
    number: int,
) -> None:
    nrows = len(time_series)
    values = data.draw(
        series(
            name="values",
            dtype=pl.Float64,
            strategy=st.floats(min_value=10, max_value=20),
            size=nrows,
        )
    )
    df = pl.DataFrame({"ts": time_series, "values": values}).sort("ts")

    result = df.groupby_dynamic(
        "ts",
        every=f"{number}{pl_every}",
        start_by=pl_startby,
        closed=closed,
    ).agg(pl.col("values").sum())

    if len(result) > 0:
        origin = result["ts"].dt.offset_by(f"-{number}{pl_every}")[0]
    else:
        origin = "epoch"
    result_pd = (
        df.to_pandas()
        .resample(
            f"{number}{pd_alias}", closed=closed, label="left", on="ts", origin=origin
        )["values"]
        .sum()
        .reset_index()
    )

    # Work around bug in pandas: https://github.com/pandas-dev/pandas/issues/53664
    if len(result_pd) == 0:
        time_zone = df["ts"].dtype.time_zone  # type: ignore[union-attr]
        result_pd["ts"] = result_pd["ts"].dt.tz_localize(time_zone)

    # pandas fills in "holes", but polars doesn't
    # https://github.com/pola-rs/polars/issues/8831
    result_pd = result_pd[result_pd["values"] != 0.0].reset_index(drop=True)

    result_pl = result.to_pandas()
    pd.testing.assert_frame_equal(result_pd, result_pl)


@given(
    time_series=strategy_time_zone_aware_series(),
    closed=st.sampled_from(("left", "right")),
    every_alias=st.sampled_from((("mo", "MS"), ("y", "YS"))),
    number=st.integers(
        min_value=1,
        # Can't currently go above 1 due to bug in pandas:
        # https://github.com/pandas-dev/pandas/issues/53662
        max_value=1,
    ),
    data=st.data(),
)
def test_monthly_and_yearly(
    time_series: pl.Series,
    closed: Literal["left", "right"],
    every_alias: tuple[str, str],
    number: int,
    data: st.DataObject,
) -> None:
    pl_every, pd_alias = every_alias
    try:
        _compare_polars_and_pandas(
            data=data,
            time_series=time_series,
            pl_every=pl_every,
            pd_alias=pd_alias,
            pl_startby="window",
            closed=closed,
            number=number,
        )
    except pl.exceptions.PolarsPanicError as exp:
        # This computation may fail in the rare case that the beginning of a month
        # lands on a DST transition.
        assert "is non-existent" in str(exp) or "is ambiguous" in str(  # noqa: PT017
            exp
        )
        reject()
    except (pytz.exceptions.NonExistentTimeError, pytz.exceptions.AmbiguousTimeError):
        reject()


@given(
    time_series=strategy_time_zone_aware_series(),
    closed=st.sampled_from(
        (
            "left",
            # Can't test closed='right', bug in pandas:
            # https://github.com/pandas-dev/pandas/issues/53612
        )
    ),
    every_alias_startby=st.sampled_from(
        (
            ("w", "W-Mon", "monday"),
            ("w", "W-Tue", "tuesday"),
            ("w", "W-Wed", "wednesday"),
            ("w", "W-Thu", "thursday"),
            ("w", "W-Fri", "friday"),
            ("w", "W-Sat", "saturday"),
            ("w", "W-Sun", "sunday"),
        )
    ),
    number=st.integers(
        min_value=1,
        # Can't currently go above 1 due to bug in pandas:
        # https://github.com/pandas-dev/pandas/issues/53662
        max_value=1,
    ),
    data=st.data(),
)
def test_weekly(
    time_series: pl.Series,
    closed: Literal["left", "right"],
    every_alias_startby: tuple[str, str, StartBy],
    number: int,
    data: st.DataObject,
) -> None:
    pl_every, pd_alias, pl_startby = every_alias_startby
    try:
        _compare_polars_and_pandas(
            data=data,
            time_series=time_series,
            pl_every=pl_every,
            pd_alias=pd_alias,
            pl_startby=pl_startby,
            closed=closed,
            number=number,
        )
    except pl.exceptions.PolarsPanicError as exp:
        # This computation may fail in the rare case that the beginning of a month
        # lands on a DST transition.
        assert "is non-existent" in str(exp) or "is ambiguous" in str(  # noqa: PT017
            exp
        )
        reject()
    except (pytz.exceptions.NonExistentTimeError, pytz.exceptions.AmbiguousTimeError):
        reject()


@given(
    time_series=strategy_time_zone_aware_series(),
    closed=st.sampled_from(
        (
            "left",
            # Can't test closed='right', bug in pandas:
            # https://github.com/pandas-dev/pandas/issues/53612
        )
    ),
    number=st.integers(min_value=1, max_value=3),
    data=st.data(),
)
def test_daily(
    time_series: pl.Series,
    closed: Literal["left", "right"],
    number: int,
    data: st.DataObject,
) -> None:
    pl_every, pd_alias = "d", "D"
    try:
        _compare_polars_and_pandas(
            data=data,
            time_series=time_series,
            pl_every=pl_every,
            pd_alias=pd_alias,
            pl_startby="window",
            closed=closed,
            number=number,
        )
    except pl.exceptions.PolarsPanicError as exp:
        # This computation may fail in the rare case that the beginning of a month
        # lands on a DST transition.
        assert "is non-existent" in str(exp) or "is ambiguous" in str(  # noqa: PT017
            exp
        )
        reject()
    except (pytz.exceptions.NonExistentTimeError, pytz.exceptions.AmbiguousTimeError):
        reject()
