from __future__ import annotations

import datetime as dt
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polars.type_aliases import StartBy

    if sys.version_info >= (3, 8):
        from typing import Literal
    else:
        from typing_extensions import Literal
if sys.version_info >= (3, 9):
    import zoneinfo
else:
    from backports import zoneinfo

import pandas as pd
import pytz
from hypothesis import given, reject
from hypothesis import strategies as st

import polars as pl


def _compare_polars_and_pandas(
    *,
    data: st.DataObject,
    datetimes: list[dt.datetime],
    timezone: str,
    pl_every: str,
    pd_alias: str,
    pl_startby: StartBy,
    closed: Literal["left", "right"],
    number: int,
) -> None:
    nrows = len(datetimes)
    values = data.draw(
        st.lists(st.floats(10, 20), min_size=nrows, max_size=nrows), label="values"
    )
    df = (
        pl.DataFrame({"ts": datetimes, "values": values})
        .sort("ts")
        .with_columns(
            pl.col("ts").dt.replace_time_zone("UTC").dt.convert_time_zone(timezone)
        )
    )

    result = df.groupby_dynamic(
        "ts",
        every=f"{number}{pl_every}",
        start_by=pl_startby,
        closed=closed,
    ).agg(pl.col("values").sum())

    result_pd = (
        df.to_pandas()
        .resample(f"{number}{pd_alias}", closed=closed, label="left", on="ts")["values"]
        .sum()
        .reset_index()
    )

    # pandas fills in "holes", but polars doesn't
    # https://github.com/pola-rs/polars/issues/8831
    result_pd = result_pd[result_pd["values"] != 0.0].reset_index(drop=True)

    result_pl = result.to_pandas()
    pd.testing.assert_frame_equal(result_pd, result_pl)


@given(
    datetimes=st.lists(
        st.datetimes(
            min_value=dt.datetime(1980, 1, 1),
            # Can't currently go beyond 2038, see
            # https://github.com/pola-rs/polars/issues/9315
            max_value=dt.datetime(2038, 1, 1),
        ),
        min_size=1,
    ),
    timezone=st.sampled_from(list(zoneinfo.available_timezones())),
    closed=st.sampled_from(("left", "right")),
    every_alias=st.sampled_from(
        (
            ("d", "D"),
            ("mo", "MS"),
            ("y", "YS"),
        )
    ),
    number=st.integers(
        min_value=1,
        # Can't currently go above 1:
        # https://github.com/pola-rs/polars/issues/9333
        max_value=1,
    ),
    data=st.data(),
)
def test_non_weekly(
    datetimes: list[dt.datetime],
    closed: Literal["left", "right"],
    timezone: str,
    every_alias: tuple[str, str],
    number: int,
    data: st.DataObject,
) -> None:
    pl_every, pd_alias = every_alias
    try:
        _compare_polars_and_pandas(
            data=data,
            datetimes=datetimes,
            pl_every=pl_every,
            pd_alias=pd_alias,
            pl_startby="window",
            closed=closed,
            number=number,
            timezone=timezone,
        )
    except pl.exceptions.ComputeError as exp:
        assert "unable to parse time zone" in str(exp)  # noqa: PT017
        reject()
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
    datetimes=st.lists(
        st.datetimes(
            min_value=dt.datetime(1980, 1, 1),
            # Can't currently go beyond 2038, see
            # https://github.com/pola-rs/polars/issues/9315
            max_value=dt.datetime(2038, 1, 1),
        ),
        min_size=1,
    ),
    timezone=st.sampled_from(list(zoneinfo.available_timezones())),
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
        # Can't currently go above 1:
        # https://github.com/pola-rs/polars/issues/9333
        max_value=1,
    ),
    data=st.data(),
)
def test_weekly(
    datetimes: list[dt.datetime],
    closed: Literal["left", "right"],
    timezone: str,
    every_alias_startby: tuple[str, str, StartBy],
    number: int,
    data: st.DataObject,
) -> None:
    # If/when the pandas bug
    # https://github.com/pandas-dev/pandas/issues/53612
    # is fixed, this should be integrated into the previous test.
    # Trying to do so at the moment results in hypothesis complaining
    # about too many skipped tests.
    pl_every, pd_alias, pl_startby = every_alias_startby
    try:
        _compare_polars_and_pandas(
            data=data,
            datetimes=datetimes,
            pl_every=pl_every,
            pd_alias=pd_alias,
            pl_startby=pl_startby,
            closed=closed,
            number=number,
            timezone=timezone,
        )
    except pl.exceptions.ComputeError as exp:
        assert "unable to parse time zone" in str(exp)  # noqa: PT017
        reject()
    except pl.exceptions.PolarsPanicError as exp:
        # This computation may fail in the rare case that the beginning of a month
        # lands on a DST transition.
        assert "is non-existent" in str(exp) or "is ambiguous" in str(  # noqa: PT017
            exp
        )
        reject()
    except (pytz.exceptions.NonExistentTimeError, pytz.exceptions.AmbiguousTimeError):
        reject()
