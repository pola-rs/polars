from datetime import datetime

import hypothesis.strategies as st
from hypothesis import given

import polars as pl
from polars.exceptions import ComputeError
from polars.testing.parametric.strategies import strategy_datetime_format
from polars.type_aliases import TimeUnit


@given(
    datetimes=st.datetimes(
        min_value=datetime(1699, 1, 1),
        max_value=datetime(9999, 12, 31),
    ),
    fmt=strategy_datetime_format(),
)
def test_to_datetime(datetimes: datetime, fmt: str) -> None:
    input = datetimes.strftime(fmt)
    expected = datetime.strptime(input, fmt)
    try:
        result = pl.Series([input]).str.to_datetime(format=fmt).item()
    except ComputeError as exc:
        # If there's an exception, check that it's either:
        # - something which polars can't parse at all: missing day or month
        # - something on which polars intentionally raises
        assert (  # noqa: PT017
            (
                (("%H" in fmt) ^ ("%M" in fmt))
                or (("%I" in fmt) ^ ("%M" in fmt))
                or ("%S" in fmt and "%H" not in fmt)
                or ("%S" in fmt and "%I" not in fmt)
                or (("%I" in fmt) ^ ("%p" in fmt))
                or (("%H" in fmt) ^ ("%p" in fmt))
            )
            and "Invalid format string" in str(exc)
        ) or (
            (
                not any(day in fmt for day in ("%d", "%j"))
                or not any(month in fmt for month in ("%b", "%B", "%m"))
            )
            and "failed in column" in str(exc)
        )
    else:
        assert result == expected


@given(
    d=st.datetimes(
        min_value=datetime(1699, 1, 1),
        max_value=datetime(9999, 12, 31),
    ),
    tu=st.sampled_from(["ms", "us"]),
)
def test_cast_to_time_and_combine(d: datetime, tu: TimeUnit) -> None:
    # round-trip date/time extraction + recombining
    df = pl.DataFrame({"d": [d]}, schema={"d": pl.Datetime(tu)})
    res = df.select(
        d=pl.col("d"),
        dt=pl.col("d").dt.date(),
        tm=pl.col("d").cast(pl.Time),
    ).with_columns(
        dtm=pl.col("dt").dt.combine(pl.col("tm")),
    )

    datetimes = res["d"].to_list()
    assert [d.date() for d in datetimes] == res["dt"].to_list()
    assert [d.time() for d in datetimes] == res["tm"].to_list()
    assert datetimes == res["dtm"].to_list()
