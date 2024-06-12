from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import hypothesis.strategies as st
import pytest
from hypothesis import given

import polars as pl
from polars.exceptions import ComputeError

if TYPE_CHECKING:
    from hypothesis.strategies import DrawFn

    from polars.type_aliases import TimeUnit


DATE_FORMATS = ["%Y{}%m{}%d", "%d{}%m{}%Y"]
SEPARATORS = ["-", "/", "."]
TIME_FORMATS = [
    "T%H:%M:%S",
    "T%H%M%S",
    "T%H:%M",
    "T%H%M",
    " %H:%M:%S",
    " %H%M%S",
    " %H:%M",
    " %H%M",
    "",  # allow no time part (plain date)
]
FRACTIONS = [
    "%.9f",
    "%.6f",
    "%.3f",
    # "%.f", # alternative which allows any number of digits
    "",
]
TIMEZONES = ["%#z", ""]
DATETIME_PATTERNS = [
    date_format.format(separator, separator) + time_format + fraction + tz
    for separator in SEPARATORS
    for date_format in DATE_FORMATS
    for time_format in TIME_FORMATS
    for fraction in FRACTIONS
    if time_format.endswith("%S") or fraction == ""
    for tz in TIMEZONES
    if date_format.startswith("%Y") and time_format != "" or tz == ""
]


@pytest.mark.parametrize("fmt", DATETIME_PATTERNS)
def test_to_datetime_inferable_formats(fmt: str) -> None:
    time_string = (
        fmt.replace("%Y", "2024")
        .replace("%m", "12")
        .replace("%d", "13")
        .replace("%H", "23")
        .replace("%M", "34")
        .replace("%S", "45")
        .replace("%.3f", ".123")
        .replace("%.6f", ".123456")
        .replace("%.9f", ".123456789")
        .replace("%#z", "+0100")
    )

    pl.Series([time_string]).str.to_datetime(strict=True)


@st.composite
def datetime_formats(draw: DrawFn) -> str:
    """Returns a strategy which generates datetime format strings."""
    parts = [
        "%m",
        "%b",
        "%B",
        "%d",
        "%j",
        "%a",
        "%A",
        "%w",
        "%H",
        "%I",
        "%p",
        "%M",
        "%S",
        "%U",
        "%W",
        "%%",
    ]
    fmt = draw(st.sets(st.sampled_from(parts)))
    fmt.add("%Y")  # Make sure year is always present
    return " ".join(fmt)


@given(
    datetimes=st.datetimes(
        min_value=datetime(1699, 1, 1),
        max_value=datetime(9999, 12, 31),
    ),
    fmt=datetime_formats(),
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
