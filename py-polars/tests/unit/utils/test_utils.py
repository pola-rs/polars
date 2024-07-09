from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pytest

import polars as pl
from polars._utils.convert import (
    date_to_int,
    datetime_to_int,
    parse_as_duration_string,
    time_to_int,
    timedelta_to_int,
)
from polars._utils.various import (
    _in_notebook,
    is_bool_sequence,
    is_int_sequence,
    is_sequence,
    is_str_sequence,
    parse_percentiles,
    parse_version,
)

if TYPE_CHECKING:
    from zoneinfo import ZoneInfo

    from polars._typing import TimeUnit
else:
    from polars._utils.convert import string_to_zoneinfo as ZoneInfo


@pytest.mark.parametrize(
    ("td", "expected"),
    [
        (timedelta(), ""),
        (timedelta(days=1), "1d"),
        (timedelta(days=-1), "-1d"),
        (timedelta(seconds=1), "1s"),
        (timedelta(seconds=-1), "-1s"),
        (timedelta(microseconds=1), "1us"),
        (timedelta(microseconds=-1), "-1us"),
        (timedelta(days=1, seconds=1), "1d1s"),
        (timedelta(minutes=-1, seconds=1), "-59s"),
        (timedelta(days=-1, seconds=-1), "-1d1s"),
        (timedelta(days=1, microseconds=1), "1d1us"),
        (timedelta(days=-1, microseconds=-1), "-1d1us"),
        (None, None),
        ("1d2s", "1d2s"),
    ],
)
def test_parse_as_duration_string(
    td: timedelta | str | None, expected: str | None
) -> None:
    assert parse_as_duration_string(td) == expected


@pytest.mark.parametrize(
    ("d", "expected"),
    [
        (date(1999, 9, 9), 10_843),
        (date(1969, 12, 31), -1),
        (date.min, -719_162),
        (date.max, 2_932_896),
    ],
)
def test_date_to_int(d: date, expected: int) -> None:
    assert date_to_int(d) == expected


@pytest.mark.parametrize(
    ("t", "expected"),
    [
        (time(0, 0, 1), 1_000_000_000),
        (time(20, 52, 10), 75_130_000_000_000),
        (time(20, 52, 10, 200), 75_130_000_200_000),
        (time.min, 0),
        (time.max, 86_399_999_999_000),
        (time(12, 0, tzinfo=None), 43_200_000_000_000),
        (time(12, 0, tzinfo=ZoneInfo("UTC")), 43_200_000_000_000),
        (time(12, 0, tzinfo=ZoneInfo("Asia/Shanghai")), 43_200_000_000_000),
        (time(12, 0, tzinfo=ZoneInfo("US/Central")), 43_200_000_000_000),
    ],
)
def test_time_to_int(t: time, expected: int) -> None:
    assert time_to_int(t) == expected


@pytest.mark.parametrize(
    "tzinfo", [None, ZoneInfo("UTC"), ZoneInfo("Asia/Shanghai"), ZoneInfo("US/Central")]
)
def test_time_to_int_with_time_zone(tzinfo: Any) -> None:
    t = time(12, 0, tzinfo=tzinfo)
    assert time_to_int(t) == 43_200_000_000_000


@pytest.mark.parametrize(
    ("dt", "time_unit", "expected"),
    [
        (datetime(2121, 1, 1), "ns", 4_765_132_800_000_000_000),
        (datetime(2121, 1, 1), "us", 4_765_132_800_000_000),
        (datetime(2121, 1, 1), "ms", 4_765_132_800_000),
        (datetime(1969, 12, 31, 23, 59, 59, 999999), "us", -1),
        (datetime(1969, 12, 30, 23, 59, 59, 999999), "us", -86_400_000_001),
        (datetime.min, "ns", -62_135_596_800_000_000_000),
        (datetime.max, "ns", 253_402_300_799_999_999_000),
        (datetime.min, "ms", -62_135_596_800_000),
        (datetime.max, "ms", 253_402_300_799_999),
    ],
)
def test_datetime_to_int(dt: datetime, time_unit: TimeUnit, expected: int) -> None:
    assert datetime_to_int(dt, time_unit) == expected


@pytest.mark.parametrize(
    ("dt", "expected"),
    [
        (
            datetime(2000, 1, 1, 12, 0, tzinfo=None),
            946_728_000_000_000,
        ),
        (
            datetime(2000, 1, 1, 12, 0, tzinfo=ZoneInfo("UTC")),
            946_728_000_000_000,
        ),
        (
            datetime(2000, 1, 1, 12, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
            946_699_200_000_000,
        ),
        (
            datetime(2000, 1, 1, 12, 0, tzinfo=ZoneInfo("US/Central")),
            946_749_600_000_000,
        ),
    ],
)
def test_datetime_to_int_with_time_zone(dt: datetime, expected: int) -> None:
    assert datetime_to_int(dt, "us") == expected


@pytest.mark.parametrize(
    ("td", "time_unit", "expected"),
    [
        (timedelta(days=1), "ns", 86_400_000_000_000),
        (timedelta(days=1), "us", 86_400_000_000),
        (timedelta(days=1), "ms", 86_400_000),
        (timedelta.min, "ns", -86_399_999_913_600_000_000_000),
        (timedelta.max, "ns", 86_399_999_999_999_999_999_000),
        (timedelta.min, "ms", -86_399_999_913_600_000),
        (timedelta.max, "ms", 86_399_999_999_999_999),
    ],
)
def test_timedelta_to_int(td: timedelta, time_unit: TimeUnit, expected: int) -> None:
    assert timedelta_to_int(td, time_unit) == expected


def test_estimated_size() -> None:
    s = pl.Series("n", list(range(100)))
    df = s.to_frame()

    for sz in (s.estimated_size(), s.estimated_size("b"), s.estimated_size("bytes")):
        assert sz == df.estimated_size()

    assert s.estimated_size("kb") == (df.estimated_size("b") / 1024)
    assert s.estimated_size("mb") == (df.estimated_size("kb") / 1024)
    assert s.estimated_size("gb") == (df.estimated_size("mb") / 1024)
    assert s.estimated_size("tb") == (df.estimated_size("gb") / 1024)

    with pytest.raises(ValueError):
        s.estimated_size("milkshake")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("v1", "v2"),
    [
        ("0.16.8", "0.16.7"),
        ("23.0.0", (3, 1000)),
        ((23, 0, 0), "3.1000"),
        (("0", "0", "2beta"), "0.0.1"),
        (("2", "5", "0", "1"), (2, 5, 0)),
    ],
)
def test_parse_version(v1: Any, v2: Any) -> None:
    assert parse_version(v1) > parse_version(v2)
    assert parse_version(v2) < parse_version(v1)


@pytest.mark.slow()
def test_in_notebook() -> None:
    # private function, but easier to test this separately and mock it in the callers
    assert not _in_notebook()


@pytest.mark.parametrize(
    ("percentiles", "expected", "inject_median"),
    [
        (None, [0.5], True),
        (0.2, [0.2, 0.5], True),
        (0.5, [0.5], True),
        ((0.25, 0.75), [0.25, 0.5, 0.75], True),
        # Undocumented effect - percentiles get sorted.
        # Can be changed, this serves as documentation of current behaviour.
        ((0.6, 0.3), [0.3, 0.5, 0.6], True),
        (None, [], False),
        (0.2, [0.2], False),
        (0.5, [0.5], False),
        ((0.25, 0.75), [0.25, 0.75], False),
        ((0.6, 0.3), [0.3, 0.6], False),
    ],
)
def test_parse_percentiles(
    percentiles: Sequence[float] | float | None,
    expected: Sequence[float],
    inject_median: bool,
) -> None:
    assert parse_percentiles(percentiles, inject_median=inject_median) == expected


@pytest.mark.parametrize(("percentiles"), [(1.1), ([-0.1])])
def test_parse_percentiles_errors(percentiles: Sequence[float] | float | None) -> None:
    with pytest.raises(ValueError):
        parse_percentiles(percentiles)


@pytest.mark.parametrize(
    ("sequence", "include_series", "expected"),
    [
        (pl.Series(["xx", "yy"]), True, False),
        (pl.Series([True, False]), False, False),
        (pl.Series([True, False]), True, True),
        (np.array([False, True]), False, True),
        (np.array([False, True]), True, True),
        ([True, False], False, True),
        (["xx", "yy"], False, False),
        (True, False, False),
    ],
)
def test_is_bool_sequence_check(
    sequence: Any,
    include_series: bool,
    expected: bool,
) -> None:
    assert is_bool_sequence(sequence, include_series=include_series) == expected
    if expected:
        assert is_sequence(sequence, include_series=include_series)


@pytest.mark.parametrize(
    ("sequence", "include_series", "expected"),
    [
        (pl.Series(["xx", "yy"]), True, False),
        (pl.Series([123, 345]), False, False),
        (pl.Series([123, 345]), True, True),
        (np.array([123, 345]), False, True),
        (np.array([123, 345]), True, True),
        (["xx", "yy"], False, False),
        ([123, 456], False, True),
        (123, False, False),
    ],
)
def test_is_int_sequence_check(
    sequence: Any,
    include_series: bool,
    expected: bool,
) -> None:
    assert is_int_sequence(sequence, include_series=include_series) == expected
    if expected:
        assert is_sequence(sequence, include_series=include_series)


@pytest.mark.parametrize(
    ("sequence", "include_series", "expected"),
    [
        (pl.Series(["xx", "yy"]), False, False),
        (pl.Series(["xx", "yy"]), True, True),
        (pl.Series([123, 345]), True, False),
        (np.array(["xx", "yy"]), False, True),
        (np.array(["xx", "yy"]), True, True),
        (["xx", "yy"], False, True),
        ([123, 456], False, False),
        ("xx", False, False),
    ],
)
def test_is_str_sequence_check(
    sequence: Any,
    include_series: bool,
    expected: bool,
) -> None:
    assert is_str_sequence(sequence, include_series=include_series) == expected
    if expected:
        assert is_sequence(sequence, include_series=include_series)
