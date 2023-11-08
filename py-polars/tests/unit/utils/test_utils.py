from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pytest

import polars as pl
from polars.utils.convert import (
    _date_to_pl_date,
    _datetime_to_pl_timestamp,
    _time_to_pl_time,
    _timedelta_to_pl_duration,
    _timedelta_to_pl_timedelta,
)
from polars.utils.various import (
    _in_notebook,
    is_bool_sequence,
    is_int_sequence,
    is_sequence,
    is_str_sequence,
    parse_percentiles,
    parse_version,
)

if TYPE_CHECKING:
    from polars.type_aliases import TimeUnit


@pytest.mark.parametrize(
    ("dt", "time_unit", "expected"),
    [
        (datetime(2121, 1, 1), "ns", 4765132800000000000),
        (datetime(2121, 1, 1), "us", 4765132800000000),
        (datetime(2121, 1, 1), "ms", 4765132800000),
    ],
)
def test_datetime_to_pl_timestamp(
    dt: datetime, time_unit: TimeUnit, expected: int
) -> None:
    out = _datetime_to_pl_timestamp(dt, time_unit)
    assert out == expected


@pytest.mark.parametrize(
    ("t", "expected"),
    [
        (time(0, 0, 0), 0),
        (time(0, 0, 1), 1_000_000_000),
        (time(20, 52, 10), 75_130_000_000_000),
        (time(20, 52, 10, 200), 75_130_000_200_000),
    ],
)
def test_time_to_pl_time(t: time, expected: int) -> None:
    assert _time_to_pl_time(t) == expected


def test_date_to_pl_date() -> None:
    d = date(1999, 9, 9)
    out = _date_to_pl_date(d)
    assert out == 10843


def test_timedelta_to_pl_timedelta() -> None:
    out = _timedelta_to_pl_timedelta(timedelta(days=1), "ns")
    assert out == 86_400_000_000_000
    out = _timedelta_to_pl_timedelta(timedelta(days=1), "us")
    assert out == 86_400_000_000
    out = _timedelta_to_pl_timedelta(timedelta(days=1), "ms")
    assert out == 86_400_000
    out = _timedelta_to_pl_timedelta(timedelta(days=1), time_unit=None)
    assert out == 86_400_000_000


@pytest.mark.parametrize(
    ("td", "expected"),
    [
        (timedelta(days=1), "1d"),
        (timedelta(days=-1), "-1d"),
        (timedelta(seconds=1), "1s"),
        (timedelta(seconds=-1), "-1s"),
        (timedelta(microseconds=1), "1us"),
        (timedelta(microseconds=-1), "-1us"),
        (timedelta(days=1, seconds=1), "1d1s"),
        (timedelta(days=-1, seconds=-1), "-1d1s"),
        (timedelta(days=1, microseconds=1), "1d1us"),
        (timedelta(days=-1, microseconds=-1), "-1d1us"),
    ],
)
def test_timedelta_to_pl_duration(td: timedelta, expected: str) -> None:
    out = _timedelta_to_pl_duration(td)
    assert out == expected


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
