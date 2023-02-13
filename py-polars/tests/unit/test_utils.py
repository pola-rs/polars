from __future__ import annotations

import inspect
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.utils import (
    _date_to_pl_date,
    _datetime_to_pl_timestamp,
    _time_to_pl_time,
    _timedelta_to_pl_timedelta,
    deprecate_nonkeyword_arguments,
)

if TYPE_CHECKING:
    from polars.internals.type_aliases import TimeUnit


@pytest.mark.parametrize(
    ("dt", "tu", "expected"),
    [
        (datetime(2121, 1, 1), "ns", 4765132800000000000),
        (datetime(2121, 1, 1), "us", 4765132800000000),
        (datetime(2121, 1, 1), "ms", 4765132800000),
    ],
)
def test_datetime_to_pl_timestamp(dt: datetime, tu: TimeUnit, expected: int) -> None:
    out = _datetime_to_pl_timestamp(dt, tu)
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
    out = _timedelta_to_pl_timedelta(timedelta(days=1), tu=None)
    assert out == 86_400_000_000


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


class Foo:
    @deprecate_nonkeyword_arguments(allowed_args=["self", "baz"])
    def bar(self, baz: str, ham: str | None = None, foobar: str | None = None) -> None:
        ...


def test_deprecate_nonkeyword_arguments_method_signature() -> None:
    # Note the added star indicating keyword-only arguments after 'baz'
    expected = "(self, baz: 'str', *, ham: 'str | None' = None, foobar: 'str | None' = None) -> 'None'"
    assert str(inspect.signature(Foo.bar)) == expected


def test_deprecate_nonkeyword_arguments_method_warning() -> None:
    msg = (
        r"All arguments of Foo\.bar except for \'baz\' will be keyword-only in the next breaking release."
        r" Use keyword arguments to silence this warning."
    )
    with pytest.deprecated_call(match=msg):
        Foo().bar("qux", "quox")
