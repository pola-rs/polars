from __future__ import annotations

import inspect
import warnings
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.utils.convert import (
    _date_to_pl_date,
    _datetime_to_pl_timestamp,
    _time_to_pl_time,
    _timedelta_to_pl_duration,
    _timedelta_to_pl_timedelta,
)
from polars.utils.decorators import deprecate_nonkeyword_arguments, redirect
from polars.utils.various import parse_version

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


def test_redirect() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        # one-to-one redirection
        @redirect({"foo": "bar"})
        class DemoClass1:
            def bar(self, upper: bool = False) -> str:
                return "BAZ" if upper else "baz"

        assert DemoClass1().foo() == "baz"  # type: ignore[attr-defined]

        # redirection with **kwargs
        @redirect({"foo": ("bar", {"upper": True})})
        class DemoClass2:
            def bar(self, upper: bool = False) -> str:
                return "BAZ" if upper else "baz"

        assert DemoClass2().foo() == "BAZ"  # type: ignore[attr-defined]
