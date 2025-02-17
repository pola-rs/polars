from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

import hypothesis.strategies as st
import pytest
from hypothesis import given

import polars as pl
from polars._utils.convert import parse_as_duration_string
from polars.testing import assert_series_equal

if TYPE_CHECKING:
    from polars._typing import TimeUnit


@given(
    value=st.datetimes(
        min_value=datetime(1000, 1, 1),
        max_value=datetime(3000, 1, 1),
    ),
    n=st.integers(min_value=1, max_value=100),
)
def test_truncate_monthly(value: date, n: int) -> None:
    result = pl.Series([value]).dt.truncate(f"{n}mo").item()
    # manual calculation
    total = value.year * 12 + value.month - 1
    remainder = total % n
    total -= remainder
    year, month = (total // 12), ((total % 12) + 1)
    expected = datetime(year, month, 1)
    assert result == expected


def test_truncate_date() -> None:
    # n vs n
    df = pl.DataFrame(
        {"a": [date(2020, 1, 1), None, date(2020, 1, 3)], "b": [None, "1mo", "1mo"]}
    )
    result = df.select(pl.col("a").dt.truncate(pl.col("b")))["a"]
    expected = pl.Series("a", [None, None, date(2020, 1, 1)])
    assert_series_equal(result, expected)

    # n vs 1
    df = pl.DataFrame(
        {"a": [date(2020, 1, 1), None, date(2020, 1, 3)], "b": [None, "1mo", "1mo"]}
    )
    result = df.select(pl.col("a").dt.truncate("1mo"))["a"]
    expected = pl.Series("a", [date(2020, 1, 1), None, date(2020, 1, 1)])
    assert_series_equal(result, expected)

    # n vs missing
    df = pl.DataFrame(
        {"a": [date(2020, 1, 1), None, date(2020, 1, 3)], "b": [None, "1mo", "1mo"]}
    )
    result = df.select(pl.col("a").dt.truncate(pl.lit(None, dtype=pl.String)))["a"]
    expected = pl.Series("a", [None, None, None], dtype=pl.Date)
    assert_series_equal(result, expected)

    # 1 vs n
    df = pl.DataFrame(
        {"a": [date(2020, 1, 1), None, date(2020, 1, 3)], "b": [None, "1mo", "1mo"]}
    )
    result = df.select(a=pl.date(2020, 1, 1).dt.truncate(pl.col("b")))["a"]
    expected = pl.Series("a", [None, date(2020, 1, 1), date(2020, 1, 1)])
    assert_series_equal(result, expected)

    # missing vs n
    df = pl.DataFrame(
        {"a": [date(2020, 1, 1), None, date(2020, 1, 3)], "b": [None, "1mo", "1mo"]}
    )
    result = df.select(a=pl.lit(None, dtype=pl.Date).dt.truncate(pl.col("b")))["a"]
    expected = pl.Series("a", [None, None, None], dtype=pl.Date)
    assert_series_equal(result, expected)


@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_truncate_datetime_simple(time_unit: TimeUnit) -> None:
    s = pl.Series([datetime(2020, 1, 2, 6)], dtype=pl.Datetime(time_unit))
    result = s.dt.truncate("1mo").item()
    assert result == datetime(2020, 1, 1)
    result = s.dt.truncate("1d").item()
    assert result == datetime(2020, 1, 2)


@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_truncate_datetime_w_expression(time_unit: TimeUnit) -> None:
    df = pl.DataFrame(
        {"a": [datetime(2020, 1, 2, 6), datetime(2020, 1, 3, 7)], "b": ["1mo", "1d"]},
        schema_overrides={"a": pl.Datetime(time_unit)},
    )
    result = df.select(pl.col("a").dt.truncate(pl.col("b")))["a"]
    assert result[0] == datetime(2020, 1, 1)
    assert result[1] == datetime(2020, 1, 3)


def test_pre_epoch_truncate_17581() -> None:
    s = pl.Series([datetime(1980, 1, 1), datetime(1969, 1, 1, 1)])
    result = s.dt.truncate("1d")
    expected = pl.Series([datetime(1980, 1, 1), datetime(1969, 1, 1)])
    assert_series_equal(result, expected)


@given(
    datetimes=st.lists(
        st.datetimes(min_value=datetime(1960, 1, 1), max_value=datetime(1980, 1, 1)),
        min_size=1,
        max_size=3,
    ),
    every=st.timedeltas(
        min_value=timedelta(microseconds=1), max_value=timedelta(days=1)
    ).map(parse_as_duration_string),
)
def test_fast_path_vs_slow_path(datetimes: list[datetime], every: str) -> None:
    s = pl.Series(datetimes)
    # Might use fastpath:
    result = s.dt.truncate(every)
    # Definitely uses slowpath:
    expected = s.dt.truncate(pl.Series([every] * len(datetimes)))
    assert_series_equal(result, expected)
