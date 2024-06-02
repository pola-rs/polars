from datetime import date, datetime

import hypothesis.strategies as st
from hypothesis import given

import polars as pl
from polars.testing import assert_series_equal


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
