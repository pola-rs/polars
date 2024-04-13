from datetime import date

import pytest

import polars as pl
from polars.testing import assert_series_equal


def test_business_day_count() -> None:
    # (Expression, expression)
    df = pl.DataFrame(
        {
            "start": [date(2020, 1, 1), date(2020, 1, 2)],
            "end": [date(2020, 1, 2), date(2020, 1, 10)],
        }
    )
    result = df.select(
        business_day_count=pl.business_day_count("start", "end"),
    )["business_day_count"]
    expected = pl.Series("business_day_count", [1, 6], pl.Int32)
    assert_series_equal(result, expected)

    # (Expression, scalar)
    result = df.select(
        business_day_count=pl.business_day_count("start", date(2020, 1, 10)),
    )["business_day_count"]
    expected = pl.Series("business_day_count", [7, 6], pl.Int32)
    assert_series_equal(result, expected)
    result = df.select(
        business_day_count=pl.business_day_count("start", pl.lit(None, dtype=pl.Date)),
    )["business_day_count"]
    expected = pl.Series("business_day_count", [None, None], pl.Int32)
    assert_series_equal(result, expected)

    # (Scalar, expression)
    result = df.select(
        business_day_count=pl.business_day_count(date(2020, 1, 1), "end"),
    )["business_day_count"]
    expected = pl.Series("business_day_count", [1, 7], pl.Int32)
    assert_series_equal(result, expected)
    result = df.select(
        business_day_count=pl.business_day_count(pl.lit(None, dtype=pl.Date), "end"),
    )["business_day_count"]
    expected = pl.Series("business_day_count", [None, None], pl.Int32)
    assert_series_equal(result, expected)

    # (Scalar, scalar)
    result = df.select(
        business_day_count=pl.business_day_count(date(2020, 1, 1), date(2020, 1, 10)),
    )["business_day_count"]
    expected = pl.Series("business_day_count", [7], pl.Int32)
    assert_series_equal(result, expected)


def test_business_day_count_w_week_mask() -> None:
    df = pl.DataFrame(
        {
            "start": [date(2020, 1, 1), date(2020, 1, 2)],
            "end": [date(2020, 1, 2), date(2020, 1, 10)],
        }
    )
    result = df.select(
        business_day_count=pl.business_day_count(
            "start", "end", week_mask=(True, True, True, True, True, True, False)
        ),
    )["business_day_count"]
    expected = pl.Series("business_day_count", [1, 7], pl.Int32)
    assert_series_equal(result, expected)

    result = df.select(
        business_day_count=pl.business_day_count(
            "start", "end", week_mask=(True, True, True, False, False, False, True)
        ),
    )["business_day_count"]
    expected = pl.Series("business_day_count", [1, 4], pl.Int32)
    assert_series_equal(result, expected)


def test_business_day_count_w_week_mask_invalid() -> None:
    with pytest.raises(ValueError, match=r"expected a sequence of length 7 \(got 2\)"):
        pl.business_day_count("start", "end", week_mask=(False, 0))  # type: ignore[arg-type]
    df = pl.DataFrame(
        {
            "start": [date(2020, 1, 1), date(2020, 1, 2)],
            "end": [date(2020, 1, 2), date(2020, 1, 10)],
        }
    )
    with pytest.raises(
        pl.ComputeError, match="`week_mask` must have at least one business day"
    ):
        df.select(pl.business_day_count("start", "end", week_mask=[False] * 7))


def test_business_day_count_schema() -> None:
    lf = pl.LazyFrame(
        {
            "start": [date(2020, 1, 1), date(2020, 1, 2)],
            "end": [date(2020, 1, 2), date(2020, 1, 10)],
        }
    )
    result = lf.select(
        business_day_count=pl.business_day_count("start", "end"),
    )
    assert result.schema["business_day_count"] == pl.Int32
    assert result.collect().schema["business_day_count"] == pl.Int32
    assert 'col("start").business_day_count([col("end")])' in result.explain()


def test_business_day_count_w_holidays() -> None:
    df = pl.DataFrame(
        {
            "start": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 2)],
            "end": [date(2020, 1, 2), date(2020, 1, 10), date(2020, 1, 9)],
        }
    )
    result = df.select(
        business_day_count=pl.business_day_count(
            "start", "end", holidays=[date(2020, 1, 1), date(2020, 1, 9)]
        ),
    )["business_day_count"]
    expected = pl.Series("business_day_count", [0, 5, 5], pl.Int32)
    assert_series_equal(result, expected)
