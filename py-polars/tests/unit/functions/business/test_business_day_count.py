from datetime import date

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
