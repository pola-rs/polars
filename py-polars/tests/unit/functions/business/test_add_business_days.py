from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_series_equal

if TYPE_CHECKING:
    from polars.type_aliases import TimeUnit


def test_add_business_days() -> None:
    # (Expression, expression)
    df = pl.DataFrame(
        {
            "start": [date(2020, 1, 1), date(2020, 1, 2)],
            "n": [-1, 5],
        }
    )
    result = df.select(result=pl.col("start").dt.add_business_days("n"))["result"]
    expected = pl.Series("result", [date(2019, 12, 31), date(2020, 1, 9)], pl.Date)
    assert_series_equal(result, expected)

    # (Expression, scalar)
    result = df.select(result=pl.col("start").dt.add_business_days(5))["result"]
    expected = pl.Series("result", [date(2020, 1, 8), date(2020, 1, 9)], pl.Date)
    assert_series_equal(result, expected)

    # (Scalar, expression)
    result = df.select(
        result=pl.lit(date(2020, 1, 1), dtype=pl.Date).dt.add_business_days(pl.col("n"))
    )["result"]
    expected = pl.Series("result", [date(2019, 12, 31), date(2020, 1, 8)], pl.Date)
    assert_series_equal(result, expected)

    # (Scalar, scalar)
    result = df.select(
        result=pl.lit(date(2020, 1, 1), dtype=pl.Date).dt.add_business_days(5)
    )["result"]
    expected = pl.Series("result", [date(2020, 1, 8)], pl.Date)
    assert_series_equal(result, expected)


def test_add_business_day_w_week_mask() -> None:
    df = pl.DataFrame(
        {
            "start": [date(2020, 1, 1), date(2020, 1, 2)],
            "n": [1, 5],
        }
    )
    result = df.select(
        result=pl.col("start").dt.add_business_days(
            "n", week_mask=(True, True, True, True, True, True, False)
        )
    )["result"]
    expected = pl.Series("result", [date(2020, 1, 2), date(2020, 1, 8)])
    assert_series_equal(result, expected)

    result = df.select(
        result=pl.col("start").dt.add_business_days(
            "n", week_mask=(True, True, True, True, False, False, True)
        )
    )["result"]
    expected = pl.Series("result", [date(2020, 1, 2), date(2020, 1, 9)])
    assert_series_equal(result, expected)


def test_add_business_day_w_week_mask_invalid() -> None:
    with pytest.raises(ValueError, match=r"expected a sequence of length 7 \(got 2\)"):
        pl.col("start").dt.add_business_days("n", week_mask=(False, 0))  # type: ignore[arg-type]
    df = pl.DataFrame(
        {
            "start": [date(2020, 1, 1), date(2020, 1, 2)],
            "n": [1, 5],
        }
    )
    with pytest.raises(
        pl.ComputeError, match="`week_mask` must have at least one business day"
    ):
        df.select(pl.col("start").dt.add_business_days("n", week_mask=[False] * 7))


def test_add_business_days_schema() -> None:
    lf = pl.LazyFrame(
        {
            "start": [date(2020, 1, 1), date(2020, 1, 2)],
            "n": [1, 5],
        }
    )
    result = lf.select(
        result=pl.col("start").dt.add_business_days("n"),
    )
    assert result.schema["result"] == pl.Date
    assert result.collect().schema["result"] == pl.Date
    assert 'col("start").add_business_days([col("n")])' in result.explain()


def test_add_business_days_w_holidays() -> None:
    df = pl.DataFrame(
        {
            "start": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 2)],
            "n": [1, 5, 7],
        }
    )
    result = df.select(
        result=pl.col("start").dt.add_business_days(
            "n", holidays=[date(2020, 1, 3), date(2020, 1, 9)]
        ),
    )["result"]
    expected = pl.Series(
        "result", [date(2020, 1, 2), date(2020, 1, 13), date(2020, 1, 15)]
    )
    assert_series_equal(result, expected)
    result = df.select(
        result=pl.col("start").dt.add_business_days(
            "n", holidays=[date(2020, 1, 1), date(2020, 1, 2)], roll="backward"
        ),
    )["result"]
    expected = pl.Series(
        "result", [date(2020, 1, 3), date(2020, 1, 9), date(2020, 1, 13)]
    )
    assert_series_equal(result, expected)
    result = df.select(
        result=pl.col("start").dt.add_business_days(
            "n",
            holidays=[
                date(2019, 1, 1),
                date(2020, 1, 1),
                date(2020, 1, 2),
                date(2021, 1, 1),
            ],
            roll="backward",
        ),
    )["result"]
    expected = pl.Series(
        "result", [date(2020, 1, 3), date(2020, 1, 9), date(2020, 1, 13)]
    )
    assert_series_equal(result, expected)


def test_add_business_days_w_roll() -> None:
    df = pl.DataFrame(
        {
            "start": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 4)],
            "n": [1, 5, 7],
        }
    )
    with pytest.raises(pl.ComputeError, match="is not a business date"):
        df.select(result=pl.col("start").dt.add_business_days("n"))
    result = df.select(
        result=pl.col("start").dt.add_business_days("n", roll="forward")
    )["result"]
    expected = pl.Series(
        "result", [date(2020, 1, 2), date(2020, 1, 9), date(2020, 1, 15)]
    )
    assert_series_equal(result, expected)
    result = df.select(
        result=pl.col("start").dt.add_business_days("n", roll="backward")
    )["result"]
    expected = pl.Series(
        "result", [date(2020, 1, 2), date(2020, 1, 9), date(2020, 1, 14)]
    )
    assert_series_equal(result, expected)


@pytest.mark.parametrize("time_zone", [None, "Europe/London", "Asia/Kathmandu"])
@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_add_business_days_datetime(time_zone: str | None, time_unit: TimeUnit) -> None:
    df = pl.DataFrame(
        {"start": [datetime(2020, 3, 28, 1), datetime(2020, 1, 10, 4)]},
        schema={"start": pl.Datetime(time_unit, time_zone)},
    )
    result = df.select(
        result=pl.col("start").dt.add_business_days(2, week_mask=[True] * 7)
    )["result"]
    expected = pl.Series(
        "result",
        [datetime(2020, 3, 30, 1), datetime(2020, 1, 12, 4)],
        pl.Datetime(time_unit, time_zone),
    )
    assert_series_equal(result, expected)

    with pytest.raises(pl.ComputeError, match="is not a business date"):
        df.select(result=pl.col("start").dt.add_business_days(2))


def test_add_business_days_invalid() -> None:
    df = pl.DataFrame({"start": [timedelta(1)]})
    with pytest.raises(pl.InvalidOperationError, match="expected date or datetime"):
        df.select(result=pl.col("start").dt.add_business_days(2, week_mask=[True] * 7))
    df = pl.DataFrame({"start": [date(2020, 1, 1)]})
    with pytest.raises(
        pl.InvalidOperationError,
        match="expected Int64, Int32, UInt64, or UInt32, got f64",
    ):
        df.select(
            result=pl.col("start").dt.add_business_days(1.5, week_mask=[True] * 7)
        )
    with pytest.raises(
        ValueError,
        match="`roll` must be one of {'raise', 'forward', 'backward'}, got cabbage",
    ):
        df.select(result=pl.col("start").dt.add_business_days(1, roll="cabbage"))  # type: ignore[arg-type]


def test_add_business_days_w_nulls() -> None:
    df = pl.DataFrame(
        {
            "start": [date(2020, 3, 28), None],
            "n": [None, 2],
        },
    )
    result = df.select(result=pl.col("start").dt.add_business_days("n"))["result"]
    expected = pl.Series("result", [None, None], dtype=pl.Date)
    assert_series_equal(result, expected)

    result = df.select(
        result=pl.col("start").dt.add_business_days(pl.lit(None, dtype=pl.Int32))
    )["result"]
    assert_series_equal(result, expected)

    result = df.select(result=pl.lit(None, dtype=pl.Date).dt.add_business_days("n"))[
        "result"
    ]
    assert_series_equal(result, expected)

    result = df.select(
        result=pl.lit(None, dtype=pl.Date).dt.add_business_days(
            pl.lit(None, dtype=pl.Int32)
        )
    )["result"]
    expected = pl.Series("result", [None], dtype=pl.Date)
    assert_series_equal(result, expected)
