from __future__ import annotations

from datetime import date, datetime

import pytest

import polars as pl
from polars.exceptions import ComputeError, ShapeError
from polars.testing import assert_series_equal


@pytest.mark.parametrize(
    ("holidays", "week_mask", "expected_values"),
    [
        ((), (True, True, True, True, True, False, False), [True, True]),
        (
            (date(2020, 1, 1),),
            (True, True, True, True, True, False, False),
            [False, True],
        ),
        (
            (date(2020, 1, 2),),
            (True, True, True, True, True, False, False),
            [True, False],
        ),
        (
            (date(2020, 1, 2), date(2020, 1, 1)),
            (True, True, True, True, True, False, False),
            [False, False],
        ),
        ((), (True, True, False, True, True, True, True), [False, True]),
        ((), (True, True, False, False, True, True, True), [False, False]),
        ((), (True, True, True, False, True, True, True), [True, False]),
        (
            (),
            (True, True, True, True, True, True, True),
            [True, True],
        ),  # no holidays :sob:
        (
            (date(2020, 1, 1),),
            (True, True, True, True, True, True, True),
            [False, True],
        ),
        (
            (date(2020, 1, 1),),
            (True, True, False, True, True, True, True),
            [False, True],
        ),
    ],
)
def test_is_business_day(
    holidays: tuple[date, ...], week_mask: tuple[bool, ...], expected_values: list[bool]
) -> None:
    # Date
    df = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)]})
    result = df.select(
        pl.col("date").dt.is_business_day(holidays=holidays, week_mask=week_mask)
    )["date"]
    expected = pl.Series("date", expected_values)
    assert_series_equal(result, expected)
    # Holidays are in Series of List of Date, of length 1:
    result = df.select(
        pl.col("date").dt.is_business_day(
            holidays=pl.Series([list(holidays)], dtype=pl.List(pl.Date)),
            week_mask=week_mask,
        )
    )["date"]
    assert_series_equal(result, expected)
    # Datetime
    df = pl.DataFrame({"date": [datetime(2020, 1, 1, 3), datetime(2020, 1, 2, 1)]})
    result = df.select(
        pl.col("date").dt.is_business_day(holidays=holidays, week_mask=week_mask)
    )["date"]
    assert_series_equal(result, expected)
    # Datetime tz-aware
    df = pl.DataFrame(
        {"date": [datetime(2020, 1, 1), datetime(2020, 1, 2)]}
    ).with_columns(pl.col("date").dt.replace_time_zone("Asia/Kathmandu"))
    result = df.select(
        pl.col("date").dt.is_business_day(holidays=holidays, week_mask=week_mask)
    )["date"]
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("dates", "holidays", "expected"),
    [
        # Each date is checked against the corresponding date:
        (
            [
                date(2026, 5, 1),
                date(2026, 9, 7),
                date(2026, 5, 1),
                date(2026, 9, 7),
            ],
            [
                [date(2026, 9, 7)],
                [date(2026, 9, 7)],
                [date(2026, 5, 1)],
                [date(2026, 5, 1)],
            ],
            [True, False, False, True],
        ),
        # Null dates still advance the holidays:
        (
            [
                date(2026, 5, 1),
                None,
                date(2026, 5, 1),
                date(2026, 9, 7),
            ],
            [
                [date(2026, 9, 7)],
                [date(2026, 9, 7)],
                [date(2026, 5, 1)],
                [date(2026, 5, 1)],
            ],
            [True, None, False, True],
        ),
        # Null holiday lists result in null:
        (
            [
                date(2026, 5, 1),
                date(2026, 9, 7),
                date(2026, 5, 1),
                date(2026, 5, 1),
            ],
            [
                None,
                [date(2026, 9, 7)],
                [date(2026, 9, 7)],
                [date(2026, 5, 1)],
            ],
            [None, False, True, False],
        ),
    ],
)
def test_different_holidays_per_day(
    dates: list[date | None],
    holidays: list[list[date | None] | None],
    expected: list[bool | None],
) -> None:
    base_df = pl.DataFrame(
        {
            "date": dates,
        }
    )
    holidays = pl.Series(holidays)
    week_mask = [True] * 7
    expected = pl.Series("date", expected)

    # Check with holidays being both Series and an Expr:
    for df, holidays_expr in [
        (base_df, holidays),
        (base_df.with_columns(holidays=holidays), pl.col("holidays")),
    ]:
        result = df.select(
            pl.col("date").dt.is_business_day(
                holidays=holidays_expr, week_mask=week_mask
            )
        )["date"]
        assert_series_equal(result, expected)


def test_is_business_day_invalid() -> None:
    df = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]})
    # A 7-day weekend:
    with pytest.raises(ComputeError):
        df.select(pl.col("date").dt.is_business_day(week_mask=[False] * 7))
    # Insufficient number of holidays lists:
    holidays = pl.Series([[date(2020, 1, 1)], []])
    with pytest.raises(ShapeError):
        df.select(
            pl.col("date").dt.is_business_day(holidays=holidays, week_mask=[True] * 7)
        )
    # Holidays are not a List:
    with pytest.raises(ComputeError):
        df.select(
            pl.col("date").dt.is_business_day(
                holidays=pl.Series(["a", "b", "c"]), week_mask=[True] * 7
            )
        )
    # Holidays are a List of something that is not a Date:
    with pytest.raises(ComputeError):
        df.select(
            pl.col("date").dt.is_business_day(
                holidays=pl.Series([["a"], ["b"], ["c"]]), week_mask=[True] * 7
            )
        )
    # List of holidays contains a null:
    with pytest.raises(ComputeError, match="list of holidays contained a null"):
        df.select(
            pl.col("date").dt.is_business_day(
                holidays=pl.Series([[], [None], [date(2025, 1, 1)]]),
                week_mask=[True] * 7,
            )
        )


def test_is_business_day_repr() -> None:
    assert "is_business_day" in repr(pl.col("date").dt.is_business_day())
