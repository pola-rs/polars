from datetime import date

import hypothesis.strategies as st
import pytest
from hypothesis import given

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.testing import assert_frame_equal, assert_series_equal


def test_from_year_week_day_iso_year_boundaries() -> None:
    df = pl.DataFrame(
        {
            "year": [2015, 2019, 2020, 2020, 2021],
            "week": [1, 1, 53, 53, 1],
            "day": [1, 1, 1, 7, 1],
        },
        schema={"year": pl.UInt16, "week": pl.UInt8, "day": pl.Int8},
    )
    result = df.lazy().select(pl.from_year_week_day("year", "week", "day"))
    expected = pl.DataFrame(
        {
            "date": [
                date(2014, 12, 29),
                date(2018, 12, 31),
                date(2020, 12, 28),
                date(2021, 1, 3),
                date(2021, 1, 4),
            ]
        }
    )
    assert result.collect_schema() == {"date": pl.Date}
    assert_frame_equal(result.collect(), expected)


def test_from_year_week_day_expressions() -> None:
    df = pl.DataFrame({"year": [2019, 2020], "week": [0, 1], "day": [0, 6]})
    result = df.select(
        pl.from_year_week_day(pl.col("year") + 1, pl.col("week") + 1, pl.col("day") + 1)
    )
    expected = pl.DataFrame({"date": [date(2019, 12, 30), date(2021, 1, 17)]})
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("component", "values", "expected"),
    [
        (0, [2019, 2020], [date(2018, 12, 31), date(2019, 12, 30)]),
        (1, [1, 53], [date(2019, 12, 30), date(2020, 12, 28)]),
        (2, [1, 7], [date(2019, 12, 30), date(2020, 1, 5)]),
    ],
)
def test_from_year_week_day_expansion(
    component: int, values: list[int], expected: list[date]
) -> None:
    df = pl.DataFrame({"a": values[:1], "b": values[1:]})
    components: list[int | pl.Expr] = [2020, 1, 1]
    components[component] = pl.all()
    result = df.select(pl.from_year_week_day(*components).name.keep())
    assert_frame_equal(result, pl.DataFrame({"a": expected[:1], "b": expected[1:]}))


@pytest.mark.parametrize("scalar", ["year", "week", "day"])
def test_from_year_week_day_broadcast(scalar: str) -> None:
    values = {"year": [2019, 2020], "week": [1, 2], "day": [1, 7]}
    values[scalar] = [values[scalar][0]] * 2
    components = [values[name][0] if name == scalar else name for name in values]
    result = pl.DataFrame(values).select(pl.from_year_week_day(*components))
    expected = pl.DataFrame(
        {
            "date": [
                date.fromisocalendar(*row) for row in zip(*values.values(), strict=True)
            ]
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("year", "week", "day"),
    [
        (2021, 53, 1),
        (2020, 0, 1),
        (2020, 54, 1),
        (2020, 1, 0),
        (2020, 1, 8),
    ],
)
def test_from_year_week_day_invalid(year: int, week: int, day: int) -> None:
    with pytest.raises(InvalidOperationError):
        pl.select(pl.from_year_week_day(year, week, day))


def test_from_year_week_day_nulls() -> None:
    df = pl.DataFrame(
        {
            "year": [None, 2020, 2020, None, 2020],
            "week": [54, None, 54, None, 53],
            "day": [8, 8, None, None, 7],
        }
    )
    result = df.select(pl.from_year_week_day("year", "week", "day"))
    expected = pl.DataFrame({"date": [None, None, None, None, date(2021, 1, 3)]})
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("null_index", [0, 1, 2])
def test_from_year_week_day_null_literal(null_index: int) -> None:
    components = [pl.lit(2020), pl.lit(53), pl.lit(7)]
    components[null_index] = pl.lit(None)
    result = pl.select(pl.from_year_week_day(*components))
    assert_frame_equal(result, pl.DataFrame({"date": [None]}, schema={"date": pl.Date}))


def test_from_year_week_day_all_nulls() -> None:
    df = pl.DataFrame({"year": [None, None], "week": [None, None], "day": [None, None]})
    result = df.select(pl.from_year_week_day("year", "week", "day"))
    expected = pl.DataFrame({"date": [None, None]}, schema={"date": pl.Date})
    assert_frame_equal(result, expected)


def test_from_year_week_day_empty() -> None:
    df = pl.DataFrame(schema={"year": pl.Int32})
    result = df.select(pl.from_year_week_day("year", 1, 1))
    assert_frame_equal(result, pl.DataFrame(schema={"date": pl.Date}))


@pytest.mark.parametrize("value", [date.min, date.max, date(2000, 2, 29)])
def test_from_year_week_day_literals(value: date) -> None:
    result = pl.select(pl.from_year_week_day(*value.isocalendar())).to_series()
    assert_series_equal(result, pl.Series("date", [value]))


def test_from_year_week_day_outside_python_date_range() -> None:
    df = pl.DataFrame({"year": [0, -1, -10_000, 10_000]})
    result = df.select(pl.from_year_week_day("year", 1, 1).dt.to_string())
    expected = pl.DataFrame(
        {"date": ["0000-01-03", "-0001-01-04", "-10000-01-03", "+10000-01-03"]}
    )
    assert_frame_equal(result, expected)


@given(values=st.lists(st.dates(), min_size=1, max_size=20))
def test_from_year_week_day_round_trip(values: list[date]) -> None:
    df = pl.DataFrame({"original": values})
    original = pl.col("original")
    result = df.select(
        pl.from_year_week_day(
            original.dt.iso_year(), original.dt.week(), original.dt.weekday()
        )
    ).to_series()
    assert_series_equal(result, df["original"].rename("date"))
