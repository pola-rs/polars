from __future__ import annotations

import os
from datetime import date, datetime

import pytest

import polars as pl
from polars.exceptions import ComputeError, InvalidOperationError, ShapeError
from polars.testing import assert_frame_equal, assert_series_equal


def test_date_range() -> None:
    # if low/high are both date, range is also be date _iff_ the granularity is >= 1d
    result = pl.date_range(date(2022, 1, 1), date(2022, 3, 1), "1mo", eager=True)
    assert result.to_list() == [date(2022, 1, 1), date(2022, 2, 1), date(2022, 3, 1)]


def test_date_range_invalid_time_unit() -> None:
    with pytest.raises(InvalidOperationError, match="'x' not supported"):
        pl.date_range(
            start=date(2021, 12, 16),
            end=date(2021, 12, 18),
            interval="1X",
            eager=True,
        )


def test_date_range_end_of_month_5441() -> None:
    result = pl.date_range(
        start=date(2020, 1, 31),
        end=date(2020, 3, 31),
        interval="1mo",
        closed="both",
        eager=True,
    )
    expected = pl.Series(
        "literal", [date(2020, 1, 31), date(2020, 2, 29), date(2020, 3, 31)]
    )
    assert_series_equal(result, expected)


def test_date_range_name() -> None:
    result_eager = pl.date_range(
        start=date(2020, 1, 1), end=date(2020, 1, 3), eager=True
    )
    assert result_eager.name == "literal"

    start = pl.Series("left", [date(2020, 1, 1)])
    result_lazy = pl.select(
        pl.date_range(pl.lit(start).first(), date(2020, 1, 3), eager=False)
    ).to_series()
    assert result_lazy.name == "left"


def test_date_range_eager() -> None:
    result = pl.date_range(start=date(2022, 1, 1), end=date(2022, 1, 3), eager=True)
    expected = pl.Series(
        "literal", [date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3)]
    )
    assert_series_equal(result, expected)


def test_date_range_input_shape_empty() -> None:
    empty = pl.Series(dtype=pl.Datetime)
    single = pl.Series([datetime(2022, 1, 2)])

    with pytest.raises(ShapeError):
        pl.date_range(start=empty, end=single, eager=True)
    with pytest.raises(ShapeError):
        pl.date_range(start=single, end=empty, eager=True)
    with pytest.raises(ShapeError):
        pl.date_range(start=empty, end=empty, eager=True)


def test_date_range_input_shape_multiple_values() -> None:
    single = pl.Series([datetime(2022, 1, 2)])
    multiple = pl.Series([datetime(2022, 1, 3), datetime(2022, 1, 4)])

    with pytest.raises(ShapeError):
        pl.date_range(start=multiple, end=single, eager=True)
    with pytest.raises(ShapeError):
        pl.date_range(start=single, end=multiple, eager=True)
    with pytest.raises(ShapeError):
        pl.date_range(start=multiple, end=multiple, eager=True)


def test_date_range_start_later_than_end() -> None:
    result = pl.date_range(start=date(2000, 3, 20), end=date(2000, 3, 5), eager=True)
    expected = pl.Series("literal", dtype=pl.Date)
    assert_series_equal(result, expected)


def test_date_range_24h_interval_raises() -> None:
    with pytest.raises(
        ComputeError,
        match="`interval` input for `date_range` must consist of full days",
    ):
        pl.date_range(
            start=date(2022, 1, 1), end=date(2022, 1, 3), interval="24h", eager=True
        )


def test_long_date_range_12461() -> None:
    morsel_size_env = os.environ.get("POLARS_IDEAL_MORSEL_SIZE")
    if morsel_size_env is not None and int(morsel_size_env) < 1000:
        pytest.skip("test is too slow for small morsel sizes")
    result = pl.date_range(
        start=date(1900, 1, 1), end=date(2300, 1, 1), interval="1d", eager=True
    )
    assert result[0] == date(1900, 1, 1)
    assert result[-1] == date(2300, 1, 1)
    assert (result.diff()[1:].dt.total_days() == 1).all()


def test_date_range_datetime_input() -> None:
    result = pl.date_range(
        start=datetime(2022, 1, 1, 12),
        end=datetime(2022, 1, 3),
        interval="1d",
        eager=True,
    )
    expected = pl.Series(
        "literal", [date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3)]
    )
    assert_series_equal(result, expected)


def test_date_ranges_datetime_input() -> None:
    result = pl.date_ranges(
        start=datetime(2022, 1, 1, 12),
        end=datetime(2022, 1, 3),
        interval="1d",
        eager=True,
    )
    expected = pl.Series(
        "literal", [[date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3)]]
    )
    assert_series_equal(result, expected)


def test_date_range_with_subclass_18470_18447() -> None:
    class MyAmazingDate(date):
        pass

    class MyAmazingDatetime(datetime):
        pass

    result = pl.datetime_range(
        start=MyAmazingDate(2020, 1, 1), end=MyAmazingDatetime(2020, 1, 2), eager=True
    )
    expected = pl.Series("literal", [datetime(2020, 1, 1), datetime(2020, 1, 2)])
    assert_series_equal(result, expected)


# start/end/interval
def test_date_range_start_end_interval_forwards() -> None:
    start = date(2025, 1, 1)
    end = date(2025, 1, 10)

    assert_series_equal(
        pl.date_range(start=start, end=end, interval="3d", closed="left", eager=True),
        pl.Series("literal", [date(2025, 1, 1), date(2025, 1, 4), date(2025, 1, 7)]),
    )
    assert_series_equal(
        pl.date_range(start=start, end=end, interval="3d", closed="right", eager=True),
        pl.Series("literal", [date(2025, 1, 4), date(2025, 1, 7), date(2025, 1, 10)]),
    )
    assert_series_equal(
        pl.date_range(start=start, end=end, interval="3d", closed="none", eager=True),
        pl.Series("literal", [date(2025, 1, 4), date(2025, 1, 7)]),
    )
    assert_series_equal(
        pl.date_range(start=start, end=end, interval="3d", closed="both", eager=True),
        pl.Series(
            "literal",
            [date(2025, 1, 1), date(2025, 1, 4), date(2025, 1, 7), date(2025, 1, 10)],
        ),
    )
    # test wrong direction is empty
    assert_series_equal(
        pl.date_range(start=end, end=start, interval="3d", eager=True),
        pl.Series("literal", [], dtype=pl.Date),
    )


def test_date_range_expr_scalar() -> None:
    df = pl.DataFrame(
        {
            "a": [date(2025, 1, 3), date(2025, 1, 1)],
            "interval": ["1d", "2d"],
        }
    )
    result = df.select(
        forward_start_end_interval=pl.date_range(
            start=pl.col("a").min(), end=pl.col("a").max(), interval="1d"
        ),
    )
    forward = [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)]
    expected = pl.DataFrame(
        {
            "forward_start_end_interval": forward,
        }
    )
    assert_frame_equal(result, expected)
