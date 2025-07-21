from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.exceptions import ComputeError, InvalidOperationError, ShapeError
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from polars._typing import ClosedInterval


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


def test_date_range_start_end_interval_backwards() -> None:
    start = date(2025, 1, 10)
    end = date(2025, 1, 1)

    assert_series_equal(
        pl.date_range(start=start, end=end, interval="-3d", closed="left", eager=True),
        pl.Series("literal", [date(2025, 1, 10), date(2025, 1, 7), date(2025, 1, 4)]),
    )
    assert_series_equal(
        pl.date_range(start=start, end=end, interval="-3d", closed="right", eager=True),
        pl.Series("literal", [date(2025, 1, 7), date(2025, 1, 4), date(2025, 1, 1)]),
    )
    assert_series_equal(
        pl.date_range(start=start, end=end, interval="-3d", closed="none", eager=True),
        pl.Series("literal", [date(2025, 1, 7), date(2025, 1, 4)]),
    )
    assert_series_equal(
        pl.date_range(start=start, end=end, interval="-3d", closed="both", eager=True),
        pl.Series(
            "literal",
            [date(2025, 1, 10), date(2025, 1, 7), date(2025, 1, 4), date(2025, 1, 1)],
        ),
    )
    # test wrong direction is empty
    assert_series_equal(
        pl.date_range(start=end, end=start, interval="-3d", eager=True),
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
        backward_start_end_interval=pl.date_range(
            start=pl.col("a").max(), end=pl.col("a").min(), interval="-1d"
        ),
        forward_start_end_samples=pl.date_range(
            start=pl.col("a").min(),
            end=pl.col("a").max(),
            num_samples=3,
        ),
        backward_start_end_samples=pl.date_range(
            start=pl.col("a").max(),
            end=pl.col("a").min(),
            num_samples=3,
        ),
        forward_start_interval_samples=pl.date_range(
            start=pl.col("a").min(),
            interval="1d",
            num_samples=3,
        ),
        backward_start_interval_samples=pl.date_range(
            start=pl.col("a").max(),
            interval="-1d",
            num_samples=3,
        ),
        forward_end_interval_samples=pl.date_range(
            end=pl.col("a").max(),
            interval="1d",
            num_samples=3,
        ),
        backward_end_interval_samples=pl.date_range(
            end=pl.col("a").min(),
            interval="-1d",
            num_samples=3,
        ),
    )
    forward = [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)]
    backward = forward[-1::-1]
    expected = pl.DataFrame(
        {
            "forward_start_end_interval": forward,
            "backward_start_end_interval": backward,
            "forward_start_end_samples": forward,
            "backward_start_end_samples": backward,
            "forward_start_interval_samples": forward,
            "backward_start_interval_samples": backward,
            "forward_end_interval_samples": forward,
            "backward_end_interval_samples": backward,
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("closed", ["left", "right", "none"])
def test_date_range_start_end_samples_invalid_closure(closed: ClosedInterval) -> None:
    msg = (
        "date_range does not support 'left', 'right', or 'none' for the 'closed' "
        "parameter when 'start', 'end', and 'num_samples' is provided."
    )
    with pytest.raises(InvalidOperationError, match=msg):
        pl.date_range(
            start=date(2025, 1, 1), end=date(2025, 1, 10), num_samples=3, closed=closed
        )


def test_date_range_start_end_samples() -> None:
    assert_series_equal(
        pl.date_range(
            start=date(2025, 1, 1), end=date(2025, 1, 10), num_samples=3, eager=True
        ),
        pl.Series("literal", [date(2025, 1, 1), date(2025, 1, 5), date(2025, 1, 10)]),
    )

    assert_series_equal(
        pl.date_range(
            start=date(2025, 1, 1), end=date(2025, 1, 5), num_samples=10, eager=True
        ),
        pl.Series(
            "literal",
            [
                date(2025, 1, 1),
                date(2025, 1, 1),
                date(2025, 1, 1),
                date(2025, 1, 2),
                date(2025, 1, 2),
                date(2025, 1, 3),
                date(2025, 1, 3),
                date(2025, 1, 4),
                date(2025, 1, 4),
                date(2025, 1, 5),
            ],
        ),
    )

    assert_series_equal(
        pl.date_range(
            start=date(2025, 1, 10), end=date(2025, 1, 1), num_samples=3, eager=True
        ),
        pl.Series("literal", [date(2025, 1, 10), date(2025, 1, 5), date(2025, 1, 1)]),
    )

    assert_series_equal(
        pl.date_range(
            start=date(2025, 1, 5), end=date(2025, 1, 1), num_samples=10, eager=True
        ),
        pl.Series(
            "literal",
            [
                date(2025, 1, 5),
                date(2025, 1, 4),
                date(2025, 1, 4),
                date(2025, 1, 3),
                date(2025, 1, 3),
                date(2025, 1, 2),
                date(2025, 1, 2),
                date(2025, 1, 1),
                date(2025, 1, 1),
                date(2025, 1, 1),
            ],
        ),
    )

    assert_series_equal(
        pl.date_range(
            start=date(2025, 1, 1), end=date(2025, 1, 5), num_samples=0, eager=True
        ),
        pl.Series("literal", [], dtype=pl.Date),
    )


# -- start/interval/samples
@pytest.mark.parametrize(
    ("closed", "expected"),
    [
        ("both", [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)]),
        ("left", [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)]),
        ("right", [date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 4)]),
        ("none", [date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 4)]),
    ],
)
def test_date_range_start_interval_samples_forward_1d(
    closed: ClosedInterval, expected: list[date]
) -> None:
    result = pl.date_range(
        start=date(2025, 1, 1), interval="1d", num_samples=3, closed=closed, eager=True
    )
    assert_series_equal(result, pl.Series("literal", expected))


@pytest.mark.parametrize(
    ("closed", "expected"),
    [
        ("both", [date(2025, 1, 3), date(2025, 1, 2), date(2025, 1, 1)]),
        ("left", [date(2025, 1, 3), date(2025, 1, 2), date(2025, 1, 1)]),
        ("right", [date(2025, 1, 2), date(2025, 1, 1), date(2024, 12, 31)]),
        ("none", [date(2025, 1, 2), date(2025, 1, 1), date(2024, 12, 31)]),
    ],
)
def test_date_range_start_interval_samples_backward_1d(
    closed: ClosedInterval, expected: list[date]
) -> None:
    result = pl.date_range(
        start=date(2025, 1, 3), interval="-1d", num_samples=3, closed=closed, eager=True
    )
    assert_series_equal(result, pl.Series("literal", expected))


@pytest.mark.parametrize(
    ("closed", "expected"),
    [
        ("both", [date(2025, 1, 1), date(2025, 1, 3), date(2025, 1, 5)]),
        ("left", [date(2025, 1, 1), date(2025, 1, 3), date(2025, 1, 5)]),
        ("right", [date(2025, 1, 3), date(2025, 1, 5), date(2025, 1, 7)]),
        ("none", [date(2025, 1, 3), date(2025, 1, 5), date(2025, 1, 7)]),
    ],
)
def test_date_range_start_interval_samples_forward_2d(
    closed: ClosedInterval, expected: list[date]
) -> None:
    result = pl.date_range(
        start=date(2025, 1, 1), interval="2d", num_samples=3, closed=closed, eager=True
    )
    assert_series_equal(result, pl.Series("literal", expected))


@pytest.mark.parametrize(
    ("closed", "expected"),
    [
        ("both", [date(2025, 1, 5), date(2025, 1, 3), date(2025, 1, 1)]),
        ("left", [date(2025, 1, 5), date(2025, 1, 3), date(2025, 1, 1)]),
        ("right", [date(2025, 1, 3), date(2025, 1, 1), date(2024, 12, 30)]),
        ("none", [date(2025, 1, 3), date(2025, 1, 1), date(2024, 12, 30)]),
    ],
)
def test_date_range_start_interval_samples_backward_2d(
    closed: ClosedInterval, expected: list[date]
) -> None:
    result = pl.date_range(
        start=date(2025, 1, 5), interval="-2d", num_samples=3, closed=closed, eager=True
    )
    assert_series_equal(result, pl.Series("literal", expected))


@pytest.mark.parametrize(
    ("closed", "expected"),
    [
        ("both", [date(2025, 1, 31), date(2025, 2, 28), date(2025, 3, 31)]),
        ("left", [date(2025, 1, 31), date(2025, 2, 28), date(2025, 3, 31)]),
        ("right", [date(2025, 2, 28), date(2025, 3, 31), date(2025, 4, 30)]),
        ("none", [date(2025, 2, 28), date(2025, 3, 31), date(2025, 4, 30)]),
    ],
)
def test_date_range_start_interval_samples_forward_1mo(
    closed: ClosedInterval, expected: list[date]
) -> None:
    result = pl.date_range(
        start=date(2025, 1, 31),
        interval="1mo",
        num_samples=3,
        closed=closed,
        eager=True,
    )
    assert_series_equal(result, pl.Series("literal", expected))


@pytest.mark.parametrize(
    ("closed", "expected"),
    [
        ("both", [date(2025, 3, 31), date(2025, 2, 28), date(2025, 1, 31)]),
        ("left", [date(2025, 3, 31), date(2025, 2, 28), date(2025, 1, 31)]),
        ("right", [date(2025, 2, 28), date(2025, 1, 31), date(2024, 12, 31)]),
        ("none", [date(2025, 2, 28), date(2025, 1, 31), date(2024, 12, 31)]),
    ],
)
def test_date_range_start_interval_samples_backward_1mo(
    closed: ClosedInterval, expected: list[date]
) -> None:
    result = pl.date_range(
        start=date(2025, 3, 31),
        interval="-1mo",
        num_samples=3,
        closed=closed,
        eager=True,
    )
    assert_series_equal(result, pl.Series("literal", expected))


# -- end/interval/samples
@pytest.mark.parametrize(
    ("closed", "expected"),
    [
        ("both", [date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 4)]),
        ("left", [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)]),
        ("right", [date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 4)]),
        ("none", [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)]),
    ],
)
def test_date_range_end_interval_samples_forward_1d(
    closed: ClosedInterval, expected: list[date]
) -> None:
    result = pl.date_range(
        end=date(2025, 1, 4), interval="1d", num_samples=3, closed=closed, eager=True
    )
    assert_series_equal(result, pl.Series("literal", expected))


@pytest.mark.parametrize(
    ("closed", "expected"),
    [
        ("both", [date(2025, 1, 3), date(2025, 1, 2), date(2025, 1, 1)]),
        ("left", [date(2025, 1, 4), date(2025, 1, 3), date(2025, 1, 2)]),
        ("right", [date(2025, 1, 3), date(2025, 1, 2), date(2025, 1, 1)]),
        ("none", [date(2025, 1, 4), date(2025, 1, 3), date(2025, 1, 2)]),
    ],
)
def test_date_range_end_interval_samples_backward_1d(
    closed: ClosedInterval, expected: list[date]
) -> None:
    result = pl.date_range(
        end=date(2025, 1, 1), interval="-1d", num_samples=3, closed=closed, eager=True
    )
    assert_series_equal(result, pl.Series("literal", expected))


@pytest.mark.parametrize(
    ("closed", "expected"),
    [
        ("both", [date(2025, 1, 1), date(2025, 1, 3), date(2025, 1, 5)]),
        ("left", [date(2024, 12, 30), date(2025, 1, 1), date(2025, 1, 3)]),
        ("right", [date(2025, 1, 1), date(2025, 1, 3), date(2025, 1, 5)]),
        ("none", [date(2024, 12, 30), date(2025, 1, 1), date(2025, 1, 3)]),
    ],
)
def test_date_range_end_interval_samples_forward_2d(
    closed: ClosedInterval, expected: list[date]
) -> None:
    result = pl.date_range(
        end=date(2025, 1, 5), interval="2d", num_samples=3, closed=closed, eager=True
    )
    assert_series_equal(result, pl.Series("literal", expected))


@pytest.mark.parametrize(
    ("closed", "expected"),
    [
        ("both", [date(2025, 1, 5), date(2025, 1, 3), date(2025, 1, 1)]),
        ("left", [date(2025, 1, 7), date(2025, 1, 5), date(2025, 1, 3)]),
        ("right", [date(2025, 1, 5), date(2025, 1, 3), date(2025, 1, 1)]),
        ("none", [date(2025, 1, 7), date(2025, 1, 5), date(2025, 1, 3)]),
    ],
)
def test_date_range_end_interval_samples_backward_2d(
    closed: ClosedInterval, expected: list[date]
) -> None:
    result = pl.date_range(
        end=date(2025, 1, 1), interval="-2d", num_samples=3, closed=closed, eager=True
    )
    assert_series_equal(result, pl.Series("literal", expected))


@pytest.mark.parametrize(
    ("closed", "expected"),
    [
        ("both", [date(2025, 1, 31), date(2025, 2, 28), date(2025, 3, 31)]),
        ("left", [date(2024, 12, 31), date(2025, 1, 31), date(2025, 2, 28)]),
        ("right", [date(2025, 1, 31), date(2025, 2, 28), date(2025, 3, 31)]),
        ("none", [date(2024, 12, 31), date(2025, 1, 31), date(2025, 2, 28)]),
    ],
)
def test_date_range_end_interval_samples_forward_1mo(
    closed: ClosedInterval, expected: list[date]
) -> None:
    result = pl.date_range(
        end=date(2025, 3, 31), interval="1mo", num_samples=3, closed=closed, eager=True
    )
    assert_series_equal(result, pl.Series("literal", expected))


@pytest.mark.parametrize(
    ("closed", "expected"),
    [
        ("both", [date(2025, 3, 31), date(2025, 2, 28), date(2025, 1, 31)]),
        ("left", [date(2025, 4, 30), date(2025, 3, 31), date(2025, 2, 28)]),
        ("right", [date(2025, 3, 31), date(2025, 2, 28), date(2025, 1, 31)]),
        ("none", [date(2025, 4, 30), date(2025, 3, 31), date(2025, 2, 28)]),
    ],
)
def test_date_range_end_interval_samples_backward_1mo(
    closed: ClosedInterval, expected: list[date]
) -> None:
    result = pl.date_range(
        end=date(2025, 1, 31),
        interval="-1mo",
        num_samples=3,
        closed=closed,
        eager=True,
    )
    assert_series_equal(result, pl.Series("literal", expected))


@pytest.mark.parametrize("closed", ["left", "right", "none"])
def test_date_range_start_end_samples_notclosed(closed: ClosedInterval) -> None:
    with pytest.raises(
        InvalidOperationError,
        match=(
            "date_range does not support 'left', 'right', or 'none' for the 'closed' "
            "parameter when 'start', 'end', and 'num_samples' is provided."
        ),
    ):
        pl.date_range(
            start=date(2025, 1, 1), end=date(2025, 1, 2), num_samples=3, closed=closed
        )
