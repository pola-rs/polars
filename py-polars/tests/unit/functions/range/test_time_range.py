from __future__ import annotations

from datetime import time, timedelta
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from polars._typing import ClosedInterval


def test_time_range_schema() -> None:
    df = pl.DataFrame({"start": [time(1)], "end": [time(1, 30)]}).lazy()
    result = df.with_columns(time_range=pl.time_ranges(pl.col("start"), pl.col("end")))
    expected_schema = {"start": pl.Time, "end": pl.Time, "time_range": pl.List(pl.Time)}
    assert result.collect_schema() == expected_schema
    assert result.collect().schema == expected_schema


def test_time_ranges_eager() -> None:
    start = pl.Series("start", [time(9, 0), time(10, 0)])
    end = pl.Series("end", [time(12, 0), time(11, 0)])

    result = pl.time_ranges(start, end, eager=True)

    expected = pl.Series(
        "start",
        [
            [time(9, 0), time(10, 0), time(11, 0), time(12, 0)],
            [time(10, 0), time(11, 0)],
        ],
    )
    assert_series_equal(result, expected)


def test_time_range_eager_explode() -> None:
    start = pl.Series("start", [time(9, 0)])
    end = pl.Series("end", [time(11, 0)])

    result = pl.time_range(start, end, eager=True)

    expected = pl.Series("start", [time(9, 0), time(10, 0), time(11, 0)])
    assert_series_equal(result, expected)


def test_time_range_input_shape_empty() -> None:
    empty = pl.Series(dtype=pl.Time)
    single = pl.Series([time(12, 0)])

    with pytest.raises(
        ComputeError, match="`start` must contain exactly one value, got 0 values"
    ):
        pl.time_range(empty, single, eager=True)
    with pytest.raises(
        ComputeError, match="`end` must contain exactly one value, got 0 values"
    ):
        pl.time_range(single, empty, eager=True)
    with pytest.raises(
        ComputeError, match="`start` must contain exactly one value, got 0 values"
    ):
        pl.time_range(empty, empty, eager=True)


def test_time_range_input_shape_multiple_values() -> None:
    single = pl.Series([time(12, 0)])
    multiple = pl.Series([time(11, 0), time(12, 0)])

    with pytest.raises(
        ComputeError, match="`start` must contain exactly one value, got 2 values"
    ):
        pl.time_range(multiple, single, eager=True)
    with pytest.raises(
        ComputeError, match="`end` must contain exactly one value, got 2 values"
    ):
        pl.time_range(single, multiple, eager=True)
    with pytest.raises(
        ComputeError, match="`start` must contain exactly one value, got 2 values"
    ):
        pl.time_range(multiple, multiple, eager=True)


def test_time_range_start_equals_end() -> None:
    t = time(12, 0)

    result = pl.time_range(t, t, closed="both", eager=True)

    expected = pl.Series("literal", [t])
    assert_series_equal(result, expected)


@pytest.mark.parametrize("closed", ["left", "right", "none"])
def test_time_range_start_equals_end_open(closed: ClosedInterval) -> None:
    t = time(12, 0)

    result = pl.time_range(t, t, closed=closed, eager=True)

    expected = pl.Series("literal", dtype=pl.Time)
    assert_series_equal(result, expected)


def test_time_range_start_later_than_end() -> None:
    result = pl.time_range(time(12), time(11), eager=True)
    expected = pl.Series("literal", dtype=pl.Time)
    assert_series_equal(result, expected)


@pytest.mark.parametrize("interval", [timedelta(0), timedelta(minutes=-10)])
def test_time_range_invalid_step(interval: timedelta) -> None:
    with pytest.raises(ComputeError, match="`interval` must be positive"):
        pl.time_range(time(11), time(12), interval=interval, eager=True)


def test_time_range_lit_lazy() -> None:
    tm = pl.select(
        pl.time_range(
            start=time(1, 2, 3),
            end=time(23, 59, 59),
            interval="5h45m10s333ms",
            closed="right",
        ).alias("tm")
    )

    assert tm["tm"].to_list() == [
        time(6, 47, 13, 333000),
        time(12, 32, 23, 666000),
        time(18, 17, 33, 999000),
    ]

    # validate unset start/end
    tm = pl.select(pl.time_range(interval="5h45m10s333ms").alias("tm"))
    assert tm["tm"].to_list() == [
        time(0, 0),
        time(5, 45, 10, 333000),
        time(11, 30, 20, 666000),
        time(17, 15, 30, 999000),
        time(23, 0, 41, 332000),
    ]

    tm = pl.select(
        pl.time_range(start=pl.lit(time(23, 59, 59, 999980)), interval="10000ns").alias(
            "tm"
        )
    )
    assert tm["tm"].to_list() == [
        time(23, 59, 59, 999980),
        time(23, 59, 59, 999990),
    ]


def test_time_range_lit_eager() -> None:
    eager = True
    tm = pl.select(
        pl.time_range(
            start=time(1, 2, 3),
            end=time(23, 59, 59),
            interval="5h45m10s333ms",
            closed="right",
            eager=eager,
        ).alias("tm")
    )
    if not eager:
        tm = tm.select(pl.col("tm").explode())
    assert tm["tm"].to_list() == [
        time(6, 47, 13, 333000),
        time(12, 32, 23, 666000),
        time(18, 17, 33, 999000),
    ]

    # validate unset start/end
    tm = pl.select(
        pl.time_range(
            interval="5h45m10s333ms",
            eager=eager,
        ).alias("tm")
    )
    if not eager:
        tm = tm.select(pl.col("tm").explode())
    assert tm["tm"].to_list() == [
        time(0, 0),
        time(5, 45, 10, 333000),
        time(11, 30, 20, 666000),
        time(17, 15, 30, 999000),
        time(23, 0, 41, 332000),
    ]

    tm = pl.select(
        pl.time_range(
            start=pl.lit(time(23, 59, 59, 999980)),
            interval="10000ns",
            eager=eager,
        ).alias("tm")
    )
    tm = tm.select(pl.col("tm").explode())
    assert tm["tm"].to_list() == [
        time(23, 59, 59, 999980),
        time(23, 59, 59, 999990),
    ]


def test_time_range_expr() -> None:
    df = pl.DataFrame(
        {
            "start": pl.time_range(interval="6h", eager=True),
            "stop": pl.time_range(start=time(2, 59), interval="5h59m", eager=True),
        }
    ).with_columns(intervals=pl.time_ranges("start", pl.col("stop"), interval="1h29m"))
    # shape: (4, 3)
    # ┌──────────┬──────────┬────────────────────────────────┐
    # │ start    ┆ stop     ┆ intervals                      │
    # │ ---      ┆ ---      ┆ ---                            │
    # │ time     ┆ time     ┆ list[time]                     │
    # ╞══════════╪══════════╪════════════════════════════════╡
    # │ 00:00:00 ┆ 02:59:00 ┆ [00:00:00, 01:29:00, 02:58:00] │
    # │ 06:00:00 ┆ 08:58:00 ┆ [06:00:00, 07:29:00, 08:58:00] │
    # │ 12:00:00 ┆ 14:57:00 ┆ [12:00:00, 13:29:00]           │
    # │ 18:00:00 ┆ 20:56:00 ┆ [18:00:00, 19:29:00]           │
    # └──────────┴──────────┴────────────────────────────────┘
    assert df.rows() == [
        (time(0, 0), time(2, 59), [time(0, 0), time(1, 29), time(2, 58)]),
        (time(6, 0), time(8, 58), [time(6, 0), time(7, 29), time(8, 58)]),
        (time(12, 0), time(14, 57), [time(12, 0), time(13, 29)]),
        (time(18, 0), time(20, 56), [time(18, 0), time(19, 29)]),
    ]


def test_time_range_name() -> None:
    expected_name = "literal"
    result_eager = pl.time_range(time(10), time(12), eager=True)
    assert result_eager.name == expected_name

    expected_name = "s1"
    result_lazy = pl.select(
        pl.time_range(
            pl.Series("s1", [time(10)]), pl.Series("s2", [time(12)]), eager=False
        )
    ).to_series()
    assert result_lazy.name == expected_name


def test_time_ranges_broadcasting() -> None:
    df = pl.DataFrame({"time": [time(10, 0), time(11, 0), time(12, 0)]})
    result = df.select(
        pl.time_ranges(start="time", end=time(12, 0)).alias("end"),
        pl.time_ranges(start=time(10, 0), end="time").alias("start"),
    )
    expected = pl.DataFrame(
        {
            "end": [
                [time(10, 0), time(11, 0), time(12, 0)],
                [time(11, 0), time(12, 0)],
                [time(12, 0)],
            ],
            "start": [
                [time(10, 0)],
                [time(10, 0), time(11, 0)],
                [time(10, 0), time(11, 0), time(12, 0)],
            ],
        }
    )
    assert_frame_equal(result, expected)


def test_time_ranges_mismatched_chunks() -> None:
    s1 = pl.Series("s1", [time(10), time(11)])
    s1.append(pl.Series([time(12)]))

    s2 = pl.Series("s2", [time(12)])
    s2.append(pl.Series([time(12), time(12)]))

    result = pl.time_ranges(s1, s2, eager=True)
    expected = pl.Series(
        "s1",
        [
            [time(10, 0), time(11, 0), time(12, 0)],
            [time(11, 0), time(12, 0)],
            [time(12, 0)],
        ],
    )
    assert_series_equal(result, expected)
