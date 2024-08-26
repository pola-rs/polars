from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from datetime import timezone

    from zoneinfo import ZoneInfo

    from polars._typing import FillNullStrategy, PolarsIntegerType
else:
    from polars._utils.convert import string_to_zoneinfo as ZoneInfo


@pytest.mark.parametrize(
    ("time_zone", "tzinfo"),
    [
        (None, None),
        ("Europe/Warsaw", ZoneInfo("Europe/Warsaw")),
    ],
)
def test_upsample(time_zone: str | None, tzinfo: ZoneInfo | timezone | None) -> None:
    df = pl.DataFrame(
        {
            "time": [
                datetime(2021, 2, 1),
                datetime(2021, 4, 1),
                datetime(2021, 5, 1),
                datetime(2021, 6, 1),
            ],
            "admin": ["Åland", "Netherlands", "Åland", "Netherlands"],
            "test2": [0, 1, 2, 3],
        }
    ).with_columns(pl.col("time").dt.replace_time_zone(time_zone).set_sorted())

    up = df.upsample(
        time_column="time",
        every="1mo",
        group_by="admin",
        maintain_order=True,
    ).select(pl.all().forward_fill())

    # this print will panic if timezones feature is not activated
    # don't remove
    print(up)

    expected = pl.DataFrame(
        {
            "time": [
                datetime(2021, 2, 1, 0, 0),
                datetime(2021, 3, 1, 0, 0),
                datetime(2021, 4, 1, 0, 0),
                datetime(2021, 5, 1, 0, 0),
                datetime(2021, 4, 1, 0, 0),
                datetime(2021, 5, 1, 0, 0),
                datetime(2021, 6, 1, 0, 0),
            ],
            "admin": [
                "Åland",
                "Åland",
                "Åland",
                "Åland",
                "Netherlands",
                "Netherlands",
                "Netherlands",
            ],
            "test2": [0, 0, 0, 2, 1, 1, 3],
        }
    )
    expected = expected.with_columns(pl.col("time").dt.replace_time_zone(time_zone))

    assert_frame_equal(up, expected)


@pytest.mark.parametrize("time_zone", [None, "US/Central"])
def test_upsample_crossing_dst(time_zone: str | None) -> None:
    df = pl.DataFrame(
        {
            "time": pl.datetime_range(
                datetime(2021, 11, 6),
                datetime(2021, 11, 8),
                time_zone=time_zone,
                eager=True,
            ),
            "values": [1, 2, 3],
        }
    )

    result = df.upsample(time_column="time", every="1d")

    expected = pl.DataFrame(
        {
            "time": [
                datetime(2021, 11, 6),
                datetime(2021, 11, 7),
                datetime(2021, 11, 8),
            ],
            "values": [1, 2, 3],
        }
    ).with_columns(pl.col("time").dt.replace_time_zone(time_zone))

    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("time_zone", "tzinfo"),
    [
        (None, None),
        ("Pacific/Rarotonga", ZoneInfo("Pacific/Rarotonga")),
    ],
)
def test_upsample_time_zones(
    time_zone: str | None, tzinfo: timezone | ZoneInfo | None
) -> None:
    df = pl.DataFrame(
        {
            "time": pl.datetime_range(
                start=datetime(2021, 12, 16),
                end=datetime(2021, 12, 16, 3),
                interval="30m",
                eager=True,
            ),
            "groups": ["a", "a", "a", "b", "b", "a", "a"],
            "values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        }
    )
    expected = pl.DataFrame(
        {
            "time": [
                datetime(2021, 12, 16, 0, 0),
                datetime(2021, 12, 16, 1, 0),
                datetime(2021, 12, 16, 2, 0),
                datetime(2021, 12, 16, 3, 0),
            ],
            "groups": ["a", "a", "b", "a"],
            "values": [1.0, 3.0, 5.0, 7.0],
        }
    )
    df = df.with_columns(pl.col("time").dt.replace_time_zone(time_zone))
    expected = expected.with_columns(pl.col("time").dt.replace_time_zone(time_zone))
    result = df.upsample(time_column="time", every="60m").fill_null(strategy="forward")
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("every", "fill", "expected_index", "expected_groups"),
    [
        (
            "1i",
            "forward",
            [1, 2, 3, 4] + [5, 6, 7],
            ["a"] * 4 + ["b"] * 3,
        ),
        (
            "1i",
            "backward",
            [1, 2, 3, 4] + [5, 6, 7],
            ["a"] * 4 + ["b"] * 3,
        ),
    ],
)
@pytest.mark.parametrize("dtype", [pl.Int32, pl.Int64, pl.UInt32, pl.UInt64])
def test_upsample_index(
    every: str,
    fill: FillNullStrategy | None,
    expected_index: list[int],
    expected_groups: list[str],
    dtype: PolarsIntegerType,
) -> None:
    df = (
        pl.DataFrame(
            {
                "index": [1, 2, 4] + [5, 7],
                "groups": ["a"] * 3 + ["b"] * 2,
            }
        )
        .with_columns(pl.col("index").cast(dtype))
        .set_sorted("index")
    )
    expected = pl.DataFrame(
        {
            "index": expected_index,
            "groups": expected_groups,
        }
    ).with_columns(pl.col("index").cast(dtype))
    result = (
        df.upsample(time_column="index", group_by="groups", every=every)
        .fill_null(strategy=fill)
        .sort(["groups", "index"])
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("maintain_order", [True, False])
def test_upsample_index_invalid(
    df: pl.DataFrame,
    maintain_order: bool,
) -> None:
    df = pl.DataFrame(
        {
            "index": [1, 2, 4, 5, 7],
            "groups": ["a"] * 3 + ["b"] * 2,
        }
    ).set_sorted("index")

    with pytest.raises(InvalidOperationError, match=r"must be a parsed integer"):
        df.upsample(
            time_column="index",
            every="1h",
            maintain_order=maintain_order,
        )


def test_upsample_sorted_only_within_group() -> None:
    df = pl.DataFrame(
        {
            "time": [
                datetime(2021, 4, 1),
                datetime(2021, 2, 1),
                datetime(2021, 5, 1),
                datetime(2021, 6, 1),
            ],
            "admin": ["Netherlands", "Åland", "Åland", "Netherlands"],
            "test2": [1, 0, 2, 3],
        }
    )

    up = df.upsample(
        time_column="time",
        every="1mo",
        group_by="admin",
        maintain_order=True,
    ).select(pl.all().forward_fill())

    expected = pl.DataFrame(
        {
            "time": [
                datetime(2021, 4, 1, 0, 0),
                datetime(2021, 5, 1, 0, 0),
                datetime(2021, 6, 1, 0, 0),
                datetime(2021, 2, 1, 0, 0),
                datetime(2021, 3, 1, 0, 0),
                datetime(2021, 4, 1, 0, 0),
                datetime(2021, 5, 1, 0, 0),
            ],
            "admin": [
                "Netherlands",
                "Netherlands",
                "Netherlands",
                "Åland",
                "Åland",
                "Åland",
                "Åland",
            ],
            "test2": [1, 1, 3, 0, 0, 0, 2],
        }
    )

    assert_frame_equal(up, expected)


def test_upsample_sorted_only_within_group_but_no_group_by_provided() -> None:
    df = pl.DataFrame(
        {
            "time": [
                datetime(2021, 4, 1),
                datetime(2021, 2, 1),
                datetime(2021, 5, 1),
                datetime(2021, 6, 1),
            ],
            "admin": ["Netherlands", "Åland", "Åland", "Netherlands"],
            "test2": [1, 0, 2, 3],
        }
    )
    with pytest.raises(
        InvalidOperationError,
        match=r"argument in operation 'upsample' is not sorted, please sort the 'expr/series/column' first",
    ):
        df.upsample(time_column="time", every="1mo")
