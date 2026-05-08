from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal


def test_join_asof_many_forward() -> None:
    left = pl.DataFrame(
        {
            "ts_a": [1, 2, 3],
            "ts_b": [2, 3, 4],
            "value": [10, 20, 30],
        }
    )
    right = pl.DataFrame(
        {
            "ts_a": [2, 4, 5],
            "ts_b": [1, 3, 6],
            "payload": [100, 200, 300],
        }
    )

    result = left.join_asof_many(
        right,
        pairs=[
            pl.AsofJoinPair(left_on="ts_a", right_on="ts_a", suffix="_a"),
            pl.AsofJoinPair(left_on="ts_b", right_on="ts_b", suffix="_b"),
        ],
        strategy="forward",
        coalesce=False,
    )

    expected = (
        left.join_asof(
            right,
            left_on="ts_a",
            right_on="ts_a",
            strategy="forward",
            suffix="_a",
            coalesce=False,
        )
        .join_asof(
            right,
            left_on="ts_b",
            right_on="ts_b",
            strategy="forward",
            suffix="_b",
            coalesce=False,
        )
    )

    assert_frame_equal(result, expected)


def test_join_asof_many_lazy() -> None:
    left = pl.LazyFrame(
        {
            "ts_a": [1, 3, 5],
            "ts_b": [2, 4, 6],
        }
    )
    right = pl.LazyFrame(
        {
            "ts_a": [2, 4, 6],
            "ts_b": [1, 5, 7],
            "payload": [5, 6, 7],
        }
    )

    result = left.join_asof_many(
        right,
        pairs=[
            pl.AsofJoinPair(left_on="ts_a", right_on="ts_a", suffix="_a"),
            pl.AsofJoinPair(left_on="ts_b", right_on="ts_b", suffix="_b"),
        ],
        strategy="backward",
        coalesce=False,
    ).collect()

    expected = (
        left.join_asof(
            right,
            left_on="ts_a",
            right_on="ts_a",
            strategy="backward",
            suffix="_a",
            coalesce=False,
        )
        .join_asof(
            right,
            left_on="ts_b",
            right_on="ts_b",
            strategy="backward",
            suffix="_b",
            coalesce=False,
        )
        .collect()
    )

    assert_frame_equal(result, expected)
