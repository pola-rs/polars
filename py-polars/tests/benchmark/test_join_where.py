"""Benchmark tests for join_where with inequality conditions."""

from __future__ import annotations

import numpy as np
import pytest

import polars as pl
from polars.exceptions import ColumnNotFoundError

pytestmark = pytest.mark.benchmark()


def test_strict_inequalities(east_west: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    east, west = east_west
    result = (
        east.lazy()
        .join_where(
            west.lazy(),
            [pl.col("dur") < pl.col("time"), pl.col("rev") > pl.col("cost")],
        )
        .collect()
    )

    assert len(result) > 0


def test_non_strict_inequalities(east_west: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    east, west = east_west
    result = (
        east.lazy()
        .join_where(
            west.lazy(),
            [pl.col("dur") <= pl.col("time"), pl.col("rev") >= pl.col("cost")],
        )
        .collect()
    )

    assert len(result) > 0


def test_single_inequality(east_west: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    east, west = east_west
    result = (
        east.lazy()
        # Reduce the number of results by scaling LHS dur column up
        .with_columns((pl.col("dur") * 30).alias("scaled_dur"))
        .join_where(
            west.lazy(),
            pl.col("scaled_dur") < pl.col("time"),
        )
        .collect()
    )

    assert len(result) > 0


@pytest.fixture(scope="module")
def east_west() -> tuple[pl.DataFrame, pl.DataFrame]:
    num_rows_left, num_rows_right = 50_000, 5_000
    rng = np.random.default_rng(42)

    # Generate two separate datasets where revenue/cost are linearly related to
    # duration/time, but add some noise to the west table so that there are some
    # rows where the cost for the same or greater time will be less than the east table.
    east_dur = rng.integers(1_000, 50_000, num_rows_left)
    east_rev = (east_dur * 0.123).astype(np.int32)
    west_time = rng.integers(1_000, 50_000, num_rows_right)
    west_cost = west_time * 0.123
    west_cost += rng.normal(0.0, 1.0, num_rows_right)
    west_cost = west_cost.astype(np.int32)

    east = pl.DataFrame(
        {
            "id": np.arange(0, num_rows_left),
            "dur": east_dur,
            "rev": east_rev,
            "cores": rng.integers(1, 10, num_rows_left),
        }
    )
    west = pl.DataFrame(
        {
            "t_id": np.arange(0, num_rows_right),
            "time": west_time,
            "cost": west_cost,
            "cores": rng.integers(1, 10, num_rows_right),
        }
    )

    return east, west


def test_join_where_invalid_column() -> None:
    df = pl.DataFrame({"x": 1})
    with pytest.raises(ColumnNotFoundError, match="y"):
        df.join_where(df, pl.col("x") < pl.col("y"))

    # Nested column
    df1 = pl.DataFrame({"a": [1, 2, 3], "b": [True, False, True]})
    df2 = pl.DataFrame(
        {
            "a": [2, 3, 4],
            "c": ["a", "b", "c"],
        }
    )
    with pytest.raises(ColumnNotFoundError, match="d"):
        df = df1.join_where(
            df2,
            ((pl.col("a") - pl.col("b")) > (pl.col("c") == "a").cast(pl.Int32))
            > (pl.col("a") - pl.col("d")),
        )
