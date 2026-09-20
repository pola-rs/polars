from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from typing import Any, Literal


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    ("values", "dtype", "forward", "backward"),
    [
        pytest.param(
            ["a", "b", "a", "c", "b"],
            pl.String,
            [1, 2, 2, 3, 3],
            [3, 3, 3, 2, 1],
            id="strings",
        ),
        pytest.param(
            [4, 1, 4, 2],
            pl.Int64,
            [1, 2, 2, 3],
            [3, 3, 2, 1],
            id="integers",
        ),
        pytest.param(
            [7, 7, 7],
            pl.Int64,
            [1, 1, 1],
            [1, 1, 1],
            id="all-equal",
        ),
        pytest.param(
            [None, 1, None, 1, 2],
            pl.Int64,
            [1, 2, 2, 2, 3],
            [3, 3, 3, 2, 1],
            id="null-counts-once",
        ),
        pytest.param(
            [None, None],
            pl.Int64,
            [1, 1],
            [1, 1],
            id="all-null",
        ),
        pytest.param(
            [None, None],
            pl.Null,
            [1, 1],
            [1, 1],
            id="null-dtype",
        ),
        pytest.param([], pl.Int64, [], [], id="empty"),
        pytest.param(
            [True, False, True, None, False],
            pl.Boolean,
            [1, 2, 2, 3, 3],
            [3, 3, 3, 2, 1],
            id="booleans",
        ),
        pytest.param(
            [1.0, float("nan"), float("nan"), 1.0],
            pl.Float64,
            [1, 2, 2, 2],
            [2, 2, 2, 1],
            id="nan-counts-once",
        ),
    ],
)
def test_cum_n_unique_values(
    values: list[Any],
    dtype: type[pl.DataType],
    forward: list[int],
    backward: list[int],
    reverse: bool,
) -> None:
    series = pl.Series("x", values, dtype=dtype)

    result = (
        series.to_frame()
        .select(pl.col("x").cum_n_unique(reverse=reverse))
        .to_series()
    )

    expected = pl.Series(
        "x",
        backward if reverse else forward,
        dtype=pl.get_index_type(),
    )
    assert_series_equal(result, expected)


def test_cum_n_unique_default_direction() -> None:
    df = pl.DataFrame({"x": ["a", "b", "a"]})

    result = df.select(pl.col("x").cum_n_unique()).to_series()

    expected = pl.Series("x", [1, 2, 2], dtype=pl.get_index_type())
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("reverse", "expected_values"),
    [
        (False, [1, 2, 2, 3]),
        (True, [3, 3, 2, 1]),
    ],
)
def test_cum_n_unique_multiple_chunks(
    reverse: bool, expected_values: list[int]
) -> None:
    series = pl.concat(
        [pl.Series("x", [1, 2]), pl.Series("x", [1, 3])],
        rechunk=False,
    )
    assert series.n_chunks() == 2

    result = (
        series.to_frame()
        .select(pl.col("x").cum_n_unique(reverse=reverse))
        .to_series()
    )

    expected = pl.Series("x", expected_values, dtype=pl.get_index_type())
    assert_series_equal(result, expected)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize(
    ("reverse", "expected_values"),
    [
        (False, [1, 1, 1, 2, 2]),
        (True, [2, 2, 2, 1, 1]),
    ],
)
def test_cum_n_unique_over_groups(
    engine: Literal["in-memory", "streaming"],
    reverse: bool,
    expected_values: list[int],
) -> None:
    df = pl.DataFrame(
        {
            "group": ["a", "b", "a", "b", "a"],
            "x": [1, 1, 1, 2, 2],
        }
    )
    query = df.lazy().select(
        pl.col("x")
        .cum_n_unique(reverse=reverse)
        .over("group")
        .alias("distinct_count")
    )

    expected = pl.DataFrame(
        {"distinct_count": expected_values},
        schema={"distinct_count": pl.get_index_type()},
    )
    assert_frame_equal(query.collect(engine=engine), expected)


@pytest.mark.parametrize("reverse", [False, True])
def test_cum_n_unique_schema_and_alias(reverse: bool) -> None:
    query = pl.LazyFrame({"x": [1, 2, 1]}).select(
        pl.col("x").cum_n_unique(reverse=reverse).alias("distinct_count")
    )

    expected_schema = pl.Schema({"distinct_count": pl.get_index_type()})
    assert query.collect_schema() == expected_schema
    assert query.collect().schema == expected_schema
