"""Tests for ``.list.index_of_in()``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

from tests.unit.operations.test_index_of import get_expected_index

if TYPE_CHECKING:
    from polars._typing import IntoExpr, PythonLiteral

IdxType = pl.get_index_type()


def assert_index_of_in_from_scalar(
    list_series: pl.Series, value: PythonLiteral
) -> None:
    expected_indexes = [
        get_expected_index(sub_series, value) for sub_series in list_series
    ]

    original_value = value
    del value
    for updated_value in (original_value, pl.lit(original_value)):
        # Eager API:
        assert_series_equal(
            list_series.list.index_of_in(updated_value),
            pl.Series(list_series.name, expected_indexes, dtype=IdxType),
        )
        # Lazy API:
        assert_frame_equal(
            pl.LazyFrame({"lists": list_series})
            .select(pl.col("lists").list.index_of_in(updated_value))
            .collect(),
            pl.DataFrame({"lists": expected_indexes}, schema={"lists": IdxType}),
        )


def assert_index_of_in_from_series(
    list_series: pl.Series,
    values: pl.Series,
) -> None:
    expected_indexes = [
        get_expected_index(sub_series, value)
        for (sub_series, value) in zip(list_series, values)
    ]

    # Eager API:
    assert_series_equal(
        list_series.list.index_of_in(values),
        pl.Series(list_series.name, expected_indexes, dtype=IdxType),
    )
    # Lazy API:
    assert_frame_equal(
        pl.LazyFrame({"lists": list_series, "values": values})
        .select(pl.col("lists").list.index_of_in(pl.col("values")))
        .collect(),
        pl.DataFrame({"lists": expected_indexes}, schema={"lists": IdxType}),
    )


# Testing plan:
# - All integers
# - Both floats, with nans
# - Strings
# - datetime, date, time, timedelta
# - nested lists
# - something with hypothesis
# - error case: non-matching lengths


def test_index_of_in_from_scalar() -> None:
    list_series = pl.Series([[3, 1], [2, 4], [5, 3, 1]])
    assert_index_of_in_from_scalar(list_series, 1)


def test_index_of_in_from_series() -> None:
    list_series = pl.Series([[3, 1], [2, 4], [5, 3, 1]])
    values = pl.Series([1, 2, 6])
    assert_index_of_in_from_series(list_series, values)
