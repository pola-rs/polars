from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

import polars as pl

if TYPE_CHECKING:
    from polars._typing import NonNestedLiteral
from polars.testing import assert_frame_equal


def assert_index_of(series: pl.Series, value: NonNestedLiteral | None) -> None:
    """``Series.index_of()`` returns the index, or ``None`` if it can't be found."""
    if isinstance(value, (np.number, float, int)) and np.isnan(value):
        expected_index = None
        for i, o in enumerate(series.to_list()):
            if o is not None and np.isnan(o):
                expected_index = i
                break
    else:
        try:
            expected_index = series.to_list().index(value)
        except ValueError:
            expected_index = None
    if expected_index == -1:
        expected_index = None
    # Eager API:
    assert series.index_of(value) == expected_index
    # Lazy API:
    assert pl.LazyFrame({"series": series}).select(
        pl.col("series").index_of(value)
    ).collect().get_column("series").to_list() == [expected_index]


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64])
def test_float(dtype: pl.DataType) -> None:
    values = [1.5, np.nan, np.inf, 3.0, None, -np.inf]
    series = pl.Series(values, dtype=dtype)
    sorted_series_asc = series.sort(descending=False)
    sorted_series_desc = series.sort(descending=True)
    chunked_series = pl.concat([pl.Series([1, 7], dtype=dtype), series], rechunk=False)

    for value in values + [
        np.int8(3),
        np.int64(2**42),
        np.float64(1.5),
        np.float32(1.5),
        np.float32(2**37),
        np.float64(2**100),
    ]:
        for s in [series, sorted_series_asc, sorted_series_desc, chunked_series]:
            assert_index_of(s, value)  # type: ignore[arg-type]


def test_null() -> None:
    series = pl.Series([None, None], dtype=pl.Null)
    assert_index_of(series, None)
    assert_index_of(series, 3)


def test_empty() -> None:
    series = pl.Series([], dtype=pl.Null)
    assert_index_of(series, None)
    series = pl.Series([], dtype=pl.Int64)
    assert_index_of(series, None)
    assert_index_of(series, 12)
    assert_index_of(series.sort(descending=True), 12)
    assert_index_of(series.sort(descending=False), 12)


@pytest.mark.parametrize(
    "dtype",
    [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64],
)
def test_integer(dtype: pl.DataType) -> None:
    values = [51, 3, None, 4]
    series = pl.Series(values, dtype=dtype)
    sorted_series_asc = series.sort(descending=False)
    sorted_series_desc = series.sort(descending=True)
    chunked_series = pl.concat(
        [pl.Series([100, 7], dtype=dtype), series], rechunk=False
    )

    for value in values + [
        np.int8(3),
        np.int64(2**42),
        np.float64(3.0),
        np.float32(3.0),
        np.uint64(2**63),
        np.int8(-7),
        np.float32(1.5),
        np.float64(1.5),
        # This caught a bug where rounding was erroneously happening:
        np.float32(3.1),
        np.float64(3.1),
    ]:
        for s in [series, sorted_series_asc, sorted_series_desc, chunked_series]:
            assert_index_of(s, value)  # type: ignore[arg-type]


def test_groupby() -> None:
    df = pl.DataFrame(
        {"label": ["a", "b", "a", "b", "a", "b"], "value": [10, 3, 20, 2, 40, 20]}
    )
    expected = pl.DataFrame(
        {"label": ["a", "b"], "value": [1, 2]},
        schema={"label": pl.String, "value": pl.UInt32},
    )
    assert_frame_equal(
        df.group_by("label", maintain_order=True).agg(pl.col("value").index_of(20)),
        expected,
    )
    assert_frame_equal(
        df.lazy()
        .group_by("label", maintain_order=True)
        .agg(pl.col("value").index_of(20))
        .collect(),
        expected,
    )


LISTS_STRATEGY = st.lists(
    st.one_of(st.none(), st.integers(min_value=10, max_value=50)), max_size=10
)


@given(
    list1=LISTS_STRATEGY,
    list2=LISTS_STRATEGY,
    list3=LISTS_STRATEGY,
)
# The examples are cases where this test previously caught bugs:
@example([], [], [None])
def test_randomized(
    list1: list[int | None], list2: list[int | None], list3: list[int | None]
) -> None:
    series = pl.concat(
        [pl.Series(values, dtype=pl.Int8) for values in [list1, list2, list3]],
        rechunk=False,
    )
    sorted_series = series.sort(descending=False)
    sorted_series2 = series.sort(descending=True)

    # Values are between 10 and 50, plus add None and add out-of-range values:
    for i in set(range(10, 51)) | {-2000, 255, 2000, None}:
        assert_index_of(series, i)
        assert_index_of(sorted_series, i)
        assert_index_of(sorted_series2, i)