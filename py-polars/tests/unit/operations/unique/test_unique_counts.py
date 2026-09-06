from datetime import datetime

from hypothesis import given

import polars as pl
from polars.testing import assert_series_equal
from polars.testing.parametric import series


def test_unique_counts() -> None:
    s = pl.Series("id", ["a", "b", "b", "c", "c", "c"])
    expected = pl.Series("id", [1, 2, 3], dtype=pl.get_index_type())
    assert_series_equal(s.unique_counts(), expected)


def test_unique_counts_on_dates() -> None:
    assert pl.DataFrame(
        {
            "dt_ns": pl.datetime_range(
                datetime(2020, 1, 1), datetime(2020, 3, 1), "1mo", eager=True
            ),
        }
    ).with_columns(
        pl.col("dt_ns").dt.cast_time_unit("us").alias("dt_us"),
        pl.col("dt_ns").dt.cast_time_unit("ms").alias("dt_ms"),
        pl.col("dt_ns").cast(pl.Date).alias("date"),
    ).select(pl.all().unique_counts().sum()).to_dict(as_series=False) == {
        "dt_ns": [3],
        "dt_us": [3],
        "dt_ms": [3],
        "date": [3],
    }


def test_unique_counts_null() -> None:
    s = pl.Series([])
    expected = pl.Series([], dtype=pl.get_index_type())
    assert_series_equal(s.unique_counts(), expected)

    s = pl.Series([None])
    expected = pl.Series([1], dtype=pl.get_index_type())
    assert_series_equal(s.unique_counts(), expected)

    s = pl.Series([None, None, None])
    expected = pl.Series([3], dtype=pl.get_index_type())
    assert_series_equal(s.unique_counts(), expected)


def test_unique_counts_boolean_sorted_nulls_last() -> None:
    # `first_null` on a sorted-flagged series with the nulls at the end returned
    # `null_count` instead of `len - null_count`, which put the null count in the
    # wrong position of the result.
    ascending = pl.Series("a", [None, False, True, False, None, True, True]).sort(
        nulls_last=True
    )
    assert ascending.to_list() == [False, False, True, True, True, None, None]
    assert ascending.flags["SORTED_ASC"]
    assert_series_equal(
        ascending.unique_counts(),
        pl.Series("a", [2, 3, 2], dtype=pl.get_index_type()),
    )

    descending = pl.Series("a", [None, False, True, False, True]).sort(
        descending=True, nulls_last=True
    )
    assert descending.to_list() == [True, True, False, False, None]
    assert descending.flags["SORTED_DESC"]
    assert_series_equal(
        descending.unique_counts(),
        pl.Series("a", [2, 2, 1], dtype=pl.get_index_type()),
    )


@given(s=series(excluded_dtypes=[pl.Object]))
def test_unique_counts_parametric(s: pl.Series) -> None:
    result = s.unique_counts()
    expected = (
        s.to_frame()
        .group_by(s.name, maintain_order=True)
        .agg(pl.len())
        .get_columns()[1]
    )

    assert_series_equal(result, expected, check_names=False)
