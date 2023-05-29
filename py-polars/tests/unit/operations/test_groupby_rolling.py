from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from polars.type_aliases import ClosedInterval


def bad_agg_parameters() -> list[Any]:
    """Currently, IntoExpr and Iterable[IntoExpr] are supported."""
    return [[("b", "sum")], [("b", ["sum"])], str, "b".join]


def good_agg_parameters() -> list[pl.Expr | list[pl.Expr]]:
    return [
        [pl.col("b").sum()],
        pl.col("b").sum(),
    ]


def test_groupby_rolling_apply() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [1, 2, 3, 4, 5],
        }
    ).set_sorted("a")

    def apply(df: pl.DataFrame) -> pl.DataFrame:
        return df.select(
            pl.col("a").min(),
            pl.col("b").max(),
        )

    expected = pl.DataFrame(
        [
            pl.Series("a", [1, 1, 2, 3, 4], dtype=pl.Int64),
            pl.Series("b", [1, 2, 3, 4, 5], dtype=pl.Int64),
        ]
    )

    out = df.groupby_rolling("a", period="2i").apply(apply, schema=df.schema)
    assert_frame_equal(out, expected)


def test_rolling_groupby_overlapping_groups() -> None:
    # this first aggregates overlapping groups
    # so they cannot be naively flattened
    df = pl.DataFrame(
        {
            "a": [41, 60, 37, 51, 52, 39, 40],
        }
    )

    assert_series_equal(
        (
            df.with_row_count()
            .with_columns(pl.col("row_nr").cast(pl.Int32))
            .groupby_rolling(
                index_column="row_nr",
                period="5i",
            )
            .agg(
                # the apply to trigger the apply on the expression engine
                pl.col("a")
                .apply(lambda x: x)
                .sum()
            )
        )["a"],
        df["a"].rolling_sum(window_size=5, min_periods=1),
    )


@pytest.mark.parametrize("lazy", [True, False])
def test_groupby_rolling_agg_input_types(lazy: bool) -> None:
    df = pl.DataFrame({"index_column": [0, 1, 2, 3], "b": [1, 3, 1, 2]}).set_sorted(
        "index_column"
    )
    df_or_lazy: pl.DataFrame | pl.LazyFrame = df.lazy() if lazy else df

    for bad_param in bad_agg_parameters():
        with pytest.raises(TypeError):  # noqa: PT012
            result = df_or_lazy.groupby_rolling(
                index_column="index_column", period="2i"
            ).agg(bad_param)
            if lazy:
                result.collect()  # type: ignore[union-attr]

    expected = pl.DataFrame({"index_column": [0, 1, 2, 3], "b": [1, 4, 4, 3]})

    for good_param in good_agg_parameters():
        result = df_or_lazy.groupby_rolling(
            index_column="index_column", period="2i"
        ).agg(good_param)
        if lazy:
            result = result.collect()  # type: ignore[union-attr]
        assert_frame_equal(result, expected)


def test_groupby_rolling_negative_offset_3914() -> None:
    df = pl.DataFrame(
        {
            "datetime": pl.date_range(
                datetime(2020, 1, 1), datetime(2020, 1, 5), "1d", eager=True
            ),
        }
    )
    assert df.groupby_rolling(index_column="datetime", period="2d", offset="-4d").agg(
        pl.count().alias("count")
    )["count"].to_list() == [0, 0, 1, 2, 2]

    df = pl.DataFrame(
        {
            "ints": range(0, 20),
        }
    )

    assert df.groupby_rolling(index_column="ints", period="2i", offset="-5i").agg(
        [pl.col("ints").alias("matches")]
    )["matches"].to_list() == [
        [],
        [],
        [],
        [0],
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [12, 13],
        [13, 14],
        [14, 15],
        [15, 16],
    ]


@pytest.mark.parametrize("time_zone", [None, "US/Central"])
def test_groupby_rolling_negative_offset_crossing_dst(time_zone: str | None) -> None:
    df = pl.DataFrame(
        {
            "datetime": pl.date_range(
                datetime(2021, 11, 6),
                datetime(2021, 11, 9),
                "1d",
                time_zone=time_zone,
                eager=True,
            ),
            "value": [1, 4, 9, 155],
        }
    )
    result = df.groupby_rolling(index_column="datetime", period="2d", offset="-1d").agg(
        pl.col("value")
    )
    expected = pl.DataFrame(
        {
            "datetime": pl.date_range(
                datetime(2021, 11, 6),
                datetime(2021, 11, 9),
                "1d",
                time_zone=time_zone,
                eager=True,
            ),
            "value": [[1, 4], [4, 9], [9, 155], [155]],
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("time_zone", [None, "US/Central"])
@pytest.mark.parametrize(
    ("offset", "closed", "expected_values"),
    [
        ("0d", "left", [[1, 4], [4, 9], [9, 155], [155]]),
        ("0d", "right", [[4, 9], [9, 155], [155], []]),
        ("0d", "both", [[1, 4, 9], [4, 9, 155], [9, 155], [155]]),
        ("0d", "none", [[4], [9], [155], []]),
        ("1d", "left", [[4, 9], [9, 155], [155], []]),
        ("1d", "right", [[9, 155], [155], [], []]),
        ("1d", "both", [[4, 9, 155], [9, 155], [155], []]),
        ("1d", "none", [[9], [155], [], []]),
    ],
)
def test_groupby_rolling_non_negative_offset_9077(
    time_zone: str | None,
    offset: str,
    closed: ClosedInterval,
    expected_values: list[list[int]],
) -> None:
    df = pl.DataFrame(
        {
            "datetime": pl.date_range(
                datetime(2021, 11, 6),
                datetime(2021, 11, 9),
                "1d",
                time_zone=time_zone,
                eager=True,
            ),
            "value": [1, 4, 9, 155],
        }
    )
    result = df.groupby_rolling(
        index_column="datetime", period="2d", offset=offset, closed=closed
    ).agg(pl.col("value"))
    expected = pl.DataFrame(
        {
            "datetime": pl.date_range(
                datetime(2021, 11, 6),
                datetime(2021, 11, 9),
                "1d",
                time_zone=time_zone,
                eager=True,
            ),
            "value": expected_values,
        }
    )
    assert_frame_equal(result, expected)


def test_groupby_rolling_dynamic_sortedness_check() -> None:
    # when the by argument is passed, the sortedness flag
    # will be unset as the take shuffles data, so we must explicitly
    # check the sortedness
    df = pl.DataFrame(
        {
            "idx": [1, 2, -1, 2, 1, 1],
            "group": [1, 1, 1, 2, 2, 1],
        }
    )

    with pytest.raises(pl.ComputeError, match=r"input data is not sorted"):
        df.groupby_dynamic("idx", every="2i", by="group").agg(
            pl.col("idx").alias("idx1")
        )

    with pytest.raises(pl.ComputeError, match=r"input data is not sorted"):
        df.groupby_rolling("idx", period="2i", by="group").agg(
            pl.col("idx").alias("idx1")
        )

    # no `by` argument
    with pytest.raises(
        pl.InvalidOperationError,
        match=r"argument in operation 'groupby_dynamic' is not explicitly sorted",
    ):
        df.groupby_dynamic("idx", every="2i").agg(pl.col("idx").alias("idx1"))

    # no `by` argument
    with pytest.raises(
        pl.InvalidOperationError,
        match=r"argument in operation 'groupby_rolling' is not explicitly sorted",
    ):
        df.groupby_rolling("idx", period="2i").agg(pl.col("idx").alias("idx1"))
