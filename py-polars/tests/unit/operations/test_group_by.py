from __future__ import annotations

import typing
from collections import OrderedDict
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from hypothesis import given

import polars as pl
import polars.selectors as cs
from polars import Expr
from polars.exceptions import (
    ColumnNotFoundError,
    InvalidOperationError,
)
from polars.meta import get_index_type
from polars.testing import assert_frame_equal, assert_series_equal
from polars.testing.parametric import column, dataframes, series

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars._typing import PolarsDataType, TimeUnit
    from tests.conftest import PlMonkeyPatch


def test_group_by() -> None:
    df = pl.DataFrame(
        {
            "a": ["a", "b", "a", "b", "b", "c"],
            "b": [1, 2, 3, 4, 5, 6],
            "c": [6, 5, 4, 3, 2, 1],
        }
    )

    # Use lazy API in eager group_by
    assert sorted(df.group_by("a").agg([pl.sum("b")]).rows()) == [
        ("a", 4),
        ("b", 11),
        ("c", 6),
    ]
    # test if it accepts a single expression
    assert df.group_by("a", maintain_order=True).agg(pl.sum("b")).rows() == [
        ("a", 4),
        ("b", 11),
        ("c", 6),
    ]

    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": ["a", "a", "b", "b", "b"],
            "c": [None, 1, None, 1, None],
        }
    )

    # check if this query runs and thus column names propagate
    df.group_by("b").agg(pl.col("c").fill_null(strategy="forward")).explode("c")

    # get a specific column
    result = df.group_by("b", maintain_order=True).agg(pl.count("a"))
    assert result.rows() == [("a", 2), ("b", 3)]
    assert result.columns == ["b", "a"]


@pytest.mark.parametrize(
    ("input", "expected", "input_dtype", "output_dtype"),
    [
        ([1, 2, 3, 4], [2, 4], pl.UInt8, pl.Float64),
        ([1, 2, 3, 4], [2, 4], pl.Int8, pl.Float64),
        ([1, 2, 3, 4], [2, 4], pl.UInt16, pl.Float64),
        ([1, 2, 3, 4], [2, 4], pl.Int16, pl.Float64),
        ([1, 2, 3, 4], [2, 4], pl.UInt32, pl.Float64),
        ([1, 2, 3, 4], [2, 4], pl.Int32, pl.Float64),
        ([1, 2, 3, 4], [2, 4], pl.UInt64, pl.Float64),
        ([1, 2, 3, 4], [2, 4], pl.Float32, pl.Float32),
        ([1, 2, 3, 4], [2, 4], pl.Float64, pl.Float64),
        ([False, True, True, True], [2 / 3, 1], pl.Boolean, pl.Float64),
        (
            [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 4), date(2023, 1, 5)],
            [datetime(2023, 1, 2, 8, 0, 0), datetime(2023, 1, 5)],
            pl.Date,
            pl.Datetime("us"),
        ),
        (
            [
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 3),
                datetime(2023, 1, 4),
            ],
            [datetime(2023, 1, 2), datetime(2023, 1, 4)],
            pl.Datetime("ms"),
            pl.Datetime("ms"),
        ),
        (
            [
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 3),
                datetime(2023, 1, 4),
            ],
            [datetime(2023, 1, 2), datetime(2023, 1, 4)],
            pl.Datetime("us"),
            pl.Datetime("us"),
        ),
        (
            [
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 3),
                datetime(2023, 1, 4),
            ],
            [datetime(2023, 1, 2), datetime(2023, 1, 4)],
            pl.Datetime("ns"),
            pl.Datetime("ns"),
        ),
        (
            [timedelta(1), timedelta(2), timedelta(3), timedelta(4)],
            [timedelta(2), timedelta(4)],
            pl.Duration("ms"),
            pl.Duration("ms"),
        ),
        (
            [timedelta(1), timedelta(2), timedelta(3), timedelta(4)],
            [timedelta(2), timedelta(4)],
            pl.Duration("us"),
            pl.Duration("us"),
        ),
        (
            [timedelta(1), timedelta(2), timedelta(3), timedelta(4)],
            [timedelta(2), timedelta(4)],
            pl.Duration("ns"),
            pl.Duration("ns"),
        ),
    ],
)
def test_group_by_mean_by_dtype(
    input: list[Any],
    expected: list[Any],
    input_dtype: PolarsDataType,
    output_dtype: PolarsDataType,
) -> None:
    # groups are defined by first 3 values, then last value
    name = str(input_dtype)
    key = ["a", "a", "a", "b"]
    df = pl.LazyFrame(
        {
            "key": key,
            name: pl.Series(input, dtype=input_dtype),
        }
    )
    result = df.group_by("key", maintain_order=True).mean()
    df_expected = pl.DataFrame(
        {
            "key": ["a", "b"],
            name: pl.Series(expected, dtype=output_dtype),
        }
    )
    assert result.collect_schema() == df_expected.schema
    assert_frame_equal(result.collect(), df_expected)


@pytest.mark.parametrize(
    ("input", "expected", "input_dtype", "output_dtype"),
    [
        ([1, 2, 4, 5], [2, 5], pl.UInt8, pl.Float64),
        ([1, 2, 4, 5], [2, 5], pl.Int8, pl.Float64),
        ([1, 2, 4, 5], [2, 5], pl.UInt16, pl.Float64),
        ([1, 2, 4, 5], [2, 5], pl.Int16, pl.Float64),
        ([1, 2, 4, 5], [2, 5], pl.UInt32, pl.Float64),
        ([1, 2, 4, 5], [2, 5], pl.Int32, pl.Float64),
        ([1, 2, 4, 5], [2, 5], pl.UInt64, pl.Float64),
        ([1, 2, 4, 5], [2, 5], pl.Float32, pl.Float32),
        ([1, 2, 4, 5], [2, 5], pl.Float64, pl.Float64),
        ([False, True, True, True], [1, 1], pl.Boolean, pl.Float64),
        (
            [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 4), date(2023, 1, 5)],
            [datetime(2023, 1, 2), datetime(2023, 1, 5)],
            pl.Date,
            pl.Datetime("us"),
        ),
        (
            [
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 4),
                datetime(2023, 1, 5),
            ],
            [datetime(2023, 1, 2), datetime(2023, 1, 5)],
            pl.Datetime("ms"),
            pl.Datetime("ms"),
        ),
        (
            [
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 4),
                datetime(2023, 1, 5),
            ],
            [datetime(2023, 1, 2), datetime(2023, 1, 5)],
            pl.Datetime("us"),
            pl.Datetime("us"),
        ),
        (
            [
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 4),
                datetime(2023, 1, 5),
            ],
            [datetime(2023, 1, 2), datetime(2023, 1, 5)],
            pl.Datetime("ns"),
            pl.Datetime("ns"),
        ),
        (
            [timedelta(1), timedelta(2), timedelta(4), timedelta(5)],
            [timedelta(2), timedelta(5)],
            pl.Duration("ms"),
            pl.Duration("ms"),
        ),
        (
            [timedelta(1), timedelta(2), timedelta(4), timedelta(5)],
            [timedelta(2), timedelta(5)],
            pl.Duration("us"),
            pl.Duration("us"),
        ),
        (
            [timedelta(1), timedelta(2), timedelta(4), timedelta(5)],
            [timedelta(2), timedelta(5)],
            pl.Duration("ns"),
            pl.Duration("ns"),
        ),
    ],
)
def test_group_by_median_by_dtype(
    input: list[Any],
    expected: list[Any],
    input_dtype: PolarsDataType,
    output_dtype: PolarsDataType,
) -> None:
    # groups are defined by first 3 values, then last value
    name = str(input_dtype)
    key = ["a", "a", "a", "b"]
    df = pl.LazyFrame(
        {
            "key": key,
            name: pl.Series(input, dtype=input_dtype),
        }
    )
    result = df.group_by("key", maintain_order=True).median()
    df_expected = pl.DataFrame(
        {
            "key": ["a", "b"],
            name: pl.Series(expected, dtype=output_dtype),
        }
    )
    assert result.collect_schema() == df_expected.schema
    assert_frame_equal(result.collect(), df_expected)


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": ["a", "a", "b", "b", "b"],
            "c": [None, 1, None, 1, None],
        }
    )


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("all", [("a", [1, 2], [None, 1]), ("b", [3, 4, 5], [None, 1, None])]),
        ("len", [("a", 2), ("b", 3)]),
        ("first", [("a", 1, None), ("b", 3, None)]),
        ("last", [("a", 2, 1), ("b", 5, None)]),
        ("max", [("a", 2, 1), ("b", 5, 1)]),
        ("mean", [("a", 1.5, 1.0), ("b", 4.0, 1.0)]),
        ("median", [("a", 1.5, 1.0), ("b", 4.0, 1.0)]),
        ("min", [("a", 1, 1), ("b", 3, 1)]),
        ("n_unique", [("a", 2, 2), ("b", 3, 2)]),
    ],
)
def test_group_by_shorthands(
    df: pl.DataFrame, method: str, expected: list[tuple[Any]]
) -> None:
    gb = df.group_by("b", maintain_order=True)
    result = getattr(gb, method)()
    assert result.rows() == expected

    gb_lazy = df.lazy().group_by("b", maintain_order=True)
    result = getattr(gb_lazy, method)().collect()
    assert result.rows() == expected


def test_group_by_shorthand_quantile(df: pl.DataFrame) -> None:
    result = df.group_by("b", maintain_order=True).quantile(0.5)
    expected = [("a", 2.0, 1.0), ("b", 4.0, 1.0)]
    assert result.rows() == expected

    result = df.lazy().group_by("b", maintain_order=True).quantile(0.5).collect()
    assert result.rows() == expected


def test_group_by_quantile_date() -> None:
    df = pl.DataFrame(
        {
            "group": [1, 1, 1, 1, 2, 2, 2, 2],
            "value": [date(2025, 1, x) for x in range(1, 9)],
        }
    )
    result = (
        df.lazy()
        .group_by("group", maintain_order=True)
        .agg(
            nearest=pl.col("value").quantile(0.5, "nearest"),
            higher=pl.col("value").quantile(0.5, "higher"),
            lower=pl.col("value").quantile(0.5, "lower"),
            linear=pl.col("value").quantile(0.5, "linear"),
        )
    )
    dt = pl.Datetime("us")
    expected = pl.DataFrame(
        {
            "group": [1, 2],
            "nearest": pl.Series(
                [datetime(2025, 1, 3), datetime(2025, 1, 7)], dtype=dt
            ),
            "higher": pl.Series([datetime(2025, 1, 3), datetime(2025, 1, 7)], dtype=dt),
            "lower": pl.Series([datetime(2025, 1, 2), datetime(2025, 1, 6)], dtype=dt),
            "linear": pl.Series(
                [datetime(2025, 1, 2, 12), datetime(2025, 1, 6, 12)], dtype=dt
            ),
        }
    )
    assert result.collect_schema() == pl.Schema(
        {  # type: ignore[arg-type]
            "group": pl.Int64,
            "nearest": dt,
            "higher": dt,
            "lower": dt,
            "linear": dt,
        }
    )
    assert_frame_equal(result.collect(), expected)


@pytest.mark.parametrize("tu", ["ms", "us", "ns"])
@pytest.mark.parametrize("time_zone", [None, "Asia/Tokyo"])
def test_group_by_quantile_datetime(tu: TimeUnit, time_zone: str) -> None:
    dt = pl.Datetime(tu, time_zone)
    tz = ZoneInfo(time_zone) if time_zone else None
    df = pl.DataFrame(
        {
            "group": [1, 1, 1, 1, 2, 2, 2, 2],
            "value": pl.Series(
                [datetime(2025, 1, x, tzinfo=tz) for x in range(1, 9)],
                dtype=dt,
            ),
        }
    )
    result = (
        df.lazy()
        .group_by("group", maintain_order=True)
        .agg(
            nearest=pl.col("value").quantile(0.5, "nearest"),
            higher=pl.col("value").quantile(0.5, "higher"),
            lower=pl.col("value").quantile(0.5, "lower"),
            linear=pl.col("value").quantile(0.5, "linear"),
        )
    )
    expected = pl.DataFrame(
        {
            "group": [1, 2],
            "nearest": pl.Series(
                [datetime(2025, 1, 3, tzinfo=tz), datetime(2025, 1, 7, tzinfo=tz)],
                dtype=dt,
            ),
            "higher": pl.Series(
                [datetime(2025, 1, 3, tzinfo=tz), datetime(2025, 1, 7, tzinfo=tz)],
                dtype=dt,
            ),
            "lower": pl.Series(
                [datetime(2025, 1, 2, tzinfo=tz), datetime(2025, 1, 6, tzinfo=tz)],
                dtype=dt,
            ),
            "linear": pl.Series(
                [
                    datetime(2025, 1, 2, 12, tzinfo=tz),
                    datetime(2025, 1, 6, 12, tzinfo=tz),
                ],
                dtype=dt,
            ),
        }
    )
    assert result.collect_schema() == pl.Schema(
        {  # type: ignore[arg-type]
            "group": pl.Int64,
            "nearest": dt,
            "higher": dt,
            "lower": dt,
            "linear": dt,
        }
    )
    assert_frame_equal(result.collect(), expected)


@pytest.mark.parametrize("tu", ["ms", "us", "ns"])
def test_group_by_quantile_duration(tu: TimeUnit) -> None:
    dt = pl.Duration(tu)
    df = pl.DataFrame(
        {
            "group": [1, 1, 1, 1, 2, 2, 2, 2],
            "value": pl.Series([timedelta(hours=x) for x in range(1, 9)], dtype=dt),
        }
    )
    result = (
        df.lazy()
        .group_by("group", maintain_order=True)
        .agg(
            nearest=pl.col("value").quantile(0.5, "nearest"),
            higher=pl.col("value").quantile(0.5, "higher"),
            lower=pl.col("value").quantile(0.5, "lower"),
            linear=pl.col("value").quantile(0.5, "linear"),
        )
    )
    expected = pl.DataFrame(
        {
            "group": [1, 2],
            "nearest": pl.Series([timedelta(hours=3), timedelta(hours=7)], dtype=dt),
            "higher": pl.Series([timedelta(hours=3), timedelta(hours=7)], dtype=dt),
            "lower": pl.Series([timedelta(hours=2), timedelta(hours=6)], dtype=dt),
            "linear": pl.Series(
                [timedelta(hours=2, minutes=30), timedelta(hours=6, minutes=30)],
                dtype=dt,
            ),
        }
    )
    assert result.collect_schema() == pl.Schema(
        {  # type: ignore[arg-type]
            "group": pl.Int64,
            "nearest": dt,
            "higher": dt,
            "lower": dt,
            "linear": dt,
        }
    )
    assert_frame_equal(result.collect(), expected)


def test_group_by_quantile_time() -> None:
    df = pl.DataFrame(
        {
            "group": [1, 1, 1, 1, 2, 2, 2, 2],
            "value": pl.Series([time(hour=x) for x in range(1, 9)]),
        }
    )
    result = (
        df.lazy()
        .group_by("group", maintain_order=True)
        .agg(
            nearest=pl.col("value").quantile(0.5, "nearest"),
            higher=pl.col("value").quantile(0.5, "higher"),
            lower=pl.col("value").quantile(0.5, "lower"),
            linear=pl.col("value").quantile(0.5, "linear"),
        )
    )
    expected = pl.DataFrame(
        {
            "group": [1, 2],
            "nearest": pl.Series([time(hour=3), time(hour=7)]),
            "higher": pl.Series([time(hour=3), time(hour=7)]),
            "lower": pl.Series([time(hour=2), time(hour=6)]),
            "linear": pl.Series([time(hour=2, minute=30), time(hour=6, minute=30)]),
        }
    )
    assert result.collect_schema() == pl.Schema(
        {
            "group": pl.Int64,
            "nearest": pl.Time,
            "higher": pl.Time,
            "lower": pl.Time,
            "linear": pl.Time,
        }
    )
    assert_frame_equal(result.collect(), expected)


def test_group_by_args() -> None:
    df = pl.DataFrame(
        {
            "a": ["a", "b", "a", "b", "b", "c"],
            "b": [1, 2, 3, 4, 5, 6],
            "c": [6, 5, 4, 3, 2, 1],
        }
    )

    # Single column name
    assert df.group_by("a").agg("b").columns == ["a", "b"]
    # Column names as list
    expected = ["a", "b", "c"]
    assert df.group_by(["a", "b"]).agg("c").columns == expected
    # Column names as positional arguments
    assert df.group_by("a", "b").agg("c").columns == expected
    # With keyword argument
    assert df.group_by("a", "b", maintain_order=True).agg("c").columns == expected
    # Multiple aggregations as list
    assert df.group_by("a").agg(["b", "c"]).columns == expected
    # Multiple aggregations as positional arguments
    assert df.group_by("a").agg("b", "c").columns == expected
    # Multiple aggregations as keyword arguments
    assert df.group_by("a").agg(q="b", r="c").columns == ["a", "q", "r"]


def test_group_by_empty() -> None:
    df = pl.DataFrame({"a": [1, 1, 2]})
    result = df.group_by("a").agg()
    expected = pl.DataFrame({"a": [1, 2]})
    assert_frame_equal(result, expected, check_row_order=False)


def test_group_by_iteration() -> None:
    df = pl.DataFrame(
        {
            "foo": ["a", "b", "a", "b", "b", "c"],
            "bar": [1, 2, 3, 4, 5, 6],
            "baz": [6, 5, 4, 3, 2, 1],
        }
    )
    expected_names = ["a", "b", "c"]
    expected_rows = [
        [("a", 1, 6), ("a", 3, 4)],
        [("b", 2, 5), ("b", 4, 3), ("b", 5, 2)],
        [("c", 6, 1)],
    ]
    gb_iter = enumerate(df.group_by("foo", maintain_order=True))
    for i, (group, data) in gb_iter:
        assert group == (expected_names[i],)
        assert data.rows() == expected_rows[i]

    # Grouped by ALL columns should give groups of a single row
    result = list(df.group_by(["foo", "bar", "baz"]))
    assert len(result) == 6

    # Iterating over groups should also work when grouping by expressions
    result2 = list(df.group_by(["foo", pl.col("bar") * pl.col("baz")]))
    assert len(result2) == 5

    # Single expression, alias in group_by
    df = pl.DataFrame({"foo": [1, 2, 3, 4, 5, 6]})
    gb = df.group_by((pl.col("foo") // 2).alias("bar"), maintain_order=True)
    result3 = [(group, df.rows()) for group, df in gb]
    expected3 = [
        ((0,), [(1,)]),
        ((1,), [(2,), (3,)]),
        ((2,), [(4,), (5,)]),
        ((3,), [(6,)]),
    ]
    assert result3 == expected3


def test_group_by_iteration_selector() -> None:
    df = pl.DataFrame({"a": ["one", "two", "one", "two"], "b": [1, 2, 3, 4]})
    result = dict(df.group_by(cs.string()))
    result_first = result["one",]
    assert result_first.to_dict(as_series=False) == {"a": ["one", "one"], "b": [1, 3]}


@pytest.mark.parametrize("input", [[pl.col("b").sum()], pl.col("b").sum()])
def test_group_by_agg_input_types(input: Any) -> None:
    df = pl.LazyFrame({"a": [1, 1, 2, 2], "b": [1, 2, 3, 4]})
    result = df.group_by("a", maintain_order=True).agg(input)
    expected = pl.LazyFrame({"a": [1, 2], "b": [3, 7]})
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("input", [str, "b".join])
def test_group_by_agg_bad_input_types(input: Any) -> None:
    df = pl.LazyFrame({"a": [1, 1, 2, 2], "b": [1, 2, 3, 4]})
    with pytest.raises(TypeError):
        df.group_by("a").agg(input)


def test_group_by_sorted_empty_dataframe_3680() -> None:
    df = (
        pl.DataFrame(
            [
                pl.Series("key", [], dtype=pl.Categorical),
                pl.Series("val", [], dtype=pl.Float64),
            ]
        )
        .lazy()
        .sort("key")
        .group_by("key")
        .tail(1)
        .collect(optimizations=pl.QueryOptFlags(check_order_observe=False))
    )
    assert df.rows() == []
    assert df.shape == (0, 2)
    assert df.schema == {"key": pl.Categorical(), "val": pl.Float64}


def test_group_by_custom_agg_empty_list() -> None:
    assert (
        pl.DataFrame(
            [
                pl.Series("key", [], dtype=pl.Categorical),
                pl.Series("val", [], dtype=pl.Float64),
            ]
        )
        .group_by("key")
        .agg(
            [
                pl.col("val").mean().alias("mean"),
                pl.col("val").std().alias("std"),
                pl.col("val").skew().alias("skew"),
                pl.col("val").kurtosis().alias("kurt"),
            ]
        )
    ).dtypes == [pl.Categorical, pl.Float64, pl.Float64, pl.Float64, pl.Float64]


def test_apply_after_take_in_group_by_3869() -> None:
    assert (
        pl.DataFrame(
            {
                "k": list("aaabbb"),
                "t": [1, 2, 3, 4, 5, 6],
                "v": [3, 1, 2, 5, 6, 4],
            }
        )
        .group_by("k", maintain_order=True)
        .agg(
            pl.col("v").get(pl.col("t").arg_max()).sqrt()
        )  # <- fails for sqrt, exp, log, pow, etc.
    ).to_dict(as_series=False) == {"k": ["a", "b"], "v": [1.4142135623730951, 2.0]}


def test_group_by_signed_transmutes() -> None:
    df = pl.DataFrame({"foo": [-1, -2, -3, -4, -5], "bar": [500, 600, 700, 800, 900]})

    for dt in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
        df = (
            df.with_columns([pl.col("foo").cast(dt), pl.col("bar")])
            .group_by("foo", maintain_order=True)
            .agg(pl.col("bar").median())
        )

        assert df.to_dict(as_series=False) == {
            "foo": [-1, -2, -3, -4, -5],
            "bar": [500.0, 600.0, 700.0, 800.0, 900.0],
        }


def test_arg_sort_sort_by_groups_update__4360() -> None:
    df = pl.DataFrame(
        {
            "group": ["a"] * 3 + ["b"] * 3 + ["c"] * 3,
            "col1": [1, 2, 3] * 3,
            "col2": [1, 2, 3, 3, 2, 1, 2, 3, 1],
        }
    )

    out = df.with_columns(
        pl.col("col2").arg_sort().over("group").alias("col2_arg_sort")
    ).with_columns(
        pl.col("col1").sort_by(pl.col("col2_arg_sort")).over("group").alias("result_a"),
        pl.col("col1")
        .sort_by(pl.col("col2").arg_sort())
        .over("group")
        .alias("result_b"),
    )

    assert_series_equal(out["result_a"], out["result_b"], check_names=False)
    assert out["result_a"].to_list() == [1, 2, 3, 3, 2, 1, 2, 3, 1]


def test_unique_order() -> None:
    df = pl.DataFrame({"a": [1, 2, 1]}).with_row_index()
    assert df.unique(keep="last", subset="a", maintain_order=True).to_dict(
        as_series=False
    ) == {
        "index": [1, 2],
        "a": [2, 1],
    }
    assert df.unique(keep="first", subset="a", maintain_order=True).to_dict(
        as_series=False
    ) == {
        "index": [0, 1],
        "a": [1, 2],
    }


def test_group_by_dynamic_flat_agg_4814() -> None:
    df = pl.DataFrame({"a": [1, 2, 2], "b": [1, 8, 12]}).set_sorted("a")

    assert df.group_by_dynamic("a", every="1i", period="2i").agg(
        [
            (pl.col("b").sum() / pl.col("a").sum()).alias("sum_ratio_1"),
            (pl.col("b").last() / pl.col("a").last()).alias("last_ratio_1"),
            (pl.col("b") / pl.col("a")).last().alias("last_ratio_2"),
        ]
    ).to_dict(as_series=False) == {
        "a": [1, 2],
        "sum_ratio_1": [4.2, 5.0],
        "last_ratio_1": [6.0, 6.0],
        "last_ratio_2": [6.0, 6.0],
    }


@pytest.mark.parametrize(
    ("every", "period"),
    [
        ("10s", timedelta(seconds=100)),
        (timedelta(seconds=10), "100s"),
    ],
)
@pytest.mark.parametrize("time_zone", [None, "UTC", "Asia/Kathmandu"])
def test_group_by_dynamic_overlapping_groups_flat_apply_multiple_5038(
    every: str | timedelta, period: str | timedelta, time_zone: str | None
) -> None:
    res = (
        (
            pl.DataFrame(
                {
                    "a": [
                        datetime(2021, 1, 1) + timedelta(seconds=2**i)
                        for i in range(10)
                    ],
                    "b": [float(i) for i in range(10)],
                }
            )
            .with_columns(pl.col("a").dt.replace_time_zone(time_zone))
            .lazy()
            .set_sorted("a")
            .group_by_dynamic("a", every=every, period=period)
            .agg([pl.col("b").var().sqrt().alias("corr")])
        )
        .collect()
        .sum()
        .to_dict(as_series=False)
    )

    assert res["corr"] == pytest.approx([6.988674024215477])
    assert res["a"] == [None]


def test_take_in_group_by() -> None:
    df = pl.DataFrame({"group": [1, 1, 1, 2, 2, 2], "values": [10, 200, 3, 40, 500, 6]})
    assert df.group_by("group").agg(
        pl.col("values").get(1) - pl.col("values").get(2)
    ).sort("group").to_dict(as_series=False) == {"group": [1, 2], "values": [197, 494]}


def test_group_by_wildcard() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [1, 2],
        }
    )
    assert df.group_by([pl.col("*")], maintain_order=True).agg(
        [pl.col("a").first().name.suffix("_agg")]
    ).to_dict(as_series=False) == {"a": [1, 2], "b": [1, 2], "a_agg": [1, 2]}


def test_group_by_all_masked_out() -> None:
    df = pl.DataFrame(
        {
            "val": pl.Series(
                [None, None, None, None], dtype=pl.Categorical, nan_to_null=True
            ).set_sorted(),
            "col": [4, 4, 4, 4],
        }
    )
    parts = df.partition_by("val")
    assert len(parts) == 1
    assert_frame_equal(parts[0], df)


def test_group_by_null_propagation_6185() -> None:
    df_1 = pl.DataFrame({"A": [0, 0], "B": [1, 2]})

    expr = pl.col("A").filter(pl.col("A") > 0)

    expected = {"B": [1, 2], "A": [None, None]}
    assert (
        df_1.group_by("B")
        .agg((expr - expr.mean()).mean())
        .sort("B")
        .to_dict(as_series=False)
        == expected
    )


def test_group_by_when_then_with_binary_and_agg_in_pred_6202() -> None:
    df = pl.DataFrame(
        {"code": ["a", "b", "b", "b", "a"], "xx": [1.0, -1.5, -0.2, -3.9, 3.0]}
    )
    assert (
        df.group_by("code", maintain_order=True).agg(
            [pl.when(pl.col("xx") > pl.min("xx")).then(True).otherwise(False)]
        )
    ).to_dict(as_series=False) == {
        "code": ["a", "b"],
        "literal": [[False, True], [True, True, False]],
    }


def test_group_by_binary_agg_with_literal() -> None:
    df = pl.DataFrame({"id": ["a", "a", "b", "b"], "value": [1, 2, 3, 4]})

    out = df.group_by("id", maintain_order=True).agg(
        pl.col("value") + pl.Series([1, 3])
    )
    assert out.to_dict(as_series=False) == {"id": ["a", "b"], "value": [[2, 5], [4, 7]]}

    out = df.group_by("id", maintain_order=True).agg(pl.col("value") + pl.lit(1))
    assert out.to_dict(as_series=False) == {"id": ["a", "b"], "value": [[2, 3], [4, 5]]}

    out = df.group_by("id", maintain_order=True).agg(pl.lit(1) + pl.lit(2))
    assert out.to_dict(as_series=False) == {"id": ["a", "b"], "literal": [3, 3]}

    out = df.group_by("id", maintain_order=True).agg(pl.lit(1) + pl.Series([2, 3]))
    assert out.to_dict(as_series=False) == {
        "id": ["a", "b"],
        "literal": [[3, 4], [3, 4]],
    }

    out = df.group_by("id", maintain_order=True).agg(
        value=pl.lit(pl.Series([1, 2])) + pl.lit(pl.Series([3, 4]))
    )
    assert out.to_dict(as_series=False) == {"id": ["a", "b"], "value": [[4, 6], [4, 6]]}


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [pl.Int32, pl.UInt32])
def test_overflow_mean_partitioned_group_by_5194(dtype: PolarsDataType) -> None:
    df = pl.DataFrame(
        [
            pl.Series("data", [10_00_00_00] * 100_000, dtype=dtype),
            pl.Series("group", [1, 2] * 50_000, dtype=dtype),
        ]
    )
    result = df.group_by("group").agg(pl.col("data").mean()).sort(by="group")
    expected = {"group": [1, 2], "data": [10000000.0, 10000000.0]}
    assert result.to_dict(as_series=False) == expected


# https://github.com/pola-rs/polars/issues/7181
def test_group_by_multiple_column_reference() -> None:
    df = pl.DataFrame(
        {
            "gr": ["a", "b", "a", "b", "a", "b"],
            "val": [1, 20, 100, 2000, 10000, 200000],
        }
    )
    result = df.group_by("gr").agg(
        pl.col("val") + pl.col("val").shift().fill_null(0),
    )

    assert result.sort("gr").to_dict(as_series=False) == {
        "gr": ["a", "b"],
        "val": [[1, 101, 10100], [20, 2020, 202000]],
    }


@pytest.mark.parametrize(
    ("aggregation", "args", "expected_values", "expected_dtype"),
    [
        ("first", [], [1, None], pl.Int64),
        ("last", [], [1, None], pl.Int64),
        ("max", [], [1, None], pl.Int64),
        ("mean", [], [1.0, None], pl.Float64),
        ("median", [], [1.0, None], pl.Float64),
        ("min", [], [1, None], pl.Int64),
        ("n_unique", [], [1, 0], pl.get_index_type()),
        ("quantile", [0.5], [1.0, None], pl.Float64),
    ],
)
def test_group_by_empty_groups(
    aggregation: str,
    args: list[object],
    expected_values: list[object],
    expected_dtype: pl.DataType,
) -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [1, 2]})
    result = df.group_by("b", maintain_order=True).agg(
        getattr(pl.col("a").filter(pl.col("b") != 2), aggregation)(*args)
    )
    expected = pl.DataFrame({"b": [1, 2], "a": expected_values}).with_columns(
        pl.col("a").cast(expected_dtype)
    )
    assert_frame_equal(result, expected)


# https://github.com/pola-rs/polars/issues/8663
def test_perfect_hash_table_null_values() -> None:
    # fmt: off
    values = ["3", "41", "17", "5", "26", "27", "43", "45", "41", "13", "45", "48", "17", "22", "31", "25", "28", "13", "7", "26", "17", "4", "43", "47", "30", "28", "8", "27", "6", "7", "26", "11", "37", "29", "49", "20", "29", "28", "23", "9", None, "38", "19", "7", "38", "3", "30", "37", "41", "5", "16", "26", "31", "6", "25", "11", "17", "31", "31", "20", "26", None, "39", "10", "38", "4", "39", "15", "13", "35", "38", "11", "39", "11", "48", "36", "18", "11", "34", "16", "28", "9", "37", "8", "17", "48", "44", "28", "25", "30", "37", "30", "18", "12", None, "27", "10", "3", "16", "27", "6"]
    groups = ["3", "41", "17", "5", "26", "27", "43", "45", "13", "48", "22", "31", "25", "28", "7", "4", "47", "30", "8", "6", "11", "37", "29", "49", "20", "23", "9", None, "38", "19", "16", "39", "10", "15", "35", "36", "18", "34", "44", "12"]
    # fmt: on

    s = pl.Series("a", values, dtype=pl.Categorical)

    result = (
        s.to_frame("a").group_by("a", maintain_order=True).agg(pl.col("a").alias("agg"))
    )

    agg_values = [
        ["3", "3", "3"],
        ["41", "41", "41"],
        ["17", "17", "17", "17", "17"],
        ["5", "5"],
        ["26", "26", "26", "26", "26"],
        ["27", "27", "27", "27"],
        ["43", "43"],
        ["45", "45"],
        ["13", "13", "13"],
        ["48", "48", "48"],
        ["22"],
        ["31", "31", "31", "31"],
        ["25", "25", "25"],
        ["28", "28", "28", "28", "28"],
        ["7", "7", "7"],
        ["4", "4"],
        ["47"],
        ["30", "30", "30", "30"],
        ["8", "8"],
        ["6", "6", "6"],
        ["11", "11", "11", "11", "11"],
        ["37", "37", "37", "37"],
        ["29", "29"],
        ["49"],
        ["20", "20"],
        ["23"],
        ["9", "9"],
        [None, None, None],
        ["38", "38", "38", "38"],
        ["19"],
        ["16", "16", "16"],
        ["39", "39", "39"],
        ["10", "10"],
        ["15"],
        ["35"],
        ["36"],
        ["18", "18"],
        ["34"],
        ["44"],
        ["12"],
    ]
    expected = pl.DataFrame(
        {
            "a": groups,
            "agg": agg_values,
        },
        schema={"a": pl.Categorical, "agg": pl.List(pl.Categorical)},
    )
    assert_frame_equal(result, expected)


def test_group_by_partitioned_ending_cast(plmonkeypatch: PlMonkeyPatch) -> None:
    plmonkeypatch.setenv("POLARS_FORCE_PARTITION", "1")
    df = pl.DataFrame({"a": [1] * 5, "b": [1] * 5})
    out = df.group_by(["a", "b"]).agg(pl.len().cast(pl.Int64).alias("num"))
    expected = pl.DataFrame({"a": [1], "b": [1], "num": [5]})
    assert_frame_equal(out, expected)


def test_group_by_series_partitioned(partition_limit: int) -> None:
    # test 15354
    df = pl.DataFrame([0, 0] * partition_limit)
    groups = pl.Series([0, 1] * partition_limit)
    df.group_by(groups).agg(pl.all().is_not_null().sum())


def test_group_by_list_scalar_11749() -> None:
    df = pl.DataFrame(
        {
            "group_name": ["a;b", "a;b", "c;d", "c;d", "a;b", "a;b"],
            "parent_name": ["a", "b", "c", "d", "a", "b"],
            "measurement": [
                ["x1", "x2"],
                ["x1", "x2"],
                ["y1", "y2"],
                ["z1", "z2"],
                ["x1", "x2"],
                ["x1", "x2"],
            ],
        }
    )
    assert (
        df.group_by("group_name").agg(
            (pl.col("measurement").first() == pl.col("measurement")).alias("eq"),
        )
    ).sort("group_name").to_dict(as_series=False) == {
        "group_name": ["a;b", "c;d"],
        "eq": [[True, True, True, True], [True, False]],
    }


def test_group_by_with_expr_as_key() -> None:
    gb = pl.select(x=1).group_by(pl.col("x").alias("key"))
    result = gb.agg(pl.all().first())
    expected = gb.agg(pl.first("x"))
    assert_frame_equal(result, expected)

    # tests: 11766
    result = gb.head(0)
    expected = gb.agg(pl.col("x").head(0)).explode("x")
    assert_frame_equal(result, expected)

    result = gb.tail(0)
    expected = gb.agg(pl.col("x").tail(0)).explode("x")
    assert_frame_equal(result, expected)


def test_lazy_group_by_reuse_11767() -> None:
    lgb = pl.select(x=1).lazy().group_by("x")
    a = lgb.len()
    b = lgb.len()
    assert_frame_equal(a, b)


def test_group_by_double_on_empty_12194() -> None:
    df = pl.DataFrame({"group": [1], "x": [1]}).clear()
    squared_deviation_sum = ((pl.col("x") - pl.col("x").mean()) ** 2).sum()
    assert df.group_by("group").agg(squared_deviation_sum).schema == OrderedDict(
        [("group", pl.Int64), ("x", pl.Float64)]
    )


def test_group_by_when_then_no_aggregation_predicate() -> None:
    df = pl.DataFrame(
        {
            "key": ["aa", "aa", "bb", "bb", "aa", "aa"],
            "val": [-3, -2, 1, 4, -3, 5],
        }
    )
    assert df.group_by("key").agg(
        pos=pl.when(pl.col("val") >= 0).then(pl.col("val")).sum(),
        neg=pl.when(pl.col("val") < 0).then(pl.col("val")).sum(),
    ).sort("key").to_dict(as_series=False) == {
        "key": ["aa", "bb"],
        "pos": [5, 5],
        "neg": [-8, 0],
    }


def test_group_by_apply_first_input_is_literal() -> None:
    df = pl.DataFrame({"x": [1, 2, 3, 4, 5], "g": [1, 1, 2, 2, 2]})
    pow = df.group_by("g").agg(2 ** pl.col("x"))
    assert pow.sort("g").to_dict(as_series=False) == {
        "g": [1, 2],
        "literal": [[2.0, 4.0], [8.0, 16.0, 32.0]],
    }


def test_group_by_all_12869() -> None:
    df = pl.DataFrame({"a": [1]})
    result = next(iter(df.group_by(pl.all())))[1]
    assert_frame_equal(df, result)


def test_group_by_named() -> None:
    df = pl.DataFrame({"a": [1, 1, 2, 2, 3, 3], "b": range(6)})
    result = df.group_by(z=pl.col("a") * 2, maintain_order=True).agg(pl.col("b").min())
    expected = df.group_by((pl.col("a") * 2).alias("z"), maintain_order=True).agg(
        pl.col("b").min()
    )
    assert_frame_equal(result, expected)


def test_group_by_with_null() -> None:
    df = pl.DataFrame(
        {"a": [None, None, None, None], "b": [1, 1, 2, 2], "c": ["x", "y", "z", "u"]}
    )
    expected = pl.DataFrame(
        {"a": [None, None], "b": [1, 2], "c": [["x", "y"], ["z", "u"]]}
    )
    output = df.group_by(["a", "b"], maintain_order=True).agg(pl.col("c"))
    assert_frame_equal(expected, output)


def test_partitioned_group_by_14954(plmonkeypatch: PlMonkeyPatch) -> None:
    plmonkeypatch.setenv("POLARS_FORCE_PARTITION", "1")
    assert (
        pl.DataFrame({"a": range(20)})
        .select(pl.col("a") % 2)
        .group_by("a")
        .agg(
            (pl.col("a") > 1000).alias("a > 1000"),
        )
    ).sort("a").to_dict(as_series=False) == {
        "a": [0, 1],
        "a > 1000": [
            [False, False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False, False],
        ],
    }


def test_partitioned_group_by_nulls_mean_21838() -> None:
    size = 10
    a = [1 for i in range(size)] + [2 for i in range(size)] + [3 for i in range(size)]
    b = [1 for i in range(size)] + [None for i in range(size * 2)]
    df = pl.DataFrame({"a": a, "b": b})
    assert df.group_by("a").mean().sort("a").to_dict(as_series=False) == {
        "a": [1, 2, 3],
        "b": [1.0, None, None],
    }


def test_aggregated_scalar_elementwise_15602() -> None:
    df = pl.DataFrame({"group": [1, 2, 1]})

    out = df.group_by("group", maintain_order=True).agg(
        foo=pl.col("group").is_between(1, pl.max("group"))
    )
    expected = pl.DataFrame({"group": [1, 2], "foo": [[True, True], [True]]})
    assert_frame_equal(out, expected)


def test_group_by_multiple_null_cols_15623() -> None:
    df = pl.DataFrame(schema={"a": pl.Null, "b": pl.Null}).group_by(pl.all()).len()
    assert df.is_empty()


@pytest.mark.release
def test_categorical_vs_str_group_by() -> None:
    # this triggers the perfect hash table
    s = pl.Series("a", np.random.randint(0, 50, 100))
    s_with_nulls = pl.select(
        pl.when(s < 3).then(None).otherwise(s).alias("a")
    ).to_series()

    for s_ in [s, s_with_nulls]:
        s_ = s_.cast(str)
        cat_out = (
            s_.cast(pl.Categorical)
            .to_frame("a")
            .group_by("a")
            .agg(pl.first().alias("first"))
        )

        str_out = s_.to_frame("a").group_by("a").agg(pl.first().alias("first"))
        cat_out.with_columns(pl.col("a").cast(str))
        assert_frame_equal(
            cat_out.with_columns(
                pl.col("a").cast(str), pl.col("first").cast(pl.List(str))
            ).sort("a"),
            str_out.sort("a"),
        )


@pytest.mark.release
def test_boolean_min_max_agg() -> None:
    np.random.seed(0)
    idx = np.random.randint(0, 500, 1000)
    c = np.random.randint(0, 500, 1000) > 250

    df = pl.DataFrame({"idx": idx, "c": c})
    aggs = [pl.col("c").min().alias("c_min"), pl.col("c").max().alias("c_max")]

    result = df.group_by("idx").agg(aggs).sum()

    schema = {"idx": pl.Int64, "c_min": pl.UInt32, "c_max": pl.UInt32}
    expected = pl.DataFrame(
        {
            "idx": [107583],
            "c_min": [120],
            "c_max": [321],
        },
        schema=schema,
    )
    assert_frame_equal(result, expected)

    nulls = np.random.randint(0, 500, 1000) < 100

    result = (
        df.with_columns(c=pl.when(pl.lit(nulls)).then(None).otherwise(pl.col("c")))
        .group_by("idx")
        .agg(aggs)
        .sum()
    )

    expected = pl.DataFrame(
        {
            "idx": [107583],
            "c_min": [133],
            "c_max": [276],
        },
        schema=schema,
    )
    assert_frame_equal(result, expected)


def test_partitioned_group_by_chunked(partition_limit: int) -> None:
    n = partition_limit
    df1 = pl.DataFrame(np.random.randn(n, 2))
    df2 = pl.DataFrame(np.random.randn(n, 2))
    gps = pl.Series(name="oo", values=[0] * n + [1] * n)
    df = pl.concat([df1, df2], rechunk=False)
    assert_frame_equal(
        df.group_by(gps).sum().sort("oo"),
        df.rechunk().group_by(gps, maintain_order=True).sum(),
    )


def test_schema_on_agg() -> None:
    lf = pl.LazyFrame({"a": ["x", "x", "y", "n"], "b": [1, 2, 3, 4]})

    result = lf.group_by("a").agg(
        pl.col("b").min().alias("min"),
        pl.col("b").max().alias("max"),
        pl.col("b").sum().alias("sum"),
        pl.col("b").first().alias("first"),
        pl.col("b").last().alias("last"),
        pl.col("b").item().alias("item"),
    )
    expected_schema = {
        "a": pl.String,
        "min": pl.Int64,
        "max": pl.Int64,
        "sum": pl.Int64,
        "first": pl.Int64,
        "last": pl.Int64,
        "item": pl.Int64,
    }
    assert result.collect_schema() == expected_schema


def test_group_by_schema_err() -> None:
    lf = pl.LazyFrame({"foo": [None, 1, 2], "bar": [1, 2, 3]})
    with pytest.raises(ColumnNotFoundError):
        lf.group_by("not-existent").agg(
            pl.col("bar").max().alias("max_bar")
        ).collect_schema()


@pytest.mark.parametrize(
    ("data", "expr", "expected_select", "expected_gb"),
    [
        (
            {"x": ["x"], "y": ["y"]},
            pl.coalesce(pl.col("x"), pl.col("y")),
            {"x": pl.String},
            {"x": pl.List(pl.String)},
        ),
        (
            {"x": [True]},
            pl.col("x").sum(),
            {"x": pl.get_index_type()},
            {"x": pl.get_index_type()},
        ),
        (
            {"a": [[1, 2]]},
            pl.col("a").list.sum(),
            {"a": pl.Int64},
            {"a": pl.List(pl.Int64)},
        ),
    ],
)
def test_schemas(
    data: dict[str, list[Any]],
    expr: pl.Expr,
    expected_select: dict[str, PolarsDataType],
    expected_gb: dict[str, PolarsDataType],
) -> None:
    df = pl.DataFrame(data)

    # test selection schema
    schema = df.select(expr).schema
    for key, dtype in expected_select.items():
        assert schema[key] == dtype

    # test group_by schema
    schema = df.group_by(pl.lit(1)).agg(expr).schema
    for key, dtype in expected_gb.items():
        assert schema[key] == dtype


def test_lit_iter_schema() -> None:
    df = pl.DataFrame(
        {
            "key": ["A", "A", "A", "A"],
            "dates": [
                date(1970, 1, 1),
                date(1970, 1, 1),
                date(1970, 1, 2),
                date(1970, 1, 3),
            ],
        }
    )

    result = df.group_by("key").agg(pl.col("dates").unique() + timedelta(days=1))
    expected = {
        "key": ["A"],
        "dates": [[date(1970, 1, 2), date(1970, 1, 3), date(1970, 1, 4)]],
    }
    assert result.to_dict(as_series=False) == expected


def test_absence_off_null_prop_8224() -> None:
    # a reminder to self to not do null propagation
    # it is inconsistent and makes output dtype
    # dependent of the data, big no!

    def sub_col_min(column: str, min_column: str) -> pl.Expr:
        return pl.col(column).sub(pl.col(min_column).min())

    df = pl.DataFrame(
        {
            "group": [1, 1, 2, 2],
            "vals_num": [10.0, 11.0, 12.0, 13.0],
            "vals_partial": [None, None, 12.0, 13.0],
            "vals_null": [None, None, None, None],
        }
    )

    q = (
        df.lazy()
        .group_by("group")
        .agg(
            sub_col_min("vals_num", "vals_num").alias("sub_num"),
            sub_col_min("vals_num", "vals_partial").alias("sub_partial"),
            sub_col_min("vals_num", "vals_null").alias("sub_null"),
        )
    )

    assert q.collect().dtypes == [
        pl.Int64,
        pl.List(pl.Float64),
        pl.List(pl.Float64),
        pl.List(pl.Float64),
    ]


@pytest.mark.parametrize("maintain_order", [False, True])
def test_grouped_slice_literals(maintain_order: bool) -> None:
    df = pl.DataFrame({"idx": [1, 2, 3]})
    q = (
        df.lazy()
        .group_by(True, maintain_order=maintain_order)
        .agg(
            x=pl.lit([1, 2]).slice(
                -1, 1
            ),  # slices a list of 1 element, so remains the same element
            x2=pl.lit(pl.Series([1, 2])).slice(-1, 1),
            x3=pl.lit(pl.Series([[1, 2]])).slice(-1, 1),
        )
    )
    out = q.collect()
    expected = pl.DataFrame(
        {"literal": [True], "x": [[[1, 2]]], "x2": [[2]], "x3": [[[1, 2]]]}
    )
    assert_frame_equal(
        out,
        expected,
        check_row_order=maintain_order,
    )
    assert q.collect_schema() == q.collect().schema


def test_positional_by_with_list_or_tuple_17540() -> None:
    with pytest.raises(TypeError, match="Hint: if you"):
        pl.DataFrame({"a": [1, 2, 3]}).group_by(by=["a"])
    with pytest.raises(TypeError, match="Hint: if you"):
        pl.LazyFrame({"a": [1, 2, 3]}).group_by(by=["a"])


def test_group_by_agg_19173() -> None:
    df = pl.DataFrame({"x": [1.0], "g": [0]})
    out = df.head(0).group_by("g").agg((pl.col.x - pl.col.x.sum() * pl.col.x) ** 2)
    assert out.to_dict(as_series=False) == {"g": [], "x": []}
    assert out.schema == pl.Schema([("g", pl.Int64), ("x", pl.List(pl.Float64))])


def test_group_by_map_groups_slice_pushdown_20002() -> None:
    schema = {
        "a": pl.Int8,
        "b": pl.UInt8,
    }

    df = (
        pl.LazyFrame(
            data={"a": [1, 2, 3, 4, 5], "b": [90, 80, 70, 60, 50]},
            schema=schema,
        )
        .group_by("a", maintain_order=True)
        .map_groups(lambda df: df * 2.0, schema=schema)
        .head(3)
        .collect()
    )

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "a": [2.0, 4.0, 6.0],
                "b": [180.0, 160.0, 140.0],
            }
        ),
    )


@typing.no_type_check
def test_group_by_lit_series(capfd: Any, plmonkeypatch: PlMonkeyPatch) -> None:
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    n = 10
    df = pl.DataFrame({"x": np.ones(2 * n), "y": n * list(range(2))})
    a = np.ones(n, dtype=float)
    df.lazy().group_by("y").agg(pl.col("x").dot(a)).collect()
    captured = capfd.readouterr().err
    assert "are not partitionable" in captured


def test_group_by_list_column() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [[1, 2], [3], [1, 2]]})
    result = df.group_by("b").agg(pl.sum("a")).sort("b")
    expected = pl.DataFrame({"b": [[1, 2], [3]], "a": [4, 2]})
    assert_frame_equal(result, expected)


def test_enum_perfect_group_by_21360() -> None:
    dtype = pl.Enum(categories=["a", "b"])

    assert_frame_equal(
        pl.from_dicts([{"col": "a"}], schema={"col": dtype})
        .group_by("col")
        .agg(pl.len()),
        pl.DataFrame(
            [
                pl.Series("col", ["a"], dtype),
                pl.Series("len", [1], get_index_type()),
            ]
        ),
    )


def test_partitioned_group_by_21634(partition_limit: int) -> None:
    n = partition_limit
    df = pl.DataFrame({"grp": [1] * n, "x": [1] * n})
    assert df.group_by("grp", True).agg().to_dict(as_series=False) == {
        "grp": [1],
        "literal": [True],
    }


def test_group_by_cse_dup_key_alias_22238() -> None:
    df = pl.LazyFrame({"a": [1, 1, 2, 2, -1], "x": [0, 1, 2, 3, 10]})
    result = df.group_by(
        pl.col("a").abs(),
        pl.col("a").abs().alias("a_with_alias"),
    ).agg(pl.col("x").sum())
    assert_frame_equal(
        result.collect(),
        pl.DataFrame({"a": [1, 2], "a_with_alias": [1, 2], "x": [11, 5]}),
        check_row_order=False,
    )


def test_group_by_22328() -> None:
    N = 20

    df1 = pl.select(
        x=pl.repeat(1, N // 2).append(pl.repeat(2, N // 2)).shuffle(),
        y=pl.lit(3.0, pl.Float32),
    ).lazy()

    df2 = pl.select(x=pl.repeat(4, N)).lazy()

    assert (
        df2.join(df1.group_by("x").mean().with_columns(z="y"), how="left", on="x")
        .with_columns(pl.col("z").fill_null(0))
        .collect()
    ).shape == (20, 3)


@pytest.mark.parametrize("maintain_order", [False, True])
def test_group_by_arrays_22574(maintain_order: bool) -> None:
    assert_frame_equal(
        pl.Series("a", [[1], [2], [2]], pl.Array(pl.Int64, 1))
        .to_frame()
        .group_by("a", maintain_order=maintain_order)
        .agg(pl.len()),
        pl.DataFrame(
            [
                pl.Series("a", [[1], [2]], pl.Array(pl.Int64, 1)),
                pl.Series("len", [1, 2], pl.get_index_type()),
            ]
        ),
        check_row_order=maintain_order,
    )

    assert_frame_equal(
        pl.Series(
            "a", [[[1, 2]], [[2, 3]], [[2, 3]]], pl.Array(pl.Array(pl.Int64, 2), 1)
        )
        .to_frame()
        .group_by("a", maintain_order=maintain_order)
        .agg(pl.len()),
        pl.DataFrame(
            [
                pl.Series(
                    "a", [[[1, 2]], [[2, 3]]], pl.Array(pl.Array(pl.Int64, 2), 1)
                ),
                pl.Series("len", [1, 2], pl.get_index_type()),
            ]
        ),
        check_row_order=maintain_order,
    )


def test_group_by_empty_rows_with_literal_21959() -> None:
    out = (
        pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [1, 1, 3]})
        .filter(pl.col("c") == 99)
        .group_by(pl.lit(1).alias("d"), pl.col("a"), pl.col("b"))
        .agg()
        .collect()
    )
    expected = pl.DataFrame(
        {"d": [], "a": [], "b": []},
        schema={"d": pl.Int32, "a": pl.Int64, "b": pl.Int64},
    )
    assert_frame_equal(out, expected)


def test_group_by_empty_dtype_22716() -> None:
    df = pl.DataFrame(schema={"a": pl.String, "b": pl.Int64})
    out = df.group_by("a").agg(x=(pl.col("b") == pl.int_range(pl.len())).all())
    assert_frame_equal(out, pl.DataFrame(schema={"a": pl.String, "x": pl.Boolean}))


def test_group_by_implode_22870() -> None:
    out = (
        pl.DataFrame({"x": ["a", "b"]})
        .group_by(pl.col.x)
        .agg(
            y=pl.col.x.replace_strict(
                pl.lit(pl.Series(["a", "b"])).implode(),
                pl.lit(pl.Series([1, 2])).implode(),
                default=-1,
            )
        )
    )
    assert_frame_equal(
        out,
        pl.DataFrame({"x": ["a", "b"], "y": [[1], [2]]}),
        check_row_order=False,
    )


# Note: the underlying bug is not guaranteed to manifest itself as it depends
# on the internal group order, i.e., for the bug to materialize, there must be
# empty groups before the non-empty group
def test_group_by_empty_groups_23338() -> None:
    # We need one non-empty and many groups
    df = pl.DataFrame(
        {
            "k": [10, 10, 20, 30, 40, 50, 60, 70, 80, 90],
            "a": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    )
    out = df.group_by("k").agg(
        pl.col("a").filter(pl.col("a") == 1).fill_nan(None).sum()
    )
    expected = df.group_by("k").agg(pl.col("a").filter(pl.col("a") == 1).sum())
    assert_frame_equal(out.sort("k"), expected.sort("k"))


def test_group_by_filter_all_22955() -> None:
    df = pl.DataFrame(
        {
            "grp": [1, 2, 3, 4, 5],
            "value": [10, 20, 30, 40, 50],
        }
    )

    assert_frame_equal(
        df.group_by("grp").agg(
            pl.all().filter(pl.col("value") > 20),
        ),
        pl.DataFrame(
            {
                "grp": [1, 2, 3, 4, 5],
                "value": [[], [], [30], [40], [50]],
            }
        ),
        check_row_order=False,
    )


@pytest.mark.parametrize("maintain_order", [False, True])
def test_group_by_series_lit_22103(maintain_order: bool) -> None:
    df = pl.DataFrame(
        {
            "g": [0, 1],
        }
    )
    assert_frame_equal(
        df.group_by("g", maintain_order=maintain_order).agg(
            foo=pl.lit(pl.Series([42, 2, 3]))
        ),
        pl.DataFrame(
            {
                "g": [0, 1],
                "foo": [[42, 2, 3], [42, 2, 3]],
            }
        ),
        check_row_order=maintain_order,
    )


@pytest.mark.parametrize("maintain_order", [False, True])
def test_group_by_filter_sum_23897(maintain_order: bool) -> None:
    testdf = pl.DataFrame(
        {
            "id": [8113, 9110, 9110],
            "value": [None, None, 1.0],
            "weight": [1.0, 1.0, 1.0],
        }
    )

    w = pl.col("weight").filter(pl.col("value").is_finite())

    w = w / w.sum()

    result = w.sum()

    assert_frame_equal(
        testdf.group_by("id", maintain_order=maintain_order).agg(result),
        pl.DataFrame({"id": [8113, 9110], "weight": [0.0, 1.0]}),
        check_row_order=maintain_order,
    )


@pytest.mark.parametrize("maintain_order", [False, True])
def test_group_by_shift_filter_23910(maintain_order: bool) -> None:
    df = pl.DataFrame({"a": [3, 7, 5, 9, 2, 1], "b": [2, 2, 2, 3, 3, 1]})

    out = df.group_by("b", maintain_order=maintain_order).agg(
        pl.col("a").filter(pl.col("a") > pl.col("a").shift(1)).sum().alias("tt")
    )

    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "b": [2, 3, 1],
                "tt": [7, 0, 0],
            }
        ),
        check_row_order=maintain_order,
    )


@pytest.mark.parametrize("maintain_order", [False, True])
def test_group_by_having(maintain_order: bool) -> None:
    df = pl.DataFrame(
        {
            "grp": ["A", "A", "B", "B", "C", "C"],
            "value": [10, 15, 5, 15, 5, 10],
        }
    )

    result = (
        df.group_by("grp", maintain_order=maintain_order)
        .having(pl.col("value").mean() >= 10)
        .agg()
    )
    expected = pl.DataFrame({"grp": ["A", "B"]})
    assert_frame_equal(result, expected, check_row_order=maintain_order)


def test_group_by_tuple_typing_24112() -> None:
    df = pl.DataFrame({"id": ["a", "b", "a"], "val": [1, 2, 3]})
    for (id_,), _ in df.group_by("id"):
        _should_work: str = id_


def test_group_by_input_independent_with_len_23868() -> None:
    out = pl.DataFrame({"a": ["A", "B", "C"]}).group_by(pl.lit("G")).agg(pl.len())
    assert_frame_equal(
        out,
        pl.DataFrame(
            {"literal": "G", "len": 3},
            schema={"literal": pl.String, "len": pl.get_index_type()},
        ),
    )


@pytest.mark.parametrize("maintain_order", [False, True])
def test_group_by_head_tail_24215(maintain_order: bool) -> None:
    df = pl.DataFrame(
        {
            "station": ["A", "A", "B"],
            "num_rides": [1, 2, 3],
        }
    )
    expected = pl.DataFrame(
        {"station": ["A", "B"], "num_rides": [1.5, 3], "rides_per_day": [[1, 2], [3]]}
    )

    result = (
        df.group_by("station", maintain_order=maintain_order)
        .agg(
            cs.numeric().mean(),
            pl.col("num_rides").alias("rides_per_day"),
        )
        .group_by("station", maintain_order=maintain_order)
        .head(1)
    )
    assert_frame_equal(result, expected, check_row_order=maintain_order)

    result = (
        df.group_by("station", maintain_order=maintain_order)
        .agg(
            cs.numeric().mean(),
            pl.col("num_rides").alias("rides_per_day"),
        )
        .group_by("station", maintain_order=maintain_order)
        .tail(1)
    )
    assert_frame_equal(result, expected, check_row_order=maintain_order)


def test_slice_group_by_offset_24259() -> None:
    df = pl.DataFrame(
        {
            "letters": ["c", "c", "a", "c", "a", "b", "d"],
            "nrs": [1, 2, 3, 4, 5, 6, None],
        }
    )
    assert df.group_by("letters").agg(
        x=pl.col("nrs").drop_nulls(),
        tail=pl.col("nrs").drop_nulls().tail(1),
    ).sort("letters").to_dict(as_series=False) == {
        "letters": ["a", "b", "c", "d"],
        "x": [[3, 5], [6], [1, 2, 4], []],
        "tail": [[5], [6], [4], []],
    }


def test_group_by_first_nondet_24278() -> None:
    values = [
        96, 86, 0, 86, 43, 50, 9, 14, 98, 39, 93, 7, 71, 1, 93, 41, 56,
        56, 93, 41, 58, 91, 81, 29, 81, 68, 5, 9, 32, 93, 78, 34, 17, 40,
        14, 2, 52, 77, 81, 4, 56, 42, 64, 12, 29, 58, 71, 98, 32, 49, 34,
        86, 29, 94, 37, 21, 41, 36, 9, 72, 23, 28, 71, 9, 66, 72, 84, 81,
        23, 12, 64, 57, 99, 15, 77, 38, 95, 64, 13, 91, 43, 61, 70, 47,
        39, 75, 47, 93, 45, 1, 95, 55, 29, 5, 83, 8, 3, 6, 45, 84,
    ]  # fmt: skip
    q = (
        pl.LazyFrame({"a": values, "idx": range(100)})
        .group_by("a")
        .agg(pl.col.idx.first())
        .select(a=pl.col.idx)
    )

    fst_value = q.collect().to_series().sum()
    for _ in range(10):
        assert q.collect().to_series().sum() == fst_value


@pytest.mark.parametrize("maintain_order", [False, True])
def test_group_by_agg_on_lit(maintain_order: bool) -> None:
    fs: list[Callable[[Expr], Expr]] = [
        Expr.min,
        Expr.max,
        Expr.mean,
        Expr.sum,
        Expr.len,
        Expr.count,
        Expr.first,
        Expr.last,
        Expr.n_unique,
        Expr.implode,
        Expr.std,
        Expr.var,
        lambda e: e.quantile(0.5),
        Expr.nan_min,
        Expr.nan_max,
        Expr.skew,
        Expr.null_count,
        Expr.product,
        lambda e: pl.corr(e, e),
    ]

    df = pl.DataFrame({"a": [1, 2], "b": [1, 1]})

    assert_frame_equal(
        df.group_by("a", maintain_order=maintain_order).agg(
            f(pl.lit(1)).alias(f"c{i}") for i, f in enumerate(fs)
        ),
        pl.select(
            [pl.lit(pl.Series("a", [1, 2]))]
            + [f(pl.lit(1)).alias(f"c{i}") for i, f in enumerate(fs)]
        ),
        check_row_order=maintain_order,
    )

    df = pl.DataFrame({"a": [1, 2], "b": [None, 1]})

    assert_frame_equal(
        df.group_by("a", maintain_order=maintain_order).agg(
            f(pl.lit(1)).alias(f"c{i}") for i, f in enumerate(fs)
        ),
        pl.select(
            [pl.lit(pl.Series("a", [1, 2]))]
            + [f(pl.lit(1)).alias(f"c{i}") for i, f in enumerate(fs)]
        ),
        check_row_order=maintain_order,
    )


def test_group_by_cum_sum_key_24489() -> None:
    df = pl.LazyFrame({"x": [1, 2]})
    out = df.group_by((pl.col.x > 1).cum_sum()).agg().collect()
    expected = pl.DataFrame({"x": [0, 1]}, schema={"x": pl.UInt32})
    assert_frame_equal(out, expected, check_row_order=False)


@pytest.mark.parametrize("maintain_order", [False, True])
def test_double_aggregations(maintain_order: bool) -> None:
    fs: list[Callable[[pl.Expr], pl.Expr]] = [
        Expr.min,
        Expr.max,
        Expr.mean,
        Expr.sum,
        Expr.len,
        Expr.count,
        Expr.first,
        Expr.last,
        Expr.n_unique,
        Expr.implode,
        Expr.std,
        Expr.var,
        lambda e: e.quantile(0.5),
        Expr.nan_min,
        Expr.nan_max,
        Expr.skew,
        Expr.null_count,
        Expr.product,
        lambda e: pl.corr(e, e),
    ]

    df = pl.DataFrame({"a": [1, 2], "b": [1, 1]})

    assert_frame_equal(
        df.group_by("a", maintain_order=maintain_order).agg(
            f(pl.col.b).alias(f"c{i}") for i, f in enumerate(fs)
        ),
        df.group_by("a", maintain_order=maintain_order).agg(
            f(pl.col.b.first()).alias(f"c{i}") for i, f in enumerate(fs)
        ),
        check_row_order=maintain_order,
    )

    df = pl.DataFrame({"a": [1, 2], "b": [None, 1]})

    assert_frame_equal(
        df.group_by("a", maintain_order=maintain_order).agg(
            f(pl.col.b).alias(f"c{i}") for i, f in enumerate(fs)
        ),
        df.group_by("a", maintain_order=maintain_order).agg(
            f(pl.col.b.first()).alias(f"c{i}") for i, f in enumerate(fs)
        ),
        check_row_order=maintain_order,
    )


def test_group_by_length_preserving_on_scalar() -> None:
    df = pl.DataFrame({"a": [[1], [2], [3]]})
    df = df.group_by(pl.lit(1, pl.Int64)).agg(
        a=pl.col.a.first().reverse(),
        b=pl.col.a.first(),
        c=pl.col.a.reverse(),
        d=pl.lit(1, pl.Int64).reverse(),
        e=pl.lit(1, pl.Int64).unique(),
    )

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "literal": [1],
                "a": [[1]],
                "b": [[1]],
                "c": [[[3], [2], [1]]],
                "d": [1],
                "e": [[1]],
            }
        ),
    )


def test_group_by_enum_min_max_18394() -> None:
    df = pl.DataFrame(
        {
            "id": ["a", "a", "b", "b", "c", "c"],
            "degree": ["low", "high", "high", "mid", "mid", "low"],
        }
    ).with_columns(pl.col("degree").cast(pl.Enum(["low", "mid", "high"])))
    out = df.group_by("id").agg(
        min_degree=pl.col("degree").min(),
        max_degree=pl.col("degree").max(),
    )
    expected = pl.DataFrame(
        {
            "id": ["a", "b", "c"],
            "min_degree": ["low", "mid", "low"],
            "max_degree": ["high", "high", "mid"],
        },
        schema={
            "id": pl.String,
            "min_degree": pl.Enum(["low", "mid", "high"]),
            "max_degree": pl.Enum(["low", "mid", "high"]),
        },
    )
    assert_frame_equal(out, expected, check_row_order=False)


@pytest.mark.parametrize("maintain_order", [False, True])
def test_group_by_filter_24838(maintain_order: bool) -> None:
    df = pl.DataFrame({"a": [1, 1, 2, 2, 3], "b": [1, 2, 1, 2, 1]})

    assert_frame_equal(
        df.group_by("a", maintain_order=maintain_order).agg(
            b=pl.lit(2, pl.Int64).filter(pl.col.b != 1)
        ),
        pl.DataFrame(
            [
                pl.Series("a", [1, 2, 3], pl.Int64),
                pl.Series("b", [[2], [2], []], pl.List(pl.Int64)),
            ]
        ),
        check_row_order=maintain_order,
    )


@pytest.mark.parametrize(
    "lhs",
    [
        pl.lit(2),
        pl.col.a,
        pl.col.a.first(),
        pl.col.a.reverse(),
        pl.col.a.fill_null(strategy="forward"),
    ],
)
@pytest.mark.parametrize(
    "rhs",
    [
        pl.col.b == 3,
        pl.col.b != 3,
        pl.col.b.reverse() == 3,
        pl.col.b.reverse() != 3,
        pl.col.b.fill_null(1) != 3,
        pl.col.b.fill_null(1) == 3,
        pl.lit(True),
        pl.lit(False),
        pl.lit(pl.Series([True])),
        pl.lit(pl.Series([False])),
        pl.lit(pl.Series([True])).first(),
        pl.lit(pl.Series([False])).first(),
    ],
)
@pytest.mark.parametrize(
    "agg",
    [
        Expr.implode,
        Expr.sum,
        Expr.first,
    ],
)
def test_group_by_filter_parametric(
    lhs: pl.Expr, rhs: pl.Expr, agg: Callable[[pl.Expr], pl.Expr]
) -> None:
    df = pl.DataFrame({"a": [1, 1, 2, 2, 3], "b": [1, 2, 1, 2, 1]})
    gb = df.group_by(pl.lit(1)).agg(a=agg(lhs.filter(rhs))).to_series(1)
    gb = gb.rename("a")
    sl = df.select(a=agg(lhs.filter(rhs))).to_series()
    assert_series_equal(gb, sl)


@given(s=series(name="a", min_size=1))
@pytest.mark.parametrize(
    ("expr", "is_scalar", "maintain_order"),
    [
        (pl.Expr.n_unique, True, True),
        (pl.Expr.unique, False, False),
        (lambda e: e.unique(maintain_order=True), False, True),
    ],
)
def test_group_by_unique_parametric(
    s: pl.Series,
    expr: Callable[[pl.Expr], pl.Expr],
    is_scalar: bool,
    maintain_order: bool,
) -> None:
    df = s.to_frame()

    sl = df.select(expr(pl.col.a))
    gb = df.group_by(pl.lit(1)).agg(expr(pl.col.a)).drop("literal")
    if not is_scalar:
        gb = gb.select(pl.col.a.explode())
    assert_frame_equal(sl, gb, check_row_order=maintain_order)

    # check scalar case
    sl_first = df.select(expr(pl.col.a.first()))
    gb = df.group_by(pl.lit(1)).agg(expr(pl.col.a.first())).drop("literal")
    if not is_scalar:
        gb = gb.select(pl.col.a.explode())
    assert_frame_equal(sl_first, gb, check_row_order=maintain_order)

    li = df.select(pl.col.a.implode().list.eval(expr(pl.element())))
    li = li.select(pl.col.a.explode())
    assert_frame_equal(sl, li, check_row_order=maintain_order)


@pytest.mark.parametrize(
    "expr",
    [
        pl.Expr.any,
        pl.Expr.all,
        lambda e: e.any(ignore_nulls=False),
        lambda e: e.all(ignore_nulls=False),
    ],
)
def test_group_by_any_all(expr: Callable[[pl.Expr], pl.Expr]) -> None:
    combinations = [
        [True, None],
        [None, None],
        [False, None],
        [True, True],
        [False, False],
        [True, False],
    ]

    cl = cs.starts_with("x")
    df = pl.DataFrame(
        [pl.Series("g", [1, 1])]
        + [pl.Series(f"x{i}", c, pl.Boolean()) for i, c in enumerate(combinations)]
    )

    # verify that we are actually calculating something
    assert len(df.lazy().select(expr(cl)).collect_schema()) == len(combinations)

    assert_frame_equal(
        df.select(expr(cl)),
        df.group_by(lit=pl.lit(1)).agg(expr(cl)).drop("lit"),
    )

    assert_frame_equal(
        df.select(expr(cl)),
        df.group_by("g").agg(expr(cl)).drop("g"),
    )

    assert_frame_equal(
        df.select(expr(cl)),
        df.select(cl.implode().list.agg(expr(pl.element()))),
    )

    df = pl.Schema({"x": pl.Boolean()}).to_frame()

    assert_frame_equal(
        df.select(expr(cl)),
        pl.DataFrame({"x": [None]})
        .group_by(lit=pl.lit(1))
        .agg(expr(pl.lit(pl.Series("x", [], pl.Boolean()))))
        .drop("lit"),
    )

    assert_frame_equal(
        df.select(expr(cl)),
        df.select(cl.implode().list.agg(expr(pl.element()))),
    )


@given(
    s=series(
        name="f",
        dtype=pl.Float64(),
        allow_chunks=False,  # bug: See #24960
    )
)
@pytest.mark.may_fail_auto_streaming  # bug: See #24960
def test_group_by_skew_kurtosis(s: pl.Series) -> None:
    df = s.to_frame()

    exprs: dict[str, Callable[[pl.Expr], pl.Expr]] = {
        "skew": lambda e: e.skew(),
        "skew_b": lambda e: e.skew(bias=False),
        "kurt": lambda e: e.kurtosis(),
        "kurt_f": lambda e: e.kurtosis(fisher=False),
        "kurt_b": lambda e: e.kurtosis(bias=False),
        "kurt_fb": lambda e: e.kurtosis(fisher=False, bias=False),
    }

    sl = df.select([e(pl.col.f).alias(n) for n, e in exprs.items()])
    if s.len() > 0:
        gb = (
            df.group_by(pl.lit(1))
            .agg([e(pl.col.f).alias(n) for n, e in exprs.items()])
            .drop("literal")
        )
        assert_frame_equal(sl, gb)

        # check scalar case
        sl_first = df.select([e(pl.col.f.first()).alias(n) for n, e in exprs.items()])
        gb = (
            df.group_by(pl.lit(1))
            .agg([e(pl.col.f.first()).alias(n) for n, e in exprs.items()])
            .drop("literal")
        )
        assert_frame_equal(sl_first, gb)

    li = df.select(pl.col.f.implode()).select(
        [pl.col.f.list.agg(e(pl.element())).alias(n) for n, e in exprs.items()]
    )
    assert_frame_equal(sl, li)


def test_group_by_rolling_fill_null_25036() -> None:
    frame = pl.DataFrame(
        {
            "date": [date(2013, 1, 1), date(2013, 1, 2), date(2013, 1, 3)] * 2,
            "group": ["A"] * 3 + ["B"] * 3,
            "value": [None, None, 3, 4, None, None],
        }
    )
    result = frame.rolling(index_column="date", period="2d", group_by="group").agg(
        pl.col("value").forward_fill(limit=None).last()
    )

    expected = pl.DataFrame(
        {
            "group": ["A"] * 3 + ["B"] * 3,
            "date": [date(2013, 1, 1), date(2013, 1, 2), date(2013, 1, 3)] * 2,
            "value": [None, None, 3, 4, 4, None],
        }
    )

    assert_frame_equal(result, expected)


exprs = [
    pl.col.a,
    pl.col.a.filter(pl.col.a <= 1),
    pl.col.a.first(),
    pl.lit(1).alias("one"),
    pl.lit(pl.Series([1])),
]


@pytest.mark.parametrize("lhs", exprs)
@pytest.mark.parametrize("rhs", exprs)
@pytest.mark.parametrize("op", [pl.Expr.add, pl.Expr.pow])
def test_group_broadcast_binary_apply_expr_25046(
    lhs: pl.Expr, rhs: pl.Expr, op: Any
) -> None:
    df = pl.DataFrame({"g": [10, 10, 20], "a": [1, 2, 3]})
    groups = pl.lit(1)
    out = df.group_by(groups).agg((op(lhs, rhs)).implode()).to_series(1)
    expected = df.select((op(lhs, rhs)).implode()).to_series()
    assert_series_equal(out, expected)


def test_group_by_explode_none_dtype_25045() -> None:
    df = pl.DataFrame({"a": [None, None, None], "b": [1.0, 2.0, None]})
    out_a = df.group_by(pl.lit(1)).agg(pl.col.a.explode())
    expected_a = pl.DataFrame({"literal": 1, "a": [[None, None, None]]})
    assert_frame_equal(out_a, expected_a)

    out_b = df.group_by(pl.lit(1)).agg(pl.col.b.explode())
    assert len(out_a["a"][0]) == len(out_b["b"][0])

    out_c = df.select(
        pl.coalesce(pl.col.a.explode(), pl.col.b.explode())
        .implode()
        .over(pl.int_range(pl.len()))
    )
    expected_c = pl.DataFrame({"a": [[1.0], [2.0], [None]]})
    assert_frame_equal(out_c, expected_c)


@pytest.mark.parametrize(
    ("expr", "is_scalar"),
    [
        (pl.Expr.forward_fill, False),
        (pl.Expr.backward_fill, False),
        (lambda e: e.forward_fill(1), False),
        (lambda e: e.backward_fill(1), False),
        (lambda e: e.forward_fill(2), False),
        (lambda e: e.backward_fill(2), False),
        (lambda e: e.forward_fill().min(), True),
        (lambda e: e.backward_fill().min(), True),
        (lambda e: e.forward_fill().first(), True),
        (lambda e: e.backward_fill().first(), True),
    ],
)
def test_group_by_forward_backward_fill(
    expr: Callable[[pl.Expr], pl.Expr], is_scalar: bool
) -> None:
    combinations = [
        [1, None, 2, None, None],
        [None, 1, 2, 3, 4],
        [None, None, None, None, None],
        [1, 2, 3, 4, 5],
        [1, None, 2, 3, 4],
        [None, None, None, None, 1],
        [1, None, None, None, None],
        [None, None, None, 1, None],
        [None, 1, None, None, None],
    ]

    cl = cs.starts_with("x")
    df = pl.DataFrame(
        [pl.Series("g", [1] * 5)]
        + [pl.Series(f"x{i}", c, pl.Int64()) for i, c in enumerate(combinations)]
    )

    # verify that we are actually calculating something
    assert len(df.lazy().select(expr(cl)).collect_schema()) == len(combinations)

    data = df.group_by(lit=pl.lit(1)).agg(expr(cl)).drop("lit")
    if not is_scalar:
        data = data.explode(cs.all())
    assert_frame_equal(df.select(expr(cl)), data)

    data = df.group_by("g").agg(expr(cl)).drop("g")
    if not is_scalar:
        data = data.explode(cs.all())
    assert_frame_equal(df.select(expr(cl)), data)

    assert_frame_equal(
        df.select(expr(cl)),
        df.select(cl.implode().list.eval(expr(pl.element())).explode()),
    )

    df = pl.Schema({"x": pl.Int64()}).to_frame()

    data = (
        pl.DataFrame({"x": [None]})
        .group_by(lit=pl.lit(1))
        .agg(expr(pl.lit(pl.Series("x", [], pl.Int64()))))
        .drop("lit")
    )
    if not is_scalar:
        data = data.select(cs.all().reshape((-1,)))
    assert_frame_equal(df.select(expr(cl)), data)

    assert_frame_equal(
        df.select(expr(cl)),
        df.select(cl.implode().list.eval(expr(pl.element())).reshape((-1,))),
    )


@given(s=series())
def test_group_by_drop_nulls(s: pl.Series) -> None:
    df = s.rename("f").to_frame()

    data = (
        df.group_by(lit=pl.lit(1))
        .agg(pl.col.f.drop_nulls())
        .drop("lit")
        .select(pl.col.f.reshape((-1,)))
    )
    assert_frame_equal(df.select(pl.col.f.drop_nulls()), data)

    assert_frame_equal(
        df.select(pl.col.f.drop_nulls()),
        df.select(
            pl.col.f.implode().list.eval(pl.element().drop_nulls()).reshape((-1,))
        ),
    )

    df = pl.Schema({"f": pl.Int64()}).to_frame()

    data = (
        pl.DataFrame({"x": [None]})
        .group_by(lit=pl.lit(1))
        .agg(pl.lit(pl.Series("f", [], pl.Int64())).drop_nulls())
        .drop("lit")
    )
    data = data.select(cs.all().reshape((-1,)))
    assert_frame_equal(df.select(pl.col.f.drop_nulls()), data)

    assert_frame_equal(
        df.select(pl.col.f.drop_nulls()),
        df.select(
            pl.col.f.implode().list.eval(pl.element().drop_nulls()).reshape((-1,))
        ),
    )


@given(s=series())
def test_group_by_drop_nans(s: pl.Series) -> None:
    df = s.rename("f").to_frame()

    data = (
        df.group_by(lit=pl.lit(1))
        .agg(pl.col.f.drop_nans())
        .select(pl.col.f.reshape((-1,)))
    )
    assert_frame_equal(df.select(pl.col.f.drop_nans()), data)

    assert_frame_equal(
        df.select(pl.col.f.drop_nans()),
        df.select(
            pl.col.f.implode().list.eval(pl.element().drop_nans()).reshape((-1,))
        ),
    )

    df = pl.Schema({"f": pl.Int64()}).to_frame()

    data = (
        pl.DataFrame({"x": [None]})
        .group_by(lit=pl.lit(1))
        .agg(pl.lit(pl.Series("f", [], pl.Int64())).drop_nans())
        .drop("lit")
    )
    data = data.select(cs.all().reshape((-1,)))
    assert_frame_equal(df.select(pl.col.f.drop_nans()), data)

    assert_frame_equal(
        df.select(pl.col.f.drop_nans()),
        df.select(
            pl.col.f.implode().list.eval(pl.element().drop_nans()).reshape((-1,))
        ),
    )


@given(
    df=dataframes(
        min_size=1,
        include_cols=[column(name="key", dtype=pl.UInt8, allow_null=False)],
    ),
)
@pytest.mark.parametrize(
    ("expr", "check_order", "returns_scalar", "length_preserving", "is_window"),
    [
        (pl.Expr.unique, False, False, False, False),
        (lambda e: e.unique(maintain_order=True), True, False, False, False),
        (pl.Expr.drop_nans, True, False, False, False),
        (pl.Expr.drop_nulls, True, False, False, False),
        (pl.Expr.null_count, True, False, False, False),
        (pl.Expr.n_unique, True, True, False, False),
        (
            lambda e: e.filter(pl.int_range(0, e.len()) % 3 == 0),
            True,
            False,
            False,
            False,
        ),
        (pl.Expr.shift, True, False, True, False),
        (pl.Expr.forward_fill, True, False, True, False),
        (pl.Expr.backward_fill, True, False, True, False),
        (pl.Expr.reverse, True, False, True, False),
        (
            lambda e: (pl.int_range(e.len() - e.len(), e.len()) % 3 == 0).any(),
            True,
            True,
            False,
            False,
        ),
        (
            lambda e: (pl.int_range(e.len() - e.len(), e.len()) % 3 == 0).all(),
            True,
            True,
            False,
            False,
        ),
        (lambda e: e.head(2), True, False, False, False),
        (pl.Expr.first, True, True, False, False),
        (pl.Expr.mode, False, False, False, False),
        (lambda e: e.fill_null(e.first()).over(e), True, False, True, True),
        (lambda e: e.first().over(e), True, False, True, True),
        (
            lambda e: e.fill_null(e.first()).over(e, mapping_strategy="join"),
            True,
            False,
            True,
            True,
        ),
        (
            lambda e: e.fill_null(e.first()).over(e, mapping_strategy="explode"),
            True,
            False,
            False,
            True,
        ),
        (
            lambda e: e.fill_null(strategy="forward").over([e, e]),
            True,
            False,
            True,
            True,
        ),
        (lambda e: e.fill_null(e.first()).over(e, order_by=e), True, False, True, True),
        (
            lambda e: e.fill_null(e.first()).over(e, order_by=e, descending=True),
            True,
            False,
            True,
            True,
        ),
        (
            lambda e: e.gather(pl.int_range(0, e.len()).slice(1, 3)),
            True,
            False,
            False,
            False,
        ),
    ],
)
def test_grouped_agg_parametric(
    df: pl.DataFrame,
    expr: Callable[[pl.Expr], pl.Expr],
    check_order: bool,
    returns_scalar: bool,
    length_preserving: bool,
    is_window: bool,
) -> None:
    types: dict[str, tuple[Callable[[pl.Expr], pl.Expr], bool, bool]] = {
        "basic": (lambda e: e, False, True),
    }

    if not is_window:
        types["first"] = (pl.Expr.first, True, False)
        types["slice"] = (lambda e: e.slice(1, 3), False, False)
        types["impl_expl"] = (lambda e: e.implode().explode(), False, False)
        types["rolling"] = (
            lambda e: e.rolling(pl.row_index(), period="3i"),
            False,
            True,
        )
        types["over"] = (lambda e: e.forward_fill().over(e), False, True)

    def slit(s: pl.Series) -> pl.Expr:
        import polars._plr as plr

        return pl.Expr._from_pyexpr(plr.lit(s._s, False, is_scalar=True))

    df = df.with_columns(pl.col.key % 4)
    gb = df.group_by("key").agg(
        *[
            expr(t(~cs.by_name("key"))).name.prefix(f"{k}_")
            for k, (t, _, _) in types.items()
        ],
        *[
            expr(slit(df[c].head(1))).alias(f"literal_{c}")
            for c in filter(lambda c: c != "key", df.columns)
        ],
    )
    ls = (
        df.group_by("key")
        .agg(pl.all())
        .select(
            pl.col.key,
            *[
                (~cs.by_name("key"))
                .list.agg(expr(t(pl.element())))
                .name.prefix(f"{k}_")
                for k, (t, _, _) in types.items()
            ],
            *[
                pl.col(c).list.agg(expr(slit(df[c].head(1)))).alias(f"literal_{c}")
                for c in filter(lambda c: c != "key", df.columns)
            ],
        )
    )

    if not is_window:
        types["literal"] = (lambda e: e, True, False)

    def verify_index(i: int) -> None:
        idx_df = df.filter(pl.col.key == pl.lit(i, pl.UInt8))
        idx_gb = gb.filter(pl.col.key == pl.lit(i, pl.UInt8))
        idx_ls = ls.filter(pl.col.key == pl.lit(i, pl.UInt8))

        for col in df.columns:
            if col == "key":
                continue

            for k, (t, t_is_scalar, t_is_length_preserving) in types.items():
                c = f"{k}_{col}"

                if k == "literal":
                    df_s = idx_df.select(
                        expr(t(slit(df[col].head(1)))).alias(c)
                    ).to_series()
                else:
                    df_s = idx_df.select(expr(t(pl.col(col))).alias(c)).to_series()

                gb_s = idx_gb[c]
                ls_s = idx_ls[c]

                result_is_scalar = False
                result_is_scalar |= returns_scalar and t_is_length_preserving
                result_is_scalar |= t_is_scalar and length_preserving
                result_is_scalar &= not is_window

                if not result_is_scalar:
                    gb_s = gb_s.explode(empty_as_null=False)
                    ls_s = ls_s.explode(empty_as_null=False)

                assert_series_equal(df_s, gb_s, check_order=check_order)
                assert_series_equal(df_s, ls_s, check_order=check_order)

    if 0 in df["key"]:
        verify_index(0)
    if 1 in df["key"]:
        verify_index(1)
    if 2 in df["key"]:
        verify_index(2)
    if 3 in df["key"]:
        verify_index(3)


@pytest.mark.parametrize("maintain_order", [False, True])
@pytest.mark.parametrize(
    ("df", "out"),
    [
        (
            pl.DataFrame(
                {
                    "key": [0, 0, 0, 0, 1],
                    "a": [True, False, False, False, False],
                }
            ).with_columns(
                a=pl.when(pl.Series([False, False, False, False, True])).then(pl.col.a)
            ),
            pl.DataFrame(
                {
                    "key": [0, 1],
                    "a": [1, 1],
                },
                schema_overrides={"a": pl.get_index_type()},
            ),
        ),
        (
            pl.DataFrame(
                {
                    "key": [0, 0, 1, 1],
                    "a": [False, False, False, False],
                }
            ).with_columns(
                a=pl.when(pl.Series([False, False, True, True])).then(pl.col.a)
            ),
            pl.DataFrame(
                {
                    "key": [0, 1],
                    "a": [1, 1],
                },
                schema_overrides={"a": pl.get_index_type()},
            ),
        ),
    ],
)
def test_n_unique_masked_bools(
    maintain_order: bool, df: pl.DataFrame, out: pl.DataFrame
) -> None:
    df = df

    assert_frame_equal(
        df.group_by("key", maintain_order=maintain_order).agg(pl.col.a.n_unique()),
        out,
        check_row_order=maintain_order,
    )
    assert_frame_equal(
        df.group_by("key", maintain_order=maintain_order)
        .agg(pl.col.a)
        .with_columns(pl.col.a.list.agg(pl.element().n_unique())),
        out,
        check_row_order=maintain_order,
    )


@pytest.mark.parametrize("maintain_order", [False, True])
@pytest.mark.parametrize("stable", [False, True])
def test_group_bool_unique_25267(maintain_order: bool, stable: bool) -> None:
    df = pl.DataFrame(
        {
            "id": ["A", "A", "B", "B", "C", "C"],
            "str_values": ["D", "E", "F", "F", "G", "G"],
            "bool_values": [True, False, True, True, False, False],
        }
    )

    gb = df.group_by("id", maintain_order=maintain_order).agg(
        pl.col("str_values", "bool_values").unique(maintain_order=stable),
    )

    ls = (
        df.group_by("id", maintain_order=maintain_order)
        .agg("str_values", "bool_values")
        .with_columns(
            pl.col("str_values", "bool_values").list.agg(
                pl.element().unique(maintain_order=stable)
            )
        )
    )

    for i in ["A", "B", "C"]:
        for c in ["str_values", "bool_values"]:
            df_s = (
                df.select(pl.col(c).filter(pl.col.id == pl.lit(i)))
                .to_series()
                .unique(maintain_order=stable)
            )
            gb_s = gb.select(
                pl.col(c).filter(pl.col.id == pl.lit(i)).reshape((-1,))
            ).to_series()
            ls_s = ls.select(
                pl.col(c).filter(pl.col.id == pl.lit(i)).reshape((-1,))
            ).to_series()

            assert_series_equal(df_s, gb_s, check_order=stable)
            assert_series_equal(df_s, ls_s, check_order=stable)


@pytest.mark.parametrize("group_as_slice", [False, True])
@pytest.mark.parametrize("n", [10, 100, 519])
@pytest.mark.parametrize(
    "dtype", [pl.Int32, pl.Boolean, pl.String, pl.Categorical, pl.List(pl.Int32)]
)
def test_group_by_first_last(
    group_as_slice: bool, n: int, dtype: PolarsDataType
) -> None:
    group_by_first_last_test_impl(group_as_slice, n, dtype)


@pytest.mark.slow
@pytest.mark.parametrize("group_as_slice", [False, True])
@pytest.mark.parametrize("n", [1056, 10_432])
@pytest.mark.parametrize(
    "dtype", [pl.Int32, pl.Boolean, pl.String, pl.Categorical, pl.List(pl.Int32)]
)
def test_group_by_first_last_big(
    group_as_slice: bool, n: int, dtype: PolarsDataType
) -> None:
    group_by_first_last_test_impl(group_as_slice, n, dtype)


def group_by_first_last_test_impl(
    group_as_slice: bool, n: int, dtype: PolarsDataType
) -> None:
    idx = pl.Series([1, 2, 3, 4, 5], dtype=pl.Int32)

    lf = pl.LazyFrame(
        {
            "idx": pl.Series(
                [1] * n + [2] * n + [3] * n + [4] * n + [5] * n, dtype=pl.Int32
            ),
            # Each successive group has an additional None spanning the elements
            "a": pl.Series(
                [
                    *[None] * 0, *list(range(1, n + 1)), *[None] * 0,  # idx = 1
                    *[None] * 1, *list(range(2, n - 0)), *[None] * 1,  # idx = 2
                    *[None] * 2, *list(range(3, n - 1)), *[None] * 2,  # idx = 3
                    *[None] * 3, *list(range(4, n - 2)), *[None] * 3,  # idx = 4
                    *[None] * 4, *list(range(5, n - 3)), *[None] * 4,  # idx = 5
                ],
                dtype=pl.Int32,
            ),
        }
    )  # fmt: skip
    if group_as_slice:
        lf = lf.set_sorted("idx")  # Use GroupSlice path

    if dtype == pl.Categorical:
        # for Categorical, we must first go through String
        lf = lf.with_columns(pl.col("a").cast(pl.String))
    lf = lf.with_columns(pl.col("a").cast(dtype))

    # first()
    result = lf.group_by("idx", maintain_order=True).agg(pl.col("a").first()).collect()
    expected_vals = pl.Series([1, None, None, None, None])
    if dtype == pl.Categorical:
        # for Categorical, we must first go through String
        expected_vals = expected_vals.cast(pl.String)

    expected_vals = expected_vals.cast(dtype)
    expected = pl.DataFrame({"idx": idx, "a": expected_vals})
    assert_frame_equal(result, expected)
    result = lf.group_by("idx", maintain_order=True).first().collect()
    assert_frame_equal(result, expected)

    # first(ignore_nulls=True)
    result = (
        lf.group_by("idx", maintain_order=True)
        .agg(pl.col("a").first(ignore_nulls=True))
        .collect()
    )
    expected_vals = pl.Series([1, 2, 3, 4, 5])
    if dtype == pl.Categorical:
        # for Categorical, we must first go through String
        expected_vals = expected_vals.cast(pl.String)

    expected_vals = expected_vals.cast(dtype)
    expected = pl.DataFrame({"idx": idx, "a": expected_vals})
    assert_frame_equal(result, expected)
    result = lf.group_by("idx", maintain_order=True).first(ignore_nulls=True).collect()
    assert_frame_equal(result, expected)

    # last()
    result = lf.group_by("idx", maintain_order=True).agg(pl.col("a").last()).collect()
    expected_vals = pl.Series([n, None, None, None, None])
    if dtype == pl.Categorical:
        # for Categorical, we must first go through String
        expected_vals = expected_vals.cast(pl.String)

    expected_vals = expected_vals.cast(dtype)
    expected = pl.DataFrame({"idx": idx, "a": expected_vals})
    assert_frame_equal(result, expected)
    result = lf.group_by("idx", maintain_order=True).last().collect()
    assert_frame_equal(result, expected)

    # last_non_null
    result = (
        lf.group_by("idx", maintain_order=True)
        .agg(pl.col("a").last(ignore_nulls=True))
        .collect()
    )
    expected_vals = pl.Series([n, n - 1, n - 2, n - 3, n - 4])
    if dtype == pl.Categorical:
        # for Categorical, we must first go through String
        expected_vals = expected_vals.cast(pl.String)

    expected_vals = expected_vals.cast(dtype)
    expected = pl.DataFrame({"idx": idx, "a": expected_vals})
    assert_frame_equal(result, expected)
    result = lf.group_by("idx", maintain_order=True).last(ignore_nulls=True).collect()
    assert_frame_equal(result, expected)

    # Test with no nulls
    lf = pl.LazyFrame(
        {
            "idx": pl.Series(
                [1] * n + [2] * n + [3] * n + [4] * n + [5] * n, dtype=pl.Int32
            ),
            # Each successive group has an additional None spanning the elements
            "a": pl.Series(
                [
                    *list(range(1, n + 1)),  # idx = 1
                    *list(range(2, n + 2)),  # idx = 2
                    *list(range(3, n + 3)),  # idx = 3
                    *list(range(4, n + 4)),  # idx = 4
                    *list(range(5, n + 5)),  # idx = 5
                ],
                dtype=pl.Int32,
            ),
        }
    )
    if group_as_slice:
        lf = lf.set_sorted("idx")  # Use GroupSlice path

    if dtype == pl.Categorical:
        # for Categorical, we must first go through String
        lf = lf.with_columns(pl.col("a").cast(pl.String))
    lf = lf.with_columns(pl.col("a").cast(dtype))

    # first()
    expected_vals = pl.Series([1, 2, 3, 4, 5])
    if dtype == pl.Categorical:
        # for Categorical, we must first go through String
        expected_vals = expected_vals.cast(pl.String)

    expected_vals = expected_vals.cast(dtype)
    expected = pl.DataFrame({"idx": idx, "a": expected_vals})
    result = lf.group_by("idx", maintain_order=True).agg(pl.col("a").first()).collect()
    assert_frame_equal(result, expected)
    result = lf.group_by("idx", maintain_order=True).first().collect()
    assert_frame_equal(result, expected)

    # first_non_null
    result = (
        lf.group_by("idx", maintain_order=True)
        .agg(pl.col("a").first(ignore_nulls=True))
        .collect()
    )
    assert_frame_equal(result, expected)
    result = lf.group_by("idx", maintain_order=True).first(ignore_nulls=True).collect()
    assert_frame_equal(result, expected)

    # last()
    expected_vals = pl.Series([n, n + 1, n + 2, n + 3, n + 4])
    if dtype == pl.Categorical:
        # for Categorical, we must first go through String
        expected_vals = expected_vals.cast(pl.String)

    expected_vals = expected_vals.cast(dtype)
    expected = pl.DataFrame({"idx": idx, "a": expected_vals})
    result = lf.group_by("idx", maintain_order=True).agg(pl.col("a").last()).collect()
    assert_frame_equal(result, expected)
    result = lf.group_by("idx", maintain_order=True).last().collect()
    assert_frame_equal(result, expected)

    # last_non_null
    result = (
        lf.group_by("idx", maintain_order=True)
        .agg(pl.col("a").last(ignore_nulls=True))
        .collect()
    )
    assert_frame_equal(result, expected)
    result = lf.group_by("idx", maintain_order=True).last(ignore_nulls=True).collect()
    assert_frame_equal(result, expected)


def test_sorted_group_by() -> None:
    lf = pl.LazyFrame(
        {
            "a": [1, 1, 2, 2, 3, 3, 3],
            "b": [4, 5, 8, 1, 0, 1, 3],
        }
    )

    lf1 = lf
    lf2 = lf.set_sorted("a")

    assert_frame_equal(
        *[
            q.group_by("a")
            .agg(b_first=pl.col.b.first(), b_sum=pl.col.b.sum(), b=pl.col.b)
            .collect(engine="streaming")
            for q in (lf1, lf2)
        ],
        check_row_order=False,
    )

    lf = lf.with_columns(c=pl.col.a.rle_id())
    lf1 = lf
    lf2 = lf.set_sorted("a", "c")

    assert_frame_equal(
        *[
            q.group_by("a", "c")
            .agg(b_first=pl.col.b.first(), b_sum=pl.col.b.sum(), b=pl.col.b)
            .collect(engine="streaming")
            for q in (lf1, lf2)
        ],
        check_row_order=False,
    )


def test_sorted_group_by_slice() -> None:
    lf = (
        pl.DataFrame({"a": [0, 5, 2, 1, 3] * 50})
        .with_row_index()
        .with_columns(pl.col.index // 5)
        .lazy()
        .set_sorted("index")
        .group_by("index", maintain_order=True)
        .agg(pl.col.a.sum() + pl.col.index.first())
    )

    expected = pl.DataFrame(
        [
            pl.Series("index", range(50), pl.get_index_type()),
            pl.Series("a", range(11, 11 + 50), pl.Int64),
        ]
    )

    assert_frame_equal(lf.head(2).collect(), expected.head(2))
    assert_frame_equal(lf.slice(1, 3).collect(), expected.slice(1, 3))
    assert_frame_equal(lf.tail(2).collect(), expected.tail(2))
    assert_frame_equal(lf.slice(5, 1).collect(), expected.slice(5, 1))
    assert_frame_equal(lf.slice(5, 0).collect(), expected.slice(5, 0))
    assert_frame_equal(lf.slice(2, 1).collect(), expected.slice(2, 1))
    assert_frame_equal(lf.slice(50, 1).collect(), expected.slice(50, 1))
    assert_frame_equal(lf.slice(20, 30).collect(), expected.slice(20, 30))
    assert_frame_equal(lf.slice(20, 30).collect(), expected.slice(20, 30))


def test_agg_first_last_non_null_25405() -> None:
    lf = pl.LazyFrame(
        {
            "a": [1, 1, 1, 1, 2, 2, 2, 2, 2],
            "b": pl.Series([1, 2, 3, None, None, 4, 5, 6, None]),
        }
    )

    # first
    result = lf.group_by("a", maintain_order=True).agg(
        pl.col("b").first(ignore_nulls=True)
    )
    expected = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [1, 4],
        }
    )
    assert_frame_equal(result.collect(), expected)

    result = lf.with_columns(pl.col("b").first(ignore_nulls=True).over("a"))
    expected = pl.DataFrame(
        {
            "a": [1, 1, 1, 1, 2, 2, 2, 2, 2],
            "b": [1, 1, 1, 1, 4, 4, 4, 4, 4],
        }
    )
    assert_frame_equal(result.collect(), expected)

    # last
    result = lf.group_by("a", maintain_order=True).agg(
        pl.col("b").last(ignore_nulls=True)
    )
    expected = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [3, 6],
        }
    )
    assert_frame_equal(result.collect(), expected)

    result = lf.with_columns(pl.col("b").last(ignore_nulls=True).over("a"))
    expected = pl.DataFrame(
        {
            "a": [1, 1, 1, 1, 2, 2, 2, 2, 2],
            "b": [3, 3, 3, 3, 6, 6, 6, 6, 6],
        }
    )
    assert_frame_equal(result.collect(), expected)


def test_group_by_sum_on_strings_should_error_24659() -> None:
    with pytest.raises(
        InvalidOperationError,
        match=r"`sum`.*operation not supported for dtype.*str",
    ):
        pl.DataFrame({"str": ["a", "b"]}).group_by(1).agg(pl.col.str.sum())


@pytest.mark.parametrize("tail", [0, 1, 4, 5, 6, 10])
def test_unique_head_tail_26429(tail: int) -> None:
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
        }
    )
    out = df.lazy().unique().tail(tail).collect()
    expected = min(tail, df.height)
    assert len(out) == expected


def test_group_by_cse_alias_26423() -> None:
    df = pl.LazyFrame({"a": [1, 2, 1, 2, 3, 4]})
    result = df.group_by("a").agg(pl.len(), pl.len().alias("len_a")).collect()
    expected = pl.DataFrame(
        {"a": [1, 2, 3, 4], "len": [2, 2, 1, 1], "len_a": [2, 2, 1, 1]},
        schema={
            "a": pl.Int64,
            "len": pl.get_index_type(),
            "len_a": pl.get_index_type(),
        },
    )
    assert_frame_equal(result, expected, check_row_order=False)
