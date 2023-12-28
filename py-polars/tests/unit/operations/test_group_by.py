from __future__ import annotations

from collections import OrderedDict
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from polars.type_aliases import PolarsDataType


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
    df.group_by("b").agg(pl.col("c").forward_fill()).explode("c")

    # get a specific column
    result = df.group_by("b", maintain_order=True).agg(pl.count("a"))
    assert result.rows() == [("a", 2), ("b", 3)]
    assert result.columns == ["b", "a"]


@pytest.fixture()
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
        ("count", [("a", 2), ("b", 3)]),
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
            [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4)],
            [datetime(2023, 1, 2), datetime(2023, 1, 4)],
            pl.Date,
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
        ([time(1), time(2), time(3), time(4)], [time(2), time(4)], pl.Time, pl.Time),
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
    df = pl.DataFrame(
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
    assert_frame_equal(result, df_expected)


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
        ([time(1), time(2), time(4), time(5)], [time(2), time(5)], pl.Time, pl.Time),
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
    df = pl.DataFrame(
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
    assert_frame_equal(result, df_expected)


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
    for i, (group, data) in enumerate(df.group_by("foo", maintain_order=True)):
        assert group == expected_names[i]
        assert data.rows() == expected_rows[i]

    # Grouped by ALL columns should give groups of a single row
    result = list(df.group_by(["foo", "bar", "baz"]))
    assert len(result) == 6

    # Iterating over groups should also work when grouping by expressions
    result2 = list(df.group_by(["foo", pl.col("bar") * pl.col("baz")]))
    assert len(result2) == 5

    # Single column, alias in group_by
    df = pl.DataFrame({"foo": [1, 2, 3, 4, 5, 6]})
    gb = df.group_by((pl.col("foo") // 2).alias("bar"), maintain_order=True)
    result3 = [(group, df.rows()) for group, df in gb]
    expected3 = [(0, [(1,)]), (1, [(2,), (3,)]), (2, [(4,), (5,)]), (3, [(6,)])]
    assert result3 == expected3


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
        .collect()
    )
    assert df.rows() == []
    assert df.shape == (0, 2)
    assert df.schema == {"key": pl.Categorical, "val": pl.Float64}


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
        [
            pl.col("col1")
            .sort_by(pl.col("col2_arg_sort"))
            .over("group")
            .alias("result_a"),
            pl.col("col1")
            .sort_by(pl.col("col2").arg_sort())
            .over("group")
            .alias("result_b"),
        ]
    )

    assert_series_equal(out["result_a"], out["result_b"], check_names=False)
    assert out["result_a"].to_list() == [1, 2, 3, 3, 2, 1, 2, 3, 1]


def test_unique_order() -> None:
    df = pl.DataFrame({"a": [1, 2, 1]}).with_row_count()
    assert df.unique(keep="last", subset="a", maintain_order=True).to_dict(
        as_series=False
    ) == {
        "row_nr": [1, 2],
        "a": [2, 1],
    }
    assert df.unique(keep="first", subset="a", maintain_order=True).to_dict(
        as_series=False
    ) == {
        "row_nr": [0, 1],
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
        "a": [0, 1, 2],
        "sum_ratio_1": [1.0, 4.2, 5.0],
        "last_ratio_1": [1.0, 6.0, 6.0],
        "last_ratio_2": [1.0, 6.0, 6.0],
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

    assert res["corr"] == pytest.approx([9.148920923684765])
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


@pytest.mark.slow()
@pytest.mark.parametrize("dtype", [pl.Int32, pl.UInt32])
def test_overflow_mean_partitioned_group_by_5194(dtype: pl.PolarsDataType) -> None:
    df = pl.DataFrame(
        [
            pl.Series("data", [10_00_00_00] * 100_000, dtype=dtype),
            pl.Series("group", [1, 2] * 50_000, dtype=dtype),
        ]
    )
    result = df.group_by("group").agg(pl.col("data").mean()).sort(by="group")
    expected = {"group": [1, 2], "data": [10000000.0, 10000000.0]}
    assert result.to_dict(as_series=False) == expected


def test_group_by_multiple_column_reference() -> None:
    # Issue #7181
    df = pl.DataFrame(
        {
            "gr": ["a", "b", "a", "b", "a", "b"],
            "val": [1, 20, 100, 2000, 10000, 200000],
        }
    )
    res = df.group_by("gr").agg(
        pl.col("val") + pl.col("val").shift().fill_null(0),
    )

    assert res.sort("gr").to_dict(as_series=False) == {
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
        ("n_unique", [], [1, None], pl.UInt32),
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


def test_perfect_hash_table_null_values_8663() -> None:
    s = pl.Series(
        "a",
        [
            "3",
            "41",
            "17",
            "5",
            "26",
            "27",
            "43",
            "45",
            "41",
            "13",
            "45",
            "48",
            "17",
            "22",
            "31",
            "25",
            "28",
            "13",
            "7",
            "26",
            "17",
            "4",
            "43",
            "47",
            "30",
            "28",
            "8",
            "27",
            "6",
            "7",
            "26",
            "11",
            "37",
            "29",
            "49",
            "20",
            "29",
            "28",
            "23",
            "9",
            None,
            "38",
            "19",
            "7",
            "38",
            "3",
            "30",
            "37",
            "41",
            "5",
            "16",
            "26",
            "31",
            "6",
            "25",
            "11",
            "17",
            "31",
            "31",
            "20",
            "26",
            None,
            "39",
            "10",
            "38",
            "4",
            "39",
            "15",
            "13",
            "35",
            "38",
            "11",
            "39",
            "11",
            "48",
            "36",
            "18",
            "11",
            "34",
            "16",
            "28",
            "9",
            "37",
            "8",
            "17",
            "48",
            "44",
            "28",
            "25",
            "30",
            "37",
            "30",
            "18",
            "12",
            None,
            "27",
            "10",
            "3",
            "16",
            "27",
            "6",
        ],
        dtype=pl.Categorical,
    )
    result = s.to_frame("a").group_by("a").agg(pl.col("a").alias("agg"))
    expected = {
        "a": [
            "3",
            "41",
            "17",
            "5",
            "26",
            "27",
            "43",
            "45",
            "13",
            "48",
            "22",
            "31",
            "25",
            "28",
            "7",
            "4",
            "47",
            "30",
            "8",
            "6",
            "11",
            "37",
            "29",
            "49",
            "20",
            "23",
            "9",
            "38",
            "19",
            "16",
            "39",
            "10",
            "15",
            "35",
            "36",
            "18",
            "34",
            "44",
            "12",
            None,
        ],
        "agg": [
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
            [None, None, None],
        ],
    }
    assert result.to_dict(as_series=False) == expected


def test_group_by_partitioned_ending_cast(monkeypatch: Any) -> None:
    monkeypatch.setenv("POLARS_FORCE_PARTITION", "1")
    df = pl.DataFrame({"a": [1] * 5, "b": [1] * 5})
    out = df.group_by(["a", "b"]).agg(pl.count().cast(pl.Int64).alias("num"))
    expected = pl.DataFrame({"a": [1], "b": [1], "num": [5]})
    assert_frame_equal(out, expected)


def test_groupby_deprecated() -> None:
    df = pl.DataFrame({"a": [1, 1, 2], "b": [3, 4, 5]})

    with pytest.deprecated_call():
        result = df.groupby("a").agg(pl.sum("b"))
    with pytest.deprecated_call():
        result_lazy = df.lazy().groupby("a").agg(pl.sum("b")).collect()

    expected = df.group_by("a").agg(pl.sum("b"))
    assert_frame_equal(result, expected, check_row_order=False)
    assert_frame_equal(result_lazy, expected, check_row_order=False)


def test_groupby_rolling_deprecated() -> None:
    df = pl.DataFrame(
        {
            "date": pl.datetime_range(
                datetime(2020, 1, 1), datetime(2020, 1, 5), eager=True
            ),
            "value": [1, 2, 3, 4, 5],
        }
    )

    with pytest.deprecated_call():
        result = df.groupby_rolling("date", period="2d").agg(pl.sum("value"))
    with pytest.deprecated_call():
        result_lazy = (
            df.lazy()
            .groupby_rolling("date", period="2d")
            .agg(pl.sum("value"))
            .collect()
        )

    expected = df.rolling("date", period="2d").agg(pl.sum("value"))
    assert_frame_equal(result, expected, check_row_order=False)
    assert_frame_equal(result_lazy, expected, check_row_order=False)


def test_group_by_rolling_deprecated() -> None:
    df = pl.DataFrame(
        {
            "date": pl.datetime_range(
                datetime(2020, 1, 1), datetime(2020, 1, 5), eager=True
            ),
            "value": [1, 2, 3, 4, 5],
        }
    )

    with pytest.deprecated_call():
        result = df.group_by_rolling("date", period="2d").agg(pl.sum("value"))
    with pytest.deprecated_call():
        result_lazy = (
            df.lazy()
            .groupby_rolling("date", period="2d")
            .agg(pl.sum("value"))
            .collect()
        )

    expected = df.rolling("date", period="2d").agg(pl.sum("value"))
    assert_frame_equal(result, expected, check_row_order=False)
    assert_frame_equal(result_lazy, expected, check_row_order=False)


def test_group_by_multiple_keys_one_literal() -> None:
    df = pl.DataFrame({"a": [1, 1, 2], "b": [4, 5, 6]})

    expected = {"a": [1, 2], "literal": [1, 1], "b": [5, 6]}
    for streaming in [True, False]:
        assert (
            df.lazy()
            .group_by("a", pl.lit(1))
            .agg(pl.col("b").max())
            .sort(["a", "b"])
            .collect(streaming=streaming)
            .to_dict(as_series=False)
            == expected
        )


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
    a = lgb.count()
    b = lgb.count()
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
        "x": [[2.0, 4.0], [8.0, 16.0, 32.0]],
    }
