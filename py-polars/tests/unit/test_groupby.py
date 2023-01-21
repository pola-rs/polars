from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_groupby() -> None:
    df = pl.DataFrame(
        {
            "a": ["a", "b", "a", "b", "b", "c"],
            "b": [1, 2, 3, 4, 5, 6],
            "c": [6, 5, 4, 3, 2, 1],
        }
    )

    assert df.groupby("a").apply(lambda df: df[["c"]].sum()).sort("c")["c"][0] == 1

    # Use lazy API in eager groupby
    assert sorted(df.groupby("a").agg([pl.sum("b")]).rows()) == [
        ("a", 4),
        ("b", 11),
        ("c", 6),
    ]
    # test if it accepts a single expression
    assert df.groupby("a", maintain_order=True).agg(pl.sum("b")).rows() == [
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
    df.groupby("b").agg(pl.col("c").forward_fill()).explode("c")

    # get a specific column
    result = df.groupby("b", maintain_order=True).agg(pl.count("a"))
    assert result.rows() == [("a", 2), ("b", 3)]
    assert result.columns == ["b", "a"]

    # make sure all the methods below run
    assert sorted(df.groupby("b").first().rows()) == [("a", 1, None), ("b", 3, None)]
    assert sorted(df.groupby("b").last().rows()) == [("a", 2, 1), ("b", 5, None)]
    assert sorted(df.groupby("b").max().rows()) == [("a", 2, 1), ("b", 5, 1)]
    assert sorted(df.groupby("b").min().rows()) == [("a", 1, 1), ("b", 3, 1)]
    assert sorted(df.groupby("b").count().rows()) == [("a", 2), ("b", 3)]
    assert sorted(df.groupby("b").mean().rows()) == [("a", 1.5, 1.0), ("b", 4.0, 1.0)]
    assert sorted(df.groupby("b").n_unique().rows()) == [("a", 2, 2), ("b", 3, 2)]
    assert sorted(df.groupby("b").median().rows()) == [("a", 1.5, 1.0), ("b", 4.0, 1.0)]
    assert sorted(df.groupby("b").agg_list().rows()) == [
        ("a", [1, 2], [None, 1]),
        ("b", [3, 4, 5], [None, 1, None]),
    ]
    # assert sorted(df.groupby("b").quantile(0.5).rows()) == ...

    # Invalid input: `by` not specified as a sequence
    with pytest.raises(TypeError):
        df.groupby("a", "b")  # type: ignore[arg-type]


def test_groupby_iteration() -> None:
    df = pl.DataFrame(
        {
            "foo": ["a", "b", "a", "b", "b", "c"],
            "bar": [1, 2, 3, 4, 5, 6],
            "baz": [6, 5, 4, 3, 2, 1],
        }
    )
    expected_shapes = [(2, 3), (3, 3), (1, 3)]
    expected_rows = [
        [("a", 1, 6), ("a", 3, 4)],
        [("b", 2, 5), ("b", 4, 3), ("b", 5, 2)],
        [("c", 6, 1)],
    ]
    for i, group in enumerate(df.groupby("foo", maintain_order=True)):
        assert group.shape == expected_shapes[i]
        assert group.rows() == expected_rows[i]

    # Grouped by ALL columns should give groups of a single row
    result = list(df.groupby(["foo", "bar", "baz"]))
    assert len(result) == 6

    # Iterating over groups should also work when grouping by expressions
    result = list(df.groupby(["foo", pl.col("bar") * pl.col("baz")]))
    assert len(result) == 5


def bad_agg_parameters() -> list[Any]:
    return [[("b", "sum")], [("b", ["sum"])], {"b": "sum"}, {"b": ["sum"]}]


def good_agg_parameters() -> list[pl.Expr | list[pl.Expr]]:
    return [
        [pl.col("b").sum()],
        pl.col("b").sum(),
    ]


@pytest.mark.parametrize("lazy", [True, False])
def test_groupby_agg_input_types(lazy: bool) -> None:
    df = pl.DataFrame({"a": [1, 1, 2, 2], "b": [1, 2, 3, 4]})
    df_or_lazy: pl.DataFrame | pl.LazyFrame = df.lazy() if lazy else df

    for bad_param in bad_agg_parameters():
        with pytest.raises(TypeError):  # noqa: PT012
            result = df_or_lazy.groupby("a").agg(bad_param)
            if lazy:
                result.collect()  # type: ignore[union-attr]

    expected = pl.DataFrame({"a": [1, 2], "b": [3, 7]})

    for good_param in good_agg_parameters():
        result = df_or_lazy.groupby("a", maintain_order=True).agg(good_param)
        if lazy:
            result = result.collect()  # type: ignore[union-attr]
        assert_frame_equal(result, expected)


@pytest.mark.parametrize("lazy", [True, False])
def test_groupby_rolling_agg_input_types(lazy: bool) -> None:
    df = pl.DataFrame({"index_column": [0, 1, 2, 3], "b": [1, 3, 1, 2]})
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


@pytest.mark.parametrize("lazy", [True, False])
def test_groupby_dynamic_agg_input_types(lazy: bool) -> None:
    df = pl.DataFrame({"index_column": [0, 1, 2, 3], "b": [1, 3, 1, 2]})
    df_or_lazy: pl.DataFrame | pl.LazyFrame = df.lazy() if lazy else df

    for bad_param in bad_agg_parameters():
        with pytest.raises(TypeError):  # noqa: PT012
            result = df_or_lazy.groupby_dynamic(
                index_column="index_column", every="2i", closed="right"
            ).agg(bad_param)
            if lazy:
                result.collect()  # type: ignore[union-attr]

    expected = pl.DataFrame({"index_column": [-2, 0, 2], "b": [1, 4, 2]})

    for good_param in good_agg_parameters():
        result = df_or_lazy.groupby_dynamic(
            index_column="index_column", every="2i", closed="right"
        ).agg(good_param)
        if lazy:
            result = result.collect()  # type: ignore[union-attr]
        assert_frame_equal(result, expected)


def test_groupby_sorted_empty_dataframe_3680() -> None:
    df = (
        pl.DataFrame(
            [
                pl.Series("key", [], dtype=pl.Categorical),
                pl.Series("val", [], dtype=pl.Float64),
            ]
        )
        .lazy()
        .sort("key")
        .groupby("key")
        .tail(1)
        .collect()
    )
    assert df.rows() == []
    assert df.shape == (0, 2)
    assert df.schema == {"key": pl.Categorical, "val": pl.Float64}


def test_groupby_custom_agg_empty_list() -> None:
    assert (
        pl.DataFrame(
            [
                pl.Series("key", [], dtype=pl.Categorical),
                pl.Series("val", [], dtype=pl.Float64),
            ]
        )
        .groupby("key")
        .agg(
            [
                pl.col("val").mean().alias("mean"),
                pl.col("val").std().alias("std"),
                pl.col("val").skew().alias("skew"),
                pl.col("val").kurtosis().alias("kurt"),
            ]
        )
    ).dtypes == [pl.Categorical, pl.Float64, pl.Float64, pl.Float64, pl.Float64]


def test_apply_after_take_in_groupby_3869() -> None:
    assert (
        pl.DataFrame(
            {
                "k": list("aaabbb"),
                "t": [1, 2, 3, 4, 5, 6],
                "v": [3, 1, 2, 5, 6, 4],
            }
        )
        .groupby("k", maintain_order=True)
        .agg(
            pl.col("v").take(pl.col("t").arg_max()).sqrt()
        )  # <- fails for sqrt, exp, log, pow, etc.
    ).to_dict(False) == {"k": ["a", "b"], "v": [1.4142135623730951, 2.0]}


def test_groupby_rolling_negative_offset_3914() -> None:
    df = pl.DataFrame(
        {
            "datetime": pl.date_range(datetime(2020, 1, 1), datetime(2020, 1, 5), "1d"),
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

    assert df.groupby_rolling(index_column="ints", period="2i", offset="-5i",).agg(
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


def test_groupby_signed_transmutes() -> None:
    df = pl.DataFrame({"foo": [-1, -2, -3, -4, -5], "bar": [500, 600, 700, 800, 900]})

    for dt in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
        df = (
            df.with_columns([pl.col("foo").cast(dt), pl.col("bar")])
            .groupby("foo", maintain_order=True)
            .agg(pl.col("bar").median())
        )

        assert df.to_dict(False) == {
            "foo": [-1, -2, -3, -4, -5],
            "bar": [500.0, 600.0, 700.0, 800.0, 900.0],
        }


def test_argsort_sort_by_groups_update__4360() -> None:
    df = pl.DataFrame(
        {
            "group": ["a"] * 3 + ["b"] * 3 + ["c"] * 3,
            "col1": [1, 2, 3] * 3,
            "col2": [1, 2, 3, 3, 2, 1, 2, 3, 1],
        }
    )

    out = df.with_column(
        pl.col("col2").arg_sort().over("group").alias("col2_argsort")
    ).with_columns(
        [
            pl.col("col1")
            .sort_by(pl.col("col2_argsort"))
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
    assert df.unique(keep="last", subset="a", maintain_order=True).to_dict(False) == {
        "row_nr": [1, 2],
        "a": [2, 1],
    }
    assert df.unique(keep="first", subset="a", maintain_order=True).to_dict(False) == {
        "row_nr": [0, 1],
        "a": [1, 2],
    }


def test_groupby_dynamic_flat_agg_4814() -> None:
    df = pl.DataFrame({"a": [1, 2, 2], "b": [1, 8, 12]})

    assert df.groupby_dynamic("a", every="1i", period="2i").agg(
        [
            (pl.col("b").sum() / pl.col("a").sum()).alias("sum_ratio_1"),
            (pl.col("b").last() / pl.col("a").last()).alias("last_ratio_1"),
            (pl.col("b") / pl.col("a")).last().alias("last_ratio_2"),
        ]
    ).to_dict(False) == {
        "a": [1, 2],
        "sum_ratio_1": [4.2, 5.0],
        "last_ratio_1": [6.0, 6.0],
        "last_ratio_2": [6.0, 6.0],
    }


def test_groupby_dynamic_overlapping_groups_flat_apply_multiple_5038() -> None:
    every: str | timedelta
    period: str | timedelta

    for every, period in (  # type: ignore[assignment]
        ("10s", timedelta(seconds=100)),
        (timedelta(seconds=10), "100s"),
    ):
        assert (
            pl.DataFrame(
                {
                    "a": [
                        datetime(2021, 1, 1) + timedelta(seconds=2**i)
                        for i in range(10)
                    ],
                    "b": [float(i) for i in range(10)],
                }
            )
            .lazy()
            .groupby_dynamic("a", every=every, period=period)
            .agg([pl.col("b").var().sqrt().alias("corr")])
        ).collect().sum().to_dict(False) == pytest.approx(
            {"a": [None], "corr": [6.988674024215477]}
        )


def test_take_in_groupby() -> None:
    df = pl.DataFrame({"group": [1, 1, 1, 2, 2, 2], "values": [10, 200, 3, 40, 500, 6]})
    assert df.groupby("group").agg(
        pl.col("values").take(1) - pl.col("values").take(2)
    ).sort("group").to_dict(False) == {"group": [1, 2], "values": [197, 494]}


def test_groupby_wildcard() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [1, 2],
        }
    )
    assert df.groupby([pl.col("*")], maintain_order=True).agg(
        [pl.col("a").first().suffix("_agg")]
    ).to_dict(False) == {"a": [1, 2], "b": [1, 2], "a_agg": [1, 2]}


def test_groupby_all_masked_out() -> None:
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
    assert parts[0].frame_equal(df)


def test_groupby_min_max_string_type() -> None:
    table = pl.from_dict({"a": [1, 1, 2, 2, 2], "b": ["a", "b", "c", "d", None]})

    expected = {"a": [1, 2], "min": ["a", "c"], "max": ["b", "d"]}

    for streaming in [True, False]:
        assert (
            table.lazy()
            .groupby("a")
            .agg([pl.min("b").alias("min"), pl.max("b").alias("max")])
            .collect(streaming=streaming)
            .sort("a")
            .to_dict(False)
            == expected
        )


def test_groupby_null_propagation_6185() -> None:
    df_1 = pl.DataFrame({"A": [0, 0], "B": [1, 2]})

    expr = pl.col("A").filter(pl.col("A") > 0)

    expected = {"B": [1, 2], "A": [None, None]}
    assert (
        df_1.groupby("B").agg((expr - expr.mean()).mean()).sort("B").to_dict(False)
        == expected
    )


def test_groupby_when_then_with_binary_and_agg_in_pred_6202() -> None:
    df = pl.DataFrame(
        {"code": ["a", "b", "b", "b", "a"], "xx": [1.0, -1.5, -0.2, -3.9, 3.0]}
    )
    assert (
        df.groupby("code", maintain_order=True).agg(
            [pl.when(pl.col("xx") > pl.min("xx")).then(True).otherwise(False)]
        )
    ).to_dict(False) == {
        "code": ["a", "b"],
        "literal": [[False, True], [True, True, False]],
    }
