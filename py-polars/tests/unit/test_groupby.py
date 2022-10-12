from __future__ import annotations

from datetime import datetime, timedelta

import pytest

import polars as pl


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

    pl.testing.assert_series_equal(out["result_a"], out["result_b"], check_names=False)
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
    assert (
        pl.DataFrame(
            {
                "a": [
                    datetime(2021, 1, 1) + timedelta(seconds=2**i) for i in range(10)
                ],
                "b": [float(i) for i in range(10)],
            }
        )
        .lazy()
        .groupby_dynamic("a", every="10s", period="100s")
        .agg([pl.col("b").var().sqrt().alias("corr")])
    ).collect().sum().to_dict(False) == pytest.approx(
        {"a": [None], "corr": [6.988674024215477]}
    )
