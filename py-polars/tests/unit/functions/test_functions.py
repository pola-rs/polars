from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np
import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from polars.type_aliases import ConcatMethod


def test_concat_align() -> None:
    a = pl.DataFrame({"a": ["a", "b", "d", "e", "e"], "b": [1, 2, 4, 5, 6]})
    b = pl.DataFrame({"a": ["a", "b", "c"], "c": [5.5, 6.0, 7.5]})
    c = pl.DataFrame({"a": ["a", "b", "c", "d", "e"], "d": ["w", "x", "y", "z", None]})

    result = pl.concat([a, b, c], how="align")

    expected = pl.DataFrame(
        {
            "a": ["a", "b", "c", "d", "e", "e"],
            "b": [1, 2, None, 4, 5, 6],
            "c": [5.5, 6.0, 7.5, None, None, None],
            "d": ["w", "x", "y", "z", None, None],
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("a", "b", "c", "strategy"),
    [
        (
            pl.DataFrame({"a": [1, 2]}),
            pl.DataFrame({"b": ["a", "b"], "c": [3, 4]}),
            pl.DataFrame({"a": [5, 6], "c": [5, 6], "d": [5, 6], "b": ["x", "y"]}),
            "diagonal",
        ),
        (
            pl.DataFrame(
                {"a": [1, 2]},
                schema_overrides={"a": pl.Int32},
            ),
            pl.DataFrame(
                {"b": ["a", "b"], "c": [3, 4]},
                schema_overrides={"c": pl.UInt8},
            ),
            pl.DataFrame(
                {"a": [5, 6], "c": [5, 6], "d": [5, 6], "b": ["x", "y"]},
                schema_overrides={"b": pl.Categorical},
            ),
            "diagonal_relaxed",
        ),
    ],
)
def test_concat_diagonal(
    a: pl.DataFrame, b: pl.DataFrame, c: pl.DataFrame, strategy: ConcatMethod
) -> None:
    for out in [
        pl.concat([a, b, c], how=strategy),
        pl.concat([a.lazy(), b.lazy(), c.lazy()], how=strategy).collect(),
    ]:
        expected = pl.DataFrame(
            {
                "a": [1, 2, None, None, 5, 6],
                "b": [None, None, "a", "b", "x", "y"],
                "c": [None, None, 3, 4, 5, 6],
                "d": [None, None, None, None, 5, 6],
            }
        )
        assert_frame_equal(out, expected)


def test_concat_horizontal() -> None:
    a = pl.DataFrame({"a": ["a", "b"], "b": [1, 2]})
    b = pl.DataFrame({"c": [5, 7, 8, 9], "d": [1, 2, 1, 2], "e": [1, 2, 1, 2]})

    out = pl.concat([a, b], how="horizontal")
    expected = pl.DataFrame(
        {
            "a": ["a", "b", None, None],
            "b": [1, 2, None, None],
            "c": [5, 7, 8, 9],
            "d": [1, 2, 1, 2],
            "e": [1, 2, 1, 2],
        }
    )
    assert_frame_equal(out, expected)


def test_concat_vertical() -> None:
    a = pl.DataFrame({"a": ["a", "b"], "b": [1, 2]})
    b = pl.DataFrame({"a": ["c", "d", "e"], "b": [3, 4, 5]})

    result = pl.concat([a, b], how="vertical")
    expected = pl.DataFrame(
        {
            "a": ["a", "b", "c", "d", "e"],
            "b": [1, 2, 3, 4, 5],
        }
    )
    assert_frame_equal(result, expected)


def test_null_handling_correlation() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, None, 4], "b": [1, 2, 3, 10, 4]})

    out = df.select(
        [
            pl.corr("a", "b").alias("pearson"),
            pl.corr("a", "b", method="spearman").alias("spearman"),
        ]
    )
    assert out["pearson"][0] == pytest.approx(1.0)
    assert out["spearman"][0] == pytest.approx(1.0)

    # see #4930
    df1 = pl.DataFrame({"a": [None, 1, 2], "b": [None, 2, 1]})
    df2 = pl.DataFrame({"a": [np.nan, 1, 2], "b": [np.nan, 2, 1]})

    assert np.isclose(df1.select(pl.corr("a", "b", method="spearman")).item(), -1.0)
    assert (
        str(
            df2.select(pl.corr("a", "b", method="spearman", propagate_nans=True)).item()
        )
        == "nan"
    )


def test_align_frames() -> None:
    import numpy as np
    import pandas as pd

    # setup some test frames
    df1 = pd.DataFrame(
        {
            "date": pd.date_range(start="2019-01-02", periods=9),
            "a": np.array([0, 1, 2, np.nan, 4, 5, 6, 7, 8], dtype=np.float64),
            "b": np.arange(9, 18, dtype=np.float64),
        }
    ).set_index("date")

    df2 = pd.DataFrame(
        {
            "date": pd.date_range(start="2019-01-04", periods=7),
            "a": np.arange(9, 16, dtype=np.float64),
            "b": np.arange(10, 17, dtype=np.float64),
        }
    ).set_index("date")

    # calculate dot-product in pandas
    pd_dot = (df1 * df2).sum(axis="columns").to_frame("dot").reset_index()

    # use "align_frames" to calculate dot-product from disjoint rows. pandas uses an
    # index to automatically infer the correct frame-alignment for the calculation;
    # we need to do it explicitly (which also makes it clearer what is happening)
    pf1, pf2 = pl.align_frames(
        pl.from_pandas(df1.reset_index()),
        pl.from_pandas(df2.reset_index()),
        on="date",
    )
    pl_dot = (
        (pf1[["a", "b"]] * pf2[["a", "b"]])
        .fill_null(0)
        .select(pl.sum_horizontal("*").alias("dot"))
        .insert_at_idx(0, pf1["date"])
    )
    # confirm we match the same operation in pandas
    assert_frame_equal(pl_dot, pl.from_pandas(pd_dot))
    pd.testing.assert_frame_equal(pd_dot, pl_dot.to_pandas())

    # (also: confirm alignment function works with lazyframes)
    lf1, lf2 = pl.align_frames(
        pl.from_pandas(df1.reset_index()).lazy(),
        pl.from_pandas(df2.reset_index()).lazy(),
        on="date",
    )
    assert isinstance(lf1, pl.LazyFrame)
    assert_frame_equal(lf1.collect(), pf1)
    assert_frame_equal(lf2.collect(), pf2)

    # misc
    assert [] == pl.align_frames(on="date")

    # expected error condition
    with pytest.raises(TypeError):
        pl.align_frames(  # type: ignore[type-var]
            pl.from_pandas(df1.reset_index()).lazy(),
            pl.from_pandas(df2.reset_index()),
            on="date",
        )

    # descending
    pf1, pf2 = pl.align_frames(
        pl.DataFrame([[3, 5, 6], [5, 8, 9]], orient="row"),
        pl.DataFrame([[2, 5, 6], [3, 8, 9], [4, 2, 0]], orient="row"),
        on="column_0",
        descending=True,
    )
    assert pf1.rows() == [(5, 8, 9), (4, None, None), (3, 5, 6), (2, None, None)]
    assert pf2.rows() == [(5, None, None), (4, 2, 0), (3, 8, 9), (2, 5, 6)]


def test_align_frames_duplicate_key() -> None:
    # setup some test frames with duplicate key/alignment values
    df1 = pl.DataFrame({"x": ["a", "a", "a", "e"], "y": [1, 2, 4, 5]})
    df2 = pl.DataFrame({"y": [0, 0, -1], "z": [5.5, 6.0, 7.5], "x": ["a", "b", "b"]})

    # align rows, confirming correctness and original column order
    af1, af2 = pl.align_frames(df1, df2, on="x")

    # shape: (6, 2)   shape: (6, 3)
    # ┌─────┬──────┐  ┌──────┬──────┬─────┐
    # │ x   ┆ y    │  │ y    ┆ z    ┆ x   │
    # │ --- ┆ ---  │  │ ---  ┆ ---  ┆ --- │
    # │ str ┆ i64  │  │ i64  ┆ f64  ┆ str │
    # ╞═════╪══════╡  ╞══════╪══════╪═════╡
    # │ a   ┆ 1    │  │ 0    ┆ 5.5  ┆ a   │
    # │ a   ┆ 2    │  │ 0    ┆ 5.5  ┆ a   │
    # │ a   ┆ 4    │  │ 0    ┆ 5.5  ┆ a   │
    # │ b   ┆ null │  │ 0    ┆ 6.0  ┆ b   │
    # │ b   ┆ null │  │ -1   ┆ 7.5  ┆ b   │
    # │ e   ┆ 5    │  │ null ┆ null ┆ e   │
    # └─────┴──────┘  └──────┴──────┴─────┘
    assert af1.rows() == [
        ("a", 1),
        ("a", 2),
        ("a", 4),
        ("b", None),
        ("b", None),
        ("e", 5),
    ]
    assert af2.rows() == [
        (0, 5.5, "a"),
        (0, 5.5, "a"),
        (0, 5.5, "a"),
        (0, 6.0, "b"),
        (-1, 7.5, "b"),
        (None, None, "e"),
    ]

    # align frames the other way round, using "left" alignment strategy
    af1, af2 = pl.align_frames(df2, df1, on="x", how="left")

    # shape: (5, 3)        shape: (5, 2)
    # ┌─────┬─────┬─────┐  ┌─────┬──────┐
    # │ y   ┆ z   ┆ x   │  │ x   ┆ y    │
    # │ --- ┆ --- ┆ --- │  │ --- ┆ ---  │
    # │ i64 ┆ f64 ┆ str │  │ str ┆ i64  │
    # ╞═════╪═════╪═════╡  ╞═════╪══════╡
    # │ 0   ┆ 5.5 ┆ a   │  │ a   ┆ 1    │
    # │ 0   ┆ 5.5 ┆ a   │  │ a   ┆ 2    │
    # │ 0   ┆ 5.5 ┆ a   │  │ a   ┆ 4    │
    # │ 0   ┆ 6.0 ┆ b   │  │ b   ┆ null │
    # │ -1  ┆ 7.5 ┆ b   │  │ b   ┆ null │
    # └─────┴─────┴─────┘  └─────┴──────┘
    assert af1.rows() == [
        (0, 5.5, "a"),
        (0, 5.5, "a"),
        (0, 5.5, "a"),
        (0, 6.0, "b"),
        (-1, 7.5, "b"),
    ]
    assert af2.rows() == [
        ("a", 1),
        ("a", 2),
        ("a", 4),
        ("b", None),
        ("b", None),
    ]


def test_coalesce() -> None:
    df = pl.DataFrame(
        {
            "a": [1, None, None, None],
            "b": [1, 2, None, None],
            "c": [5, None, 3, None],
        }
    )

    # List inputs
    expected = pl.Series("d", [1, 2, 3, 10]).to_frame()
    result = df.select(pl.coalesce(["a", "b", "c", 10]).alias("d"))
    assert_frame_equal(expected, result)

    # Positional inputs
    expected = pl.Series("d", [1.0, 2.0, 3.0, 10.0]).to_frame()
    result = df.select(pl.coalesce(pl.col(["a", "b", "c"]), 10.0).alias("d"))
    assert_frame_equal(result, expected)


def test_overflow_diff() -> None:
    df = pl.DataFrame(
        {
            "a": [20, 10, 30],
        }
    )
    assert df.select(pl.col("a").cast(pl.UInt64).diff()).to_dict(False) == {
        "a": [None, -10, 20]
    }


def test_fill_null_unknown_output_type() -> None:
    df = pl.DataFrame(
        {
            "a": [
                None,
                2,
                3,
                4,
                5,
            ]
        }
    )
    assert df.with_columns(
        np.exp(pl.col("a")).fill_null(pl.lit(1, pl.Float64))
    ).to_dict(False) == {
        "a": [
            1.0,
            7.38905609893065,
            20.085536923187668,
            54.598150033144236,
            148.4131591025766,
        ]
    }


def test_abs_logical_type() -> None:
    s = pl.Series([timedelta(hours=1), timedelta(hours=-1)])
    assert s.abs().to_list() == [timedelta(hours=1), timedelta(hours=1)]


def test_approx_n_unique() -> None:
    df1 = pl.DataFrame({"a": [None, 1, 2], "b": [None, 2, 1]})

    assert_frame_equal(
        df1.select(pl.approx_n_unique("b")),
        pl.DataFrame({"b": pl.Series(values=[3], dtype=pl.UInt32)}),
    )

    assert_frame_equal(
        df1.select(pl.approx_n_unique(pl.col("b"))),
        pl.DataFrame({"b": pl.Series(values=[3], dtype=pl.UInt32)}),
    )

    assert_frame_equal(
        df1.select(pl.col("b").approx_n_unique()),
        pl.DataFrame({"b": pl.Series(values=[3], dtype=pl.UInt32)}),
    )


def test_lazy_functions() -> None:
    df = pl.DataFrame({"a": ["foo", "bar", "2"], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})
    out = df.select(pl.count("a"))
    assert list(out["a"]) == [3]
    with pytest.deprecated_call():
        assert pl.count(df["a"]) == 3
    out = df.select(
        [
            pl.var("b").alias("1"),
            pl.std("b").alias("2"),
            pl.max("b").alias("3"),
            pl.min("b").alias("4"),
            pl.sum("b").alias("5"),
            pl.mean("b").alias("6"),
            pl.median("b").alias("7"),
            pl.n_unique("b").alias("8"),
            pl.first("b").alias("9"),
            pl.last("b").alias("10"),
        ]
    )
    expected = 1.0
    assert np.isclose(out.to_series(0), expected)
    assert np.isclose(df["b"].var(), expected)  # type: ignore[arg-type]

    expected = 1.0
    assert np.isclose(out.to_series(1), expected)
    assert np.isclose(df["b"].std(), expected)  # type: ignore[arg-type]

    expected = 3
    assert np.isclose(out.to_series(2), expected)
    assert np.isclose(df["b"].max(), expected)  # type: ignore[arg-type]

    expected = 1
    assert np.isclose(out.to_series(3), expected)
    assert np.isclose(df["b"].min(), expected)  # type: ignore[arg-type]

    expected = 6
    assert np.isclose(out.to_series(4), expected)
    assert np.isclose(df["b"].sum(), expected)

    expected = 2
    assert np.isclose(out.to_series(5), expected)
    assert np.isclose(df["b"].mean(), expected)  # type: ignore[arg-type]

    expected = 2
    assert np.isclose(out.to_series(6), expected)
    assert np.isclose(df["b"].median(), expected)  # type: ignore[arg-type]

    expected = 3
    assert np.isclose(out.to_series(7), expected)
    assert np.isclose(df["b"].n_unique(), expected)

    expected = 1
    assert np.isclose(out.to_series(8), expected)
    assert np.isclose(df["b"][0], expected)

    expected = 3
    assert np.isclose(out.to_series(9), expected)
    assert np.isclose(df["b"][-1], expected)

    # regex selection
    out = df.select(
        [
            pl.struct(pl.max("^a|b$")).alias("x"),
            pl.struct(pl.min("^.*[bc]$")).alias("y"),
            pl.struct(pl.sum("^[^b]$")).alias("z"),
        ]
    )
    assert out.rows() == [
        ({"a": "foo", "b": 3}, {"b": 1, "c": 1.0}, {"a": None, "c": 6.0})
    ]


def test_head_tail(fruits_cars: pl.DataFrame) -> None:
    res_expr = fruits_cars.select([pl.head("A", 2)])
    with pytest.deprecated_call():
        res_series = pl.head(fruits_cars["A"], 2)
    expected = pl.Series("A", [1, 2])
    assert_series_equal(res_expr.to_series(0), expected)
    assert_series_equal(res_series, expected)

    res_expr = fruits_cars.select([pl.tail("A", 2)])
    with pytest.deprecated_call():
        res_series = pl.tail(fruits_cars["A"], 2)
    expected = pl.Series("A", [4, 5])
    assert_series_equal(res_expr.to_series(0), expected)
    assert_series_equal(res_series, expected)
