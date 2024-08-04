from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import polars as pl
from polars.exceptions import DuplicateError, InvalidOperationError
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from polars._typing import ConcatMethod


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


def test_concat_align_no_common_cols() -> None:
    df1 = pl.DataFrame({"a": [1, 2], "b": [1, 2]})
    df2 = pl.DataFrame({"c": [3, 4], "d": [3, 4]})

    with pytest.raises(
        InvalidOperationError,
        match="'align' strategy requires at least one common column",
    ):
        pl.concat((df1, df2), how="align")


data2 = pl.DataFrame({"field3": [3, 4], "field4": ["C", "D"]})


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


def test_concat_diagonal_relaxed_with_empty_frame() -> None:
    df1 = pl.DataFrame()
    df2 = pl.DataFrame(
        {
            "a": ["a", "b"],
            "b": [1, 2],
        }
    )
    out = pl.concat((df1, df2), how="diagonal_relaxed")
    expected = df2
    assert_frame_equal(out, expected)


@pytest.mark.parametrize("lazy", [False, True])
def test_concat_horizontal(lazy: bool) -> None:
    a = pl.DataFrame({"a": ["a", "b"], "b": [1, 2]})
    b = pl.DataFrame({"c": [5, 7, 8, 9], "d": [1, 2, 1, 2], "e": [1, 2, 1, 2]})

    if lazy:
        out = pl.concat([a.lazy(), b.lazy()], how="horizontal").collect()
    else:
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


@pytest.mark.parametrize("lazy", [False, True])
def test_concat_horizontal_three_dfs(lazy: bool) -> None:
    a = pl.DataFrame({"a1": [1, 2, 3], "a2": ["a", "b", "c"]})
    b = pl.DataFrame({"b1": [0.25, 0.5]})
    c = pl.DataFrame({"c1": [1, 2, 3, 4], "c2": [5, 6, 7, 8], "c3": [9, 10, 11, 12]})

    if lazy:
        out = pl.concat([a.lazy(), b.lazy(), c.lazy()], how="horizontal").collect()
    else:
        out = pl.concat([a, b, c], how="horizontal")

    expected = pl.DataFrame(
        {
            "a1": [1, 2, 3, None],
            "a2": ["a", "b", "c", None],
            "b1": [0.25, 0.5, None, None],
            "c1": [1, 2, 3, 4],
            "c2": [5, 6, 7, 8],
            "c3": [9, 10, 11, 12],
        }
    )
    assert_frame_equal(out, expected)


@pytest.mark.parametrize("lazy", [False, True])
def test_concat_horizontal_single_df(lazy: bool) -> None:
    a = pl.DataFrame({"a": ["a", "b"], "b": [1, 2]})

    if lazy:
        out = pl.concat([a.lazy()], how="horizontal").collect()
    else:
        out = pl.concat([a], how="horizontal")

    expected = a
    assert_frame_equal(out, expected)


def test_concat_horizontal_duplicate_col() -> None:
    a = pl.LazyFrame({"a": ["a", "b"], "b": [1, 2]})
    b = pl.LazyFrame({"c": [5, 7, 8, 9], "d": [1, 2, 1, 2], "a": [1, 2, 1, 2]})

    with pytest.raises(DuplicateError):
        pl.concat([a, b], how="horizontal").collect()


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


def test_extend_ints() -> None:
    a = pl.DataFrame({"a": [1 for _ in range(1)]}, schema={"a": pl.Int64})
    with pytest.raises(pl.exceptions.SchemaError):
        a.extend(a.select(pl.lit(0, dtype=pl.Int32).alias("a")))


def test_null_handling_correlation() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, None, 4], "b": [1, 2, 3, 10, 4]})

    out = df.select(
        pl.corr("a", "b").alias("pearson"),
        pl.corr("a", "b", method="spearman").alias("spearman"),
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
    pdf1 = pd.DataFrame(
        {
            "date": pd.date_range(start="2019-01-02", periods=9),
            "a": np.array([0, 1, 2, np.nan, 4, 5, 6, 7, 8], dtype=np.float64),
            "b": np.arange(9, 18, dtype=np.float64),
        }
    ).set_index("date")

    pdf2 = pd.DataFrame(
        {
            "date": pd.date_range(start="2019-01-04", periods=7),
            "a": np.arange(9, 16, dtype=np.float64),
            "b": np.arange(10, 17, dtype=np.float64),
        }
    ).set_index("date")

    # calculate dot-product in pandas
    pd_dot = (pdf1 * pdf2).sum(axis="columns").to_frame("dot").reset_index()

    # use "align_frames" to calculate dot-product from disjoint rows. pandas uses an
    # index to automatically infer the correct frame-alignment for the calculation;
    # we need to do it explicitly (which also makes it clearer what is happening)
    pf1, pf2 = pl.align_frames(
        pl.from_pandas(pdf1.reset_index()),
        pl.from_pandas(pdf2.reset_index()),
        on="date",
    )
    pl_dot = (
        (pf1[["a", "b"]] * pf2[["a", "b"]])
        .fill_null(0)
        .select(pl.sum_horizontal("*").alias("dot"))
        .insert_column(0, pf1["date"])
    )
    # confirm we match the same operation in pandas
    assert_frame_equal(pl_dot, pl.from_pandas(pd_dot))
    pd.testing.assert_frame_equal(pd_dot, pl_dot.to_pandas())

    # (also: confirm alignment function works with lazyframes)
    lf1, lf2 = pl.align_frames(
        pl.from_pandas(pdf1.reset_index()).lazy(),
        pl.from_pandas(pdf2.reset_index()).lazy(),
        on="date",
    )
    assert isinstance(lf1, pl.LazyFrame)
    assert_frame_equal(lf1.collect(), pf1)
    assert_frame_equal(lf2.collect(), pf2)

    # misc
    assert pl.align_frames(on="date") == []

    # expected error condition
    with pytest.raises(TypeError):
        pl.align_frames(  # type: ignore[type-var]
            pl.from_pandas(pdf1.reset_index()).lazy(),
            pl.from_pandas(pdf2.reset_index()),
            on="date",
        )

    # descending result
    df1 = pl.DataFrame([[3, 5, 6], [5, 8, 9]], orient="row")
    df2 = pl.DataFrame([[2, 5, 6], [3, 8, 9], [4, 2, 0]], orient="row")

    pf1, pf2 = pl.align_frames(df1, df2, on="column_0", descending=True)
    assert pf1.rows() == [(5, 8, 9), (4, None, None), (3, 5, 6), (2, None, None)]
    assert pf2.rows() == [(5, None, None), (4, 2, 0), (3, 8, 9), (2, 5, 6)]

    # handle identical frames
    pf1, pf2, pf3 = pl.align_frames(df1, df2, df2, on="column_0", descending=True)
    assert pf1.rows() == [(5, 8, 9), (4, None, None), (3, 5, 6), (2, None, None)]
    for pf in (pf2, pf3):
        assert pf.rows() == [(5, None, None), (4, 2, 0), (3, 8, 9), (2, 5, 6)]


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
    assert df.select(pl.col("a").cast(pl.UInt64).diff()).to_dict(as_series=False) == {
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
    ).to_dict(as_series=False) == {
        "a": [
            1.0,
            7.38905609893065,
            20.085536923187668,
            54.598150033144236,
            148.4131591025766,
        ]
    }


def test_approx_n_unique() -> None:
    df1 = pl.DataFrame({"a": [None, 1, 2], "b": [None, 2, 1]})

    assert_frame_equal(
        df1.select(pl.approx_n_unique("b")),
        pl.DataFrame({"b": pl.Series(values=[3], dtype=pl.UInt32)}),
    )

    assert_frame_equal(
        df1.select(pl.col("b").approx_n_unique()),
        pl.DataFrame({"b": pl.Series(values=[3], dtype=pl.UInt32)}),
    )


def test_lazy_functions() -> None:
    df = pl.DataFrame(
        {
            "a": ["foo", "bar", "foo"],
            "b": [1, 2, 3],
            "c": [-1.0, 2.0, 4.0],
        }
    )

    # test function expressions against frame
    out = df.select(
        pl.var("b").name.suffix("_var"),
        pl.std("b").name.suffix("_std"),
        pl.max("a", "b").name.suffix("_max"),
        pl.min("a", "b").name.suffix("_min"),
        pl.sum("b", "c").name.suffix("_sum"),
        pl.mean("b", "c").name.suffix("_mean"),
        pl.median("c", "b").name.suffix("_median"),
        pl.n_unique("b", "a").name.suffix("_n_unique"),
        pl.first("a").name.suffix("_first"),
        pl.first("b", "c").name.suffix("_first"),
        pl.last("c", "b", "a").name.suffix("_last"),
    )
    expected: dict[str, list[Any]] = {
        "b_var": [1.0],
        "b_std": [1.0],
        "a_max": ["foo"],
        "b_max": [3],
        "a_min": ["bar"],
        "b_min": [1],
        "b_sum": [6],
        "c_sum": [5.0],
        "b_mean": [2.0],
        "c_mean": [5 / 3],
        "c_median": [2.0],
        "b_median": [2.0],
        "b_n_unique": [3],
        "a_n_unique": [2],
        "a_first": ["foo"],
        "b_first": [1],
        "c_first": [-1.0],
        "c_last": [4.0],
        "b_last": [3],
        "a_last": ["foo"],
    }
    assert_frame_equal(
        out,
        pl.DataFrame(
            data=expected,
            schema_overrides={
                "a_n_unique": pl.UInt32,
                "b_n_unique": pl.UInt32,
            },
        ),
    )

    # test function expressions against series
    for name, value in expected.items():
        col, fn = name.split("_", 1)
        if series_fn := getattr(df[col], fn, None):
            assert series_fn() == value[0]

    # regex selection
    out = df.select(
        pl.struct(pl.max("^a|b$")).alias("x"),
        pl.struct(pl.min("^.*[bc]$")).alias("y"),
        pl.struct(pl.sum("^[^a]$")).alias("z"),
    )
    assert out.rows() == [
        ({"a": "foo", "b": 3}, {"b": 1, "c": -1.0}, {"b": 6, "c": 5.0})
    ]


def test_count() -> None:
    df = pl.DataFrame({"a": [1, 1, 1], "b": [None, "xx", "yy"]})
    out = df.select(pl.count("a"))
    assert list(out["a"]) == [3]

    for count_expr in (
        pl.count("b", "a"),
        [pl.count("b"), pl.count("a")],
    ):
        out = df.select(count_expr)
        assert out.rows() == [(2, 3)]


def test_head_tail(fruits_cars: pl.DataFrame) -> None:
    res_expr = fruits_cars.select(pl.head("A", 2))
    expected = pl.Series("A", [1, 2])
    assert_series_equal(res_expr.to_series(), expected)

    res_expr = fruits_cars.select(pl.tail("A", 2))
    expected = pl.Series("A", [4, 5])
    assert_series_equal(res_expr.to_series(), expected)
