from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from polars.internals.type_aliases import PivotAgg


def test_pivot_list() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [[1, 1], [2, 2], [3, 3]]})

    expected = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "1": [[1, 1], None, None],
            "2": [None, [2, 2], None],
            "3": [None, None, [3, 3]],
        }
    )
    out = df.pivot("b", index="a", columns="a", aggregate_fn="first", sort_columns=True)
    assert out.frame_equal(expected, null_equal=True)


def test_pivot() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 1, 2, 2, 3],
            "b": ["a", "a", "b", "b", "b"],
            "c": [2, 4, None, 8, 10],
        }
    )
    gb = df.groupby("b").pivot(
        pivot_column="a",
        values_column="c",
    )
    assert gb.count().rows() == [("a", 2, None, None), ("b", None, 2, 1)]
    assert gb.first().rows() == [("a", 2, None, None), ("b", None, None, 10)]
    assert gb.max().rows() == [("a", 4, None, None), ("b", None, 8, 10)]
    assert gb.mean().rows() == [("a", 3.0, None, None), ("b", None, 8.0, 10.0)]
    assert gb.median().rows() == [("a", 3.0, None, None), ("b", None, 8.0, 10.0)]
    assert gb.min().rows() == [("a", 2, None, None), ("b", None, 8, 10)]
    assert gb.sum().rows() == [("a", 6, None, None), ("b", None, 8, 10)]

    agg_fns: list[PivotAgg] = ["sum", "min", "max", "mean", "count", "median", "mean"]
    for agg_fn in agg_fns:
        out = df.pivot(
            values="c", index="b", columns="a", aggregate_fn=agg_fn, sort_columns=True
        )
        assert out.shape == (2, 4)
        assert out.rows() == getattr(gb, agg_fn)().rows()

    # example in polars-book
    df = pl.DataFrame(
        {
            "foo": ["A", "A", "B", "B", "C"],
            "N": [1, 2, 2, 4, 2],
            "bar": ["k", "l", "m", "n", "o"],
        }
    )
    out = df.groupby("foo").pivot(pivot_column="bar", values_column="N").first()
    assert out.shape == (3, 6)
    assert out.rows() == [
        ("A", 1, 2, None, None, None),
        ("B", None, None, 2, 4, None),
        ("C", None, None, None, None, 2),
    ]


def test_pivot_categorical_3968() -> None:
    df = pl.DataFrame(
        {
            "foo": ["one", "one", "one", "two", "two", "two"],
            "bar": ["A", "B", "C", "A", "B", "C"],
            "baz": [1, 2, 3, 4, 5, 6],
        }
    )

    assert df.with_column(pl.col("baz").cast(str).cast(pl.Categorical)).to_dict(
        False
    ) == {
        "foo": ["one", "one", "one", "two", "two", "two"],
        "bar": ["A", "B", "C", "A", "B", "C"],
        "baz": ["1", "2", "3", "4", "5", "6"],
    }


def test_pivot_categorical_index() -> None:
    df = pl.DataFrame(
        {"A": ["Fire", "Water", "Water", "Fire"], "B": ["Car", "Car", "Car", "Ship"]},
        columns=[("A", pl.Categorical), ("B", pl.Categorical)],
    )

    expected = {"A": ["Fire", "Water"], "Car": [1, 2], "Ship": [1, None]}
    assert (
        df.pivot(values="B", index=["A"], columns="B", aggregate_fn="count").to_dict(
            False
        )
        == expected
    )

    # test expression dispatch
    assert (
        df.pivot(values="B", index=["A"], columns="B", aggregate_fn=pl.count()).to_dict(
            False
        )
        == expected
    )

    df = pl.DataFrame(
        {
            "A": ["Fire", "Water", "Water", "Fire"],
            "B": ["Car", "Car", "Car", "Ship"],
            "C": ["Paper", "Paper", "Paper", "Paper"],
        },
        columns=[("A", pl.Categorical), ("B", pl.Categorical), ("C", pl.Categorical)],
    )
    assert df.pivot(
        values="B", index=["A", "C"], columns="B", aggregate_fn="count"
    ).to_dict(False) == {
        "A": ["Fire", "Water"],
        "C": ["Paper", "Paper"],
        "Car": [1, 2],
        "Ship": [1, None],
    }


def test_pivot_multiple_values_column_names_5116() -> None:
    df = pl.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5, 6, 7, 8],
            "x2": [8, 7, 6, 5, 4, 3, 2, 1],
            "c1": ["A", "B"] * 4,
            "c2": ["C", "C", "D", "D"] * 2,
        }
    )
    assert df.pivot(values=["x1", "x2"], index="c1", columns="c2").to_dict(False) == {
        "c1": ["A", "B"],
        "x1_C": [1, 2],
        "x1_D": [3, 4],
        "x2_C": [8, 7],
        "x2_D": [6, 5],
    }


def test_pivot_floats() -> None:

    df = pl.DataFrame(
        {
            "article": ["a", "a", "a", "b", "b", "b"],
            "weight": [1.0, 1.0, 4.4, 1.0, 8.8, 1.0],
            "quantity": [1.0, 5.0, 1.0, 1.0, 1.0, 7.5],
            "price": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )

    assert df.pivot(values="price", index="weight", columns="quantity",).to_dict(
        False
    ) == {
        "weight": [1.0, 4.4, 8.8],
        "1.0": [1.0, 3.0, 5.0],
        "5.0": [2.0, None, None],
        "7.5": [6.0, None, None],
    }

    assert df.pivot(
        values="price",
        index=["article", "weight"],
        columns="quantity",
    ).to_dict(False) == {
        "article": ["a", "a", "b", "b"],
        "weight": [1.0, 4.4, 1.0, 8.8],
        "1.0": [1.0, 3.0, 4.0, 5.0],
        "5.0": [2.0, None, None, None],
        "7.5": [None, None, 6.0, None],
    }
