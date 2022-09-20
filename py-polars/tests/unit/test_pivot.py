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
            "a": [1, 2, 3, 4, 5],
            "b": ["a", "a", "b", "b", "b"],
            "c": [None, 1, None, 1, None],
        }
    )
    gb = df.groupby("b").pivot("a", "c")
    assert gb.first().shape == (2, 6)
    assert gb.max().shape == (2, 6)
    assert gb.mean().shape == (2, 6)
    assert gb.count().shape == (2, 6)
    assert gb.median().shape == (2, 6)

    agg_fns: list[PivotAgg] = ["sum", "min", "max", "mean", "count", "median", "mean"]
    for agg_fn in agg_fns:
        out = df.pivot(
            values="c", index="b", columns="a", aggregate_fn=agg_fn, sort_columns=True
        )
        assert out.shape == (2, 6)

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
