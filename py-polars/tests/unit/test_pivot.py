from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars.internals.type_aliases import PivotAgg


def test_pivot() -> None:
    df = pl.DataFrame(
        {
            "foo": ["A", "A", "B", "B", "C"],
            "N": [1, 2, 2, 4, 2],
            "bar": ["k", "l", "m", "n", "o"],
        }
    )
    result = df.pivot(values="N", index="foo", columns="bar")

    expected = pl.DataFrame(
        [
            ("A", 1, 2, None, None, None),
            ("B", None, None, 2, 4, None),
            ("C", None, None, None, None, 2),
        ],
        columns=["foo", "k", "l", "m", "n", "o"],
    )
    assert_frame_equal(result, expected)


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
    assert_frame_equal(out, expected)


@pytest.mark.parametrize(
    ("agg_fn", "expected_rows"),
    [
        ("first", [("a", 2, None, None), ("b", None, None, 10)]),
        ("count", [("a", 2, None, None), ("b", None, 2, 1)]),
        ("min", [("a", 2, None, None), ("b", None, 8, 10)]),
        ("max", [("a", 4, None, None), ("b", None, 8, 10)]),
        ("sum", [("a", 6, None, None), ("b", None, 8, 10)]),
        ("mean", [("a", 3.0, None, None), ("b", None, 8.0, 10.0)]),
        ("median", [("a", 3.0, None, None), ("b", None, 8.0, 10.0)]),
    ],
)
def test_pivot_aggregate(agg_fn: PivotAgg, expected_rows: list[tuple[Any]]) -> None:
    df = pl.DataFrame(
        {
            "a": [1, 1, 2, 2, 3],
            "b": ["a", "a", "b", "b", "b"],
            "c": [2, 4, None, 8, 10],
        }
    )
    result = df.pivot(
        values="c", index="b", columns="a", aggregate_fn=agg_fn, sort_columns=True
    )
    assert result.rows() == expected_rows


@pytest.mark.parametrize(
    ("agg_fn", "expected_rows"),
    [
        ("first", [("a", 2, None, None), ("b", None, None, 10)]),
        ("count", [("a", 2, None, None), ("b", None, 2, 1)]),
        ("min", [("a", 2, None, None), ("b", None, 8, 10)]),
        ("max", [("a", 4, None, None), ("b", None, 8, 10)]),
        ("sum", [("a", 6, None, None), ("b", None, 8, 10)]),
        ("mean", [("a", 3.0, None, None), ("b", None, 8.0, 10.0)]),
        ("median", [("a", 3.0, None, None), ("b", None, 8.0, 10.0)]),
    ],
)
def test_pivot_groupby_aggregate(
    agg_fn: PivotAgg, expected_rows: list[tuple[Any]]
) -> None:
    df = pl.DataFrame(
        {
            "a": [1, 1, 2, 2, 3],
            "b": ["a", "a", "b", "b", "b"],
            "c": [2, 4, None, 8, 10],
        }
    )
    with pytest.deprecated_call():
        pivot = df.groupby("b").pivot(pivot_column="a", values_column="c")
    result = getattr(pivot, agg_fn)()
    assert result.rows() == expected_rows


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

    result = df.pivot(values="B", index=["A"], columns="B", aggregate_fn="count")
    expected = {"A": ["Fire", "Water"], "Car": [1, 2], "Ship": [1, None]}
    assert result.to_dict(False) == expected

    # test expression dispatch
    result = df.pivot(values="B", index=["A"], columns="B", aggregate_fn=pl.count())
    assert result.to_dict(False) == expected

    df = pl.DataFrame(
        {
            "A": ["Fire", "Water", "Water", "Fire"],
            "B": ["Car", "Car", "Car", "Ship"],
            "C": ["Paper", "Paper", "Paper", "Paper"],
        },
        columns=[("A", pl.Categorical), ("B", pl.Categorical), ("C", pl.Categorical)],
    )
    result = df.pivot(values="B", index=["A", "C"], columns="B", aggregate_fn="count")
    expected = {
        "A": ["Fire", "Water"],
        "C": ["Paper", "Paper"],
        "Car": [1, 2],
        "Ship": [1, None],
    }
    assert result.to_dict(False) == expected


def test_pivot_multiple_values_column_names_5116() -> None:
    df = pl.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5, 6, 7, 8],
            "x2": [8, 7, 6, 5, 4, 3, 2, 1],
            "c1": ["A", "B"] * 4,
            "c2": ["C", "C", "D", "D"] * 2,
        }
    )
    result = df.pivot(values=["x1", "x2"], index="c1", columns="c2")
    expected = {
        "c1": ["A", "B"],
        "x1_C": [1, 2],
        "x1_D": [3, 4],
        "x2_C": [8, 7],
        "x2_D": [6, 5],
    }
    assert result.to_dict(False) == expected


def test_pivot_floats() -> None:
    df = pl.DataFrame(
        {
            "article": ["a", "a", "a", "b", "b", "b"],
            "weight": [1.0, 1.0, 4.4, 1.0, 8.8, 1.0],
            "quantity": [1.0, 5.0, 1.0, 1.0, 1.0, 7.5],
            "price": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )

    result = df.pivot(values="price", index="weight", columns="quantity")
    expected = {
        "weight": [1.0, 4.4, 8.8],
        "1.0": [1.0, 3.0, 5.0],
        "5.0": [2.0, None, None],
        "7.5": [6.0, None, None],
    }
    assert result.to_dict(False) == expected

    result = df.pivot(values="price", index=["article", "weight"], columns="quantity")
    expected = {
        "article": ["a", "a", "b", "b"],
        "weight": [1.0, 4.4, 1.0, 8.8],
        "1.0": [1.0, 3.0, 4.0, 5.0],
        "5.0": [2.0, None, None, None],
        "7.5": [None, None, 6.0, None],
    }
    assert result.to_dict(False) == expected


def test_pivot_reinterpret_5907() -> None:
    df = pl.DataFrame(
        {
            "A": pl.Series([3, -2, 3, -2], dtype=pl.Int32),
            "B": ["x", "x", "y", "y"],
            "C": [100, 50, 500, -80],
        }
    )

    result = df.pivot(
        index=["A"], values=["C"], columns=["B"], aggregate_fn=pl.element().sum()
    )
    expected = {"A": [3, -2], "x": [100, 50], "y": [500, -80]}
    assert result.to_dict(False) == expected


def test_pivot_subclassed_df() -> None:
    class SubClassedDataFrame(pl.DataFrame):
        pass

    df = SubClassedDataFrame({"a": [1, 2], "b": [3, 4]})
    result = df.pivot(values="b", index="a", columns="a", aggregate_fn="first")
    assert isinstance(result, SubClassedDataFrame)
