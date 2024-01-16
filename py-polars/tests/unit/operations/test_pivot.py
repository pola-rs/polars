from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
import polars.selectors as cs
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars.type_aliases import PivotAgg


def test_pivot() -> None:
    df = pl.DataFrame(
        {
            "foo": ["A", "A", "B", "B", "C"],
            "N": [1, 2, 2, 4, 2],
            "bar": ["k", "l", "m", "n", "o"],
        }
    )
    result = df.pivot(values="N", index="foo", columns="bar", aggregate_function=None)

    expected = pl.DataFrame(
        [
            ("A", 1, 2, None, None, None),
            ("B", None, None, 2, 4, None),
            ("C", None, None, None, None, 2),
        ],
        schema=["foo", "k", "l", "m", "n", "o"],
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
    out = df.pivot(
        "b", index="a", columns="a", aggregate_function="first", sort_columns=True
    )
    assert_frame_equal(out, expected)


@pytest.mark.parametrize(
    ("agg_fn", "expected_rows"),
    [
        ("first", [("a", 2, None, None), ("b", None, None, 10)]),
        ("len", [("a", 2, None, None), ("b", None, 2, 1)]),
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
        values="c", index="b", columns="a", aggregate_function=agg_fn, sort_columns=True
    )
    assert result.rows() == expected_rows


def test_pivot_categorical_3968() -> None:
    df = pl.DataFrame(
        {
            "foo": ["one", "one", "one", "two", "two", "two"],
            "bar": ["A", "B", "C", "A", "B", "C"],
            "baz": [1, 2, 3, 4, 5, 6],
        }
    )

    result = df.with_columns(pl.col("baz").cast(str).cast(pl.Categorical))

    expected = pl.DataFrame(
        {
            "foo": ["one", "one", "one", "two", "two", "two"],
            "bar": ["A", "B", "C", "A", "B", "C"],
            "baz": ["1", "2", "3", "4", "5", "6"],
        },
        schema_overrides={"baz": pl.Categorical},
    )
    assert_frame_equal(result, expected, categorical_as_str=True)


def test_pivot_categorical_index() -> None:
    df = pl.DataFrame(
        {"A": ["Fire", "Water", "Water", "Fire"], "B": ["Car", "Car", "Car", "Ship"]},
        schema=[("A", pl.Categorical), ("B", pl.Categorical)],
    )

    result = df.pivot(values="B", index=["A"], columns="B", aggregate_function="len")
    expected = {"A": ["Fire", "Water"], "Car": [1, 2], "Ship": [1, None]}
    assert result.to_dict(as_series=False) == expected

    # test expression dispatch
    result = df.pivot(values="B", index=["A"], columns="B", aggregate_function=pl.len())
    assert result.to_dict(as_series=False) == expected

    df = pl.DataFrame(
        {
            "A": ["Fire", "Water", "Water", "Fire"],
            "B": ["Car", "Car", "Car", "Ship"],
            "C": ["Paper", "Paper", "Paper", "Paper"],
        },
        schema=[("A", pl.Categorical), ("B", pl.Categorical), ("C", pl.Categorical)],
    )
    result = df.pivot(
        values="B", index=["A", "C"], columns="B", aggregate_function="len"
    )
    expected = {
        "A": ["Fire", "Water"],
        "C": ["Paper", "Paper"],
        "Car": [1, 2],
        "Ship": [1, None],
    }
    assert result.to_dict(as_series=False) == expected


def test_pivot_multiple_values_column_names_5116() -> None:
    df = pl.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5, 6, 7, 8],
            "x2": [8, 7, 6, 5, 4, 3, 2, 1],
            "c1": ["A", "B"] * 4,
            "c2": ["C", "C", "D", "D"] * 2,
        }
    )

    with pytest.raises(ComputeError, match="found multiple elements in the same group"):
        df.pivot(
            values=["x1", "x2"],
            index="c1",
            columns="c2",
            separator="|",
            aggregate_function=None,
        )

    result = df.pivot(
        values=["x1", "x2"],
        index="c1",
        columns="c2",
        separator="|",
        aggregate_function="first",
    )
    expected = {
        "c1": ["A", "B"],
        "x1|c2|C": [1, 2],
        "x1|c2|D": [3, 4],
        "x2|c2|C": [8, 7],
        "x2|c2|D": [6, 5],
    }
    assert result.to_dict(as_series=False) == expected


def test_pivot_duplicate_names_7731() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 4],
            "b": [1.5, 2.5],
            "c": ["x", "x"],
            "d": [7, 8],
            "e": ["x", "y"],
        }
    )
    assert df.pivot(
        values=cs.integer(),
        index=cs.float(),
        columns=cs.string(),
        aggregate_function="first",
    ).to_dict(as_series=False) == {
        "b": [1.5, 2.5],
        "a_c_x": [1, 4],
        "d_c_x": [7, 8],
        "a_e_x": [1, None],
        "a_e_y": [None, 4],
        "d_e_x": [7, None],
        "d_e_y": [None, 8],
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

    with pytest.raises(ComputeError, match="found multiple elements in the same group"):
        result = df.pivot(
            values="price", index="weight", columns="quantity", aggregate_function=None
        )

    result = df.pivot(
        values="price", index="weight", columns="quantity", aggregate_function="first"
    )
    expected = {
        "weight": [1.0, 4.4, 8.8],
        "1.0": [1.0, 3.0, 5.0],
        "5.0": [2.0, None, None],
        "7.5": [6.0, None, None],
    }
    assert result.to_dict(as_series=False) == expected

    result = df.pivot(
        values="price",
        index=["article", "weight"],
        columns="quantity",
        aggregate_function=None,
    )
    expected = {
        "article": ["a", "a", "b", "b"],
        "weight": [1.0, 4.4, 1.0, 8.8],
        "1.0": [1.0, 3.0, 4.0, 5.0],
        "5.0": [2.0, None, None, None],
        "7.5": [None, None, 6.0, None],
    }
    assert result.to_dict(as_series=False) == expected


def test_pivot_reinterpret_5907() -> None:
    df = pl.DataFrame(
        {
            "A": pl.Series([3, -2, 3, -2], dtype=pl.Int32),
            "B": ["x", "x", "y", "y"],
            "C": [100, 50, 500, -80],
        }
    )

    result = df.pivot(
        index=["A"], values=["C"], columns=["B"], aggregate_function=pl.element().sum()
    )
    expected = {"A": [3, -2], "x": [100, 50], "y": [500, -80]}
    assert result.to_dict(as_series=False) == expected


def test_pivot_subclassed_df() -> None:
    class SubClassedDataFrame(pl.DataFrame):
        pass

    df = SubClassedDataFrame({"a": [1, 2], "b": [3, 4]})
    result = df.pivot(values="b", index="a", columns="a", aggregate_function="first")
    assert isinstance(result, SubClassedDataFrame)


def test_pivot_temporal_logical_types() -> None:
    date_lst = [datetime(_, 1, 1) for _ in range(1960, 1980)]

    df = pl.DataFrame(
        {
            "idx": date_lst[-3:] + date_lst[0:5],
            "foo": ["a"] * 3 + ["b"] * 5,
            "value": [0] * 8,
        }
    )
    assert df.pivot(
        index="idx", columns="foo", values="value", aggregate_function=None
    ).to_dict(as_series=False) == {
        "idx": [
            datetime(1977, 1, 1, 0, 0),
            datetime(1978, 1, 1, 0, 0),
            datetime(1979, 1, 1, 0, 0),
            datetime(1960, 1, 1, 0, 0),
            datetime(1961, 1, 1, 0, 0),
            datetime(1962, 1, 1, 0, 0),
            datetime(1963, 1, 1, 0, 0),
            datetime(1964, 1, 1, 0, 0),
        ],
        "a": [0, 0, 0, None, None, None, None, None],
        "b": [None, None, None, 0, 0, 0, 0, 0],
    }


def test_pivot_negative_duration() -> None:
    df1 = pl.DataFrame({"root": [date(2020, i, 15) for i in (1, 2)]})
    df2 = pl.DataFrame({"delta": [timedelta(days=i) for i in (-2, -1, 0, 1)]})

    df = df1.join(df2, how="cross").with_columns(
        [pl.Series(name="value", values=range(len(df1) * len(df2)))]
    )
    assert df.pivot(
        index="delta", columns="root", values="value", aggregate_function=None
    ).to_dict(as_series=False) == {
        "delta": [
            timedelta(days=-2),
            timedelta(days=-1),
            timedelta(0),
            timedelta(days=1),
        ],
        "2020-01-15": [0, 1, 2, 3],
        "2020-02-15": [4, 5, 6, 7],
    }


def test_aggregate_function_deprecation_warning() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": ["foo", "foo"], "c": ["x", "x"]})
    with pytest.raises(
        pl.ComputeError, match="found multiple elements in the same group"
    ):
        df.pivot("a", "b", "c")


def test_pivot_struct() -> None:
    data = {
        "id": ["a", "a", "b", "c", "c", "c"],
        "week": ["1", "2", "3", "4", "3", "1"],
        "num1": [1, 3, 5, 4, 3, 6],
        "num2": [4, 5, 3, 4, 6, 6],
    }
    df = pl.DataFrame(data).with_columns(nums=pl.struct(["num1", "num2"]))

    assert df.pivot(
        values="nums", index="id", columns="week", aggregate_function="first"
    ).to_dict(as_series=False) == {
        "id": ["a", "b", "c"],
        "1": [
            {"num1": 1, "num2": 4},
            {"num1": None, "num2": None},
            {"num1": 6, "num2": 6},
        ],
        "2": [
            {"num1": 3, "num2": 5},
            {"num1": None, "num2": None},
            {"num1": None, "num2": None},
        ],
        "3": [
            {"num1": None, "num2": None},
            {"num1": 5, "num2": 3},
            {"num1": 3, "num2": 6},
        ],
        "4": [
            {"num1": None, "num2": None},
            {"num1": None, "num2": None},
            {"num1": 4, "num2": 4},
        ],
    }
