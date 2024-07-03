from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
import polars.selectors as cs
from polars.exceptions import ComputeError, DuplicateError
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars._typing import PivotAgg


def test_pivot() -> None:
    df = pl.DataFrame(
        {
            "foo": ["A", "A", "B", "B", "C"],
            "bar": ["k", "l", "m", "n", "o"],
            "N": [1, 2, 2, 4, 2],
        }
    )
    result = df.pivot("bar", values="N", aggregate_function=None)

    expected = pl.DataFrame(
        [
            ("A", 1, 2, None, None, None),
            ("B", None, None, 2, 4, None),
            ("C", None, None, None, None, 2),
        ],
        schema=["foo", "k", "l", "m", "n", "o"],
        orient="row",
    )
    assert_frame_equal(result, expected)


def test_pivot_no_values() -> None:
    df = pl.DataFrame(
        {
            "foo": ["A", "A", "B", "B", "C"],
            "bar": ["k", "l", "m", "n", "o"],
            "N1": [1, 2, 2, 4, 2],
            "N2": [1, 2, 2, 4, 2],
        }
    )
    result = df.pivot(on="bar", index="foo", aggregate_function=None)
    expected = pl.DataFrame(
        {
            "foo": ["A", "B", "C"],
            "N1_k": [1, None, None],
            "N1_l": [2, None, None],
            "N1_m": [None, 2, None],
            "N1_n": [None, 4, None],
            "N1_o": [None, None, 2],
            "N2_k": [1, None, None],
            "N2_l": [2, None, None],
            "N2_m": [None, 2, None],
            "N2_n": [None, 4, None],
            "N2_o": [None, None, 2],
        }
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
        index="a",
        on="a",
        values="b",
        aggregate_function="first",
        sort_columns=True,
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
        index="b", on="a", values="c", aggregate_function=agg_fn, sort_columns=True
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

    result = df.pivot(index=["A"], on="B", values="B", aggregate_function="len")
    expected = {"A": ["Fire", "Water"], "Car": [1, 2], "Ship": [1, None]}
    assert result.to_dict(as_series=False) == expected

    # test expression dispatch
    result = df.pivot(index=["A"], on="B", values="B", aggregate_function=pl.len())
    assert result.to_dict(as_series=False) == expected

    df = pl.DataFrame(
        {
            "A": ["Fire", "Water", "Water", "Fire"],
            "B": ["Car", "Car", "Car", "Ship"],
            "C": ["Paper", "Paper", "Paper", "Paper"],
        },
        schema=[("A", pl.Categorical), ("B", pl.Categorical), ("C", pl.Categorical)],
    )
    result = df.pivot(index=["A", "C"], on="B", values="B", aggregate_function="len")
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
            index="c1",
            on="c2",
            values=["x1", "x2"],
            separator="|",
            aggregate_function=None,
        )

    result = df.pivot(
        index="c1",
        on="c2",
        values=["x1", "x2"],
        separator="|",
        aggregate_function="first",
    )
    expected = {
        "c1": ["A", "B"],
        "x1|C": [1, 2],
        "x1|D": [3, 4],
        "x2|C": [8, 7],
        "x2|D": [6, 5],
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
    result = df.pivot(
        index=cs.float(),
        on=cs.string(),
        values=cs.integer(),
        aggregate_function="first",
    ).to_dict(as_series=False)
    expected = {
        "b": [1.5, 2.5],
        'a_{"x","x"}': [1, None],
        'a_{"x","y"}': [None, 4],
        'd_{"x","x"}': [7, None],
        'd_{"x","y"}': [None, 8],
    }
    assert result == expected


def test_pivot_duplicate_names_11663() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [1, 2], "c": ["x", "x"], "d": ["x", "y"]})
    result = df.pivot(index="b", on=["c", "d"], values="a").to_dict(as_series=False)
    expected = {"b": [1, 2], '{"x","x"}': [1, None], '{"x","y"}': [None, 2]}
    assert result == expected


def test_pivot_multiple_columns_12407() -> None:
    df = pl.DataFrame(
        {
            "a": ["beep", "bop"],
            "b": ["a", "b"],
            "c": ["s", "f"],
            "d": [7, 8],
            "e": ["x", "y"],
        }
    )
    result = df.pivot(
        index="b", on=["c", "e"], values=["a"], aggregate_function="len"
    ).to_dict(as_series=False)
    expected = {"b": ["a", "b"], '{"s","x"}': [1, None], '{"f","y"}': [None, 1]}
    assert result == expected


def test_pivot_struct_13120() -> None:
    df = pl.DataFrame(
        {
            "index": [1, 2, 3, 1, 2, 3],
            "item_type": ["a", "a", "a", "b", "b", "b"],
            "item_id": [123, 123, 123, 456, 456, 456],
            "values": [4, 5, 6, 7, 8, 9],
        }
    )
    df = df.with_columns(pl.struct(["item_type", "item_id"]).alias("columns")).drop(
        "item_type", "item_id"
    )
    result = df.pivot(index="index", on="columns", values="values").to_dict(
        as_series=False
    )
    expected = {"index": [1, 2, 3], '{"a",123}': [4, 5, 6], '{"b",456}': [7, 8, 9]}
    assert result == expected


def test_pivot_index_struct_14101() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 1],
            "b": [{"a": 1}, {"a": 1}, {"a": 2}],
            "c": ["x", "y", "y"],
            "d": [1, 1, 3],
        }
    )
    result = df.pivot(index="b", on="c", values="a")
    expected = pl.DataFrame({"b": [{"a": 1}, {"a": 2}], "x": [1, None], "y": [2, 1]})
    assert_frame_equal(result, expected)


def test_pivot_name_already_exists() -> None:
    # This should be extremely rare...but still, good to check it
    df = pl.DataFrame(
        {
            "a": ["a", "b"],
            "b": ["a", "b"],
            '{"a","b"}': [1, 2],
        }
    )
    with pytest.raises(ComputeError, match="already exists in the DataFrame"):
        df.pivot(
            values='{"a","b"}',
            index="a",
            on=["a", "b"],
            aggregate_function="first",
        )


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
            index="weight", on="quantity", values="price", aggregate_function=None
        )

    result = df.pivot(
        index="weight", on="quantity", values="price", aggregate_function="first"
    )
    expected = {
        "weight": [1.0, 4.4, 8.8],
        "1.0": [1.0, 3.0, 5.0],
        "5.0": [2.0, None, None],
        "7.5": [6.0, None, None],
    }
    assert result.to_dict(as_series=False) == expected

    result = df.pivot(
        index=["article", "weight"],
        on="quantity",
        values="price",
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
        index=["A"], on=["B"], values=["C"], aggregate_function=pl.element().sum()
    )
    expected = {"A": [3, -2], "x": [100, 50], "y": [500, -80]}
    assert result.to_dict(as_series=False) == expected


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
        index="idx", on="foo", values="value", aggregate_function=None
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
        pl.Series(name="value", values=range(len(df1) * len(df2)))
    )
    assert df.pivot(
        index="delta", on="root", values="value", aggregate_function=None
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


def test_aggregate_function_default() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": ["foo", "foo"], "c": ["x", "x"]})
    with pytest.raises(ComputeError, match="found multiple elements in the same group"):
        df.pivot(index="b", on="c", values="a")


def test_pivot_aggregate_function_count_deprecated() -> None:
    df = pl.DataFrame(
        {
            "foo": ["A", "A", "B", "B", "C"],
            "N": [1, 2, 2, 4, 2],
            "bar": ["k", "l", "m", "n", "o"],
        }
    )
    with pytest.deprecated_call():
        df.pivot(index="foo", on="bar", values="N", aggregate_function="count")  # type: ignore[arg-type]


def test_pivot_struct() -> None:
    data = {
        "id": ["a", "a", "b", "c", "c", "c"],
        "week": ["1", "2", "3", "4", "3", "1"],
        "num1": [1, 3, 5, 4, 3, 6],
        "num2": [4, 5, 3, 4, 6, 6],
    }
    df = pl.DataFrame(data).with_columns(nums=pl.struct(["num1", "num2"]))

    assert df.pivot(
        values="nums", index="id", on="week", aggregate_function="first"
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


def test_duplicate_column_names_which_should_raise_14305() -> None:
    df = pl.DataFrame({"a": [1, 3, 2], "c": ["a", "a", "a"], "d": [7, 8, 9]})
    with pytest.raises(DuplicateError, match="has more than one occurrences"):
        df.pivot(index="a", on="c", values="d")


def test_multi_index_containing_struct() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 1],
            "b": [{"a": 1}, {"a": 1}, {"a": 2}],
            "c": ["x", "y", "y"],
            "d": [1, 1, 3],
        }
    )
    result = df.pivot(index=("b", "d"), on="c", values="a")
    expected = pl.DataFrame(
        {"b": [{"a": 1}, {"a": 2}], "d": [1, 3], "x": [1, None], "y": [2, 1]}
    )
    assert_frame_equal(result, expected)


def test_list_pivot() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 1],
            "b": [[1, 2], [3, 4], [5, 6], [1, 2]],
            "c": ["x", "x", "y", "y"],
            "d": [1, 2, 3, 4],
        }
    )
    assert df.pivot(
        index=["a", "b"],
        on="c",
        values="d",
    ).to_dict(as_series=False) == {
        "a": [1, 2, 3],
        "b": [[1, 2], [3, 4], [5, 6]],
        "x": [1, 2, None],
        "y": [4, None, 3],
    }


def test_pivot_string_17081() -> None:
    df = pl.DataFrame(
        {
            "a": ["1", "2", "3"],
            "b": ["4", "5", "6"],
            "c": ["7", "8", "9"],
        }
    )
    assert df.pivot(index="a", on="b", values="c", aggregate_function="min").to_dict(
        as_series=False
    ) == {
        "a": ["1", "2", "3"],
        "4": ["7", None, None],
        "5": [None, "8", None],
        "6": [None, None, "9"],
    }


def test_pivot_invalid() -> None:
    with pytest.raises(
        pl.exceptions.InvalidOperationError,
        match="`index` and `values` cannot both be None in `pivot` operation",
    ):
        pl.DataFrame({"a": [1, 2], "b": [2, 3], "c": [3, 4]}).pivot("a")
