from __future__ import annotations

import datetime
from typing import Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_any_expr(fruits_cars: pl.DataFrame) -> None:
    assert fruits_cars.select(pl.any_horizontal("A", "B")).to_series()[0] is True


def test_all_any_horizontally() -> None:
    df = pl.DataFrame(
        [
            [False, False, True],
            [False, False, True],
            [True, False, False],
            [False, None, True],
            [None, None, False],
        ],
        schema=["var1", "var2", "var3"],
    )
    result = df.select(
        any=pl.any_horizontal(pl.col("var2"), pl.col("var3")),
        all=pl.all_horizontal(pl.col("var2"), pl.col("var3")),
    )
    expected = pl.DataFrame(
        {
            "any": [True, True, False, True, None],
            "all": [False, False, False, None, False],
        }
    )
    assert_frame_equal(result, expected)

    # note: a kwargs filter will use an internal call to all_horizontal
    dfltr = df.lazy().filter(var1=None, var3=False)
    assert dfltr.collect().rows() == [(None, None, False)]

    # confirm that we reduce the horizontal filter components
    # (eg: explain does not contain an "all_horizontal" node)
    assert "horizontal" not in dfltr.explain().lower()


def test_all_any_accept_expr() -> None:
    lf = pl.LazyFrame(
        {
            "a": [1, None, 2, None],
            "b": [1, 2, None, None],
        }
    )

    result = lf.select(
        pl.any_horizontal(pl.all().is_null()).alias("null_in_row"),
        pl.all_horizontal(pl.all().is_null()).alias("all_null_in_row"),
    )

    expected = pl.LazyFrame(
        {
            "null_in_row": [False, True, True, True],
            "all_null_in_row": [False, False, False, True],
        }
    )
    assert_frame_equal(result, expected)


def test_max_min_multiple_columns(fruits_cars: pl.DataFrame) -> None:
    result = fruits_cars.select(max=pl.max_horizontal("A", "B"))
    expected = pl.Series("max", [5, 4, 3, 4, 5])
    assert_series_equal(result.to_series(), expected)

    result = fruits_cars.select(min=pl.min_horizontal("A", "B"))
    expected = pl.Series("min", [1, 2, 3, 2, 1])
    assert_series_equal(result.to_series(), expected)


def test_max_min_nulls_consistency() -> None:
    df = pl.DataFrame({"a": [None, 2, 3], "b": [4, None, 6], "c": [7, 5, 0]})

    result = df.select(max=pl.max_horizontal("a", "b", "c")).to_series()
    expected = pl.Series("max", [7, 5, 6])
    assert_series_equal(result, expected)

    result = df.select(min=pl.min_horizontal("a", "b", "c")).to_series()
    expected = pl.Series("min", [4, 2, 0])
    assert_series_equal(result, expected)


def test_nested_min_max() -> None:
    df = pl.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})

    result = df.with_columns(
        pl.max_horizontal(
            pl.min_horizontal("a", "b"), pl.min_horizontal("c", "d")
        ).alias("t")
    )

    expected = pl.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4], "t": [3]})
    assert_frame_equal(result, expected)


def test_empty_inputs_raise() -> None:
    with pytest.raises(
        pl.ComputeError,
        match="cannot return empty fold because the number of output rows is unknown",
    ):
        pl.select(pl.any_horizontal())

    with pytest.raises(
        pl.ComputeError,
        match="cannot return empty fold because the number of output rows is unknown",
    ):
        pl.select(pl.all_horizontal())


def test_max_min_wildcard_columns(fruits_cars: pl.DataFrame) -> None:
    result = fruits_cars.select(pl.col(pl.datatypes.Int64)).select(
        min=pl.min_horizontal("*")
    )
    expected = pl.Series("min", [1, 2, 3, 2, 1])
    assert_series_equal(result.to_series(), expected)

    result = fruits_cars.select(pl.col(pl.datatypes.Int64)).select(
        min=pl.min_horizontal(pl.all())
    )
    assert_series_equal(result.to_series(), expected)

    result = fruits_cars.select(pl.col(pl.datatypes.Int64)).select(
        max=pl.max_horizontal("*")
    )
    expected = pl.Series("max", [5, 4, 3, 4, 5])
    assert_series_equal(result.to_series(), expected)

    result = fruits_cars.select(pl.col(pl.datatypes.Int64)).select(
        max=pl.max_horizontal(pl.all())
    )
    assert_series_equal(result.to_series(), expected)

    result = fruits_cars.select(pl.col(pl.datatypes.Int64)).select(
        max=pl.max_horizontal(pl.all(), "A", "*")
    )
    assert_series_equal(result.to_series(), expected)


@pytest.mark.parametrize(
    ("input", "expected_data"),
    [
        (pl.col("^a|b$"), [1, 2]),
        (pl.col("a", "b"), [1, 2]),
        (pl.col("a"), [1, 4]),
        (pl.lit(5, dtype=pl.Int64), [5]),
        (5.0, [5.0]),
    ],
)
def test_min_horizontal_single_input(input: Any, expected_data: list[Any]) -> None:
    df = pl.DataFrame({"a": [1, 4], "b": [3, 2]})
    result = df.select(min=pl.min_horizontal(input)).to_series()
    expected = pl.Series("min", expected_data)
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("inputs", "expected_data"),
    [
        ((["a", "b"]), [1, 2]),
        (("a", "b"), [1, 2]),
        (("a", 3), [1, 3]),
    ],
)
def test_min_horizontal_multi_input(
    inputs: tuple[Any, ...], expected_data: list[Any]
) -> None:
    df = pl.DataFrame({"a": [1, 4], "b": [3, 2]})
    result = df.select(min=pl.min_horizontal(*inputs))
    expected = pl.DataFrame({"min": expected_data})
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("input", "expected_data"),
    [
        (pl.col("^a|b$"), [3, 4]),
        (pl.col("a", "b"), [3, 4]),
        (pl.col("a"), [1, 4]),
        (pl.lit(5, dtype=pl.Int64), [5]),
        (5.0, [5.0]),
    ],
)
def test_max_horizontal_single_input(input: Any, expected_data: list[Any]) -> None:
    df = pl.DataFrame({"a": [1, 4], "b": [3, 2]})
    result = df.select(max=pl.max_horizontal(input))
    expected = pl.DataFrame({"max": expected_data})
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("inputs", "expected_data"),
    [
        ((["a", "b"]), [3, 4]),
        (("a", "b"), [3, 4]),
        (("a", 3), [3, 4]),
    ],
)
def test_max_horizontal_multi_input(
    inputs: tuple[Any, ...], expected_data: list[Any]
) -> None:
    df = pl.DataFrame({"a": [1, 4], "b": [3, 2]})
    result = df.select(max=pl.max_horizontal(*inputs))
    expected = pl.DataFrame({"max": expected_data})
    assert_frame_equal(result, expected)


def test_expanding_sum() -> None:
    df = pl.DataFrame(
        {
            "x": [0, 1, 2],
            "y_1": [1.1, 2.2, 3.3],
            "y_2": [1.0, 2.5, 3.5],
        }
    )

    result = df.with_columns(pl.sum_horizontal(pl.col(r"^y_.*$")).alias("y_sum"))[
        "y_sum"
    ]
    assert result.to_list() == [2.1, 4.7, 6.8]


def test_sum_max_min() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    out = df.select(
        sum=pl.sum_horizontal("a", "b"),
        max=pl.max_horizontal("a", pl.col("b") ** 2),
        min=pl.min_horizontal("a", pl.col("b") ** 2),
    )
    assert_series_equal(out["sum"], pl.Series("sum", [2.0, 4.0, 6.0]))
    assert_series_equal(out["max"], pl.Series("max", [1.0, 4.0, 9.0]))
    assert_series_equal(out["min"], pl.Series("min", [1.0, 2.0, 3.0]))


def test_str_sum_horizontal() -> None:
    df = pl.DataFrame(
        {"A": ["a", "b", None, "c", None], "B": ["f", "g", "h", None, None]}
    )
    out = df.select(pl.sum_horizontal("A", "B"))
    assert_series_equal(out["A"], pl.Series("A", ["af", "bg", "h", "c", ""]))


def test_cum_sum_horizontal() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [3, 4],
            "c": [5, 6],
        }
    )
    result = df.select(pl.cum_sum_horizontal("a", "c"))
    expected = pl.DataFrame({"cum_sum": [{"a": 1, "c": 6}, {"a": 2, "c": 8}]})
    assert_frame_equal(result, expected)


def test_cumsum_horizontal_deprecated() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [3, 4],
            "c": [5, 6],
        }
    )
    with pytest.deprecated_call():
        result = df.select(pl.cumsum_horizontal("a", "c"))
    expected = df = pl.DataFrame({"cumsum": [{"a": 1, "c": 6}, {"a": 2, "c": 8}]})
    assert_frame_equal(result, expected)


def test_sum_dtype_12028() -> None:
    result = pl.select(
        pl.sum_horizontal([pl.duration(seconds=10)]).alias("sum_duration")
    )
    expected = pl.DataFrame(
        [
            pl.Series(
                "sum_duration",
                [datetime.timedelta(seconds=10)],
                dtype=pl.Duration(time_unit="us"),
            ),
        ]
    )
    assert_frame_equal(expected, result)


def test_horizontal_expr_use_left_name() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [3, 4],
        }
    )

    assert df.select(pl.sum_horizontal("a", "b")).columns == ["a"]
    assert df.select(pl.max_horizontal("*")).columns == ["a"]
    assert df.select(pl.min_horizontal("b", "a")).columns == ["b"]
    assert df.select(pl.any_horizontal("b", "a")).columns == ["b"]
    assert df.select(pl.all_horizontal("a", "b")).columns == ["a"]


def test_horizontal_broadcasting() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 3],
            "b": [3, 6],
        }
    )

    assert_series_equal(
        df.select(sum=pl.sum_horizontal(1, "a", "b")).to_series(),
        pl.Series("sum", [5, 10]),
    )
    assert_series_equal(
        df.select(max=pl.max_horizontal(4, "*")).to_series(), pl.Series("max", [4, 6])
    )
    assert_series_equal(
        df.select(min=pl.min_horizontal(2, "b", "a")).to_series(),
        pl.Series("min", [1, 2]),
    )
    assert_series_equal(
        df.select(any=pl.any_horizontal(False, pl.Series([True, False]))).to_series(),
        pl.Series("any", [True, False]),
    )
    assert_series_equal(
        df.select(all=pl.all_horizontal(True, pl.Series([True, False]))).to_series(),
        pl.Series("all", [True, False]),
    )
