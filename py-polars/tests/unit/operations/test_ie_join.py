from __future__ import annotations

from typing import TYPE_CHECKING, Any

import hypothesis.strategies as st
import numpy as np
from hypothesis import given

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from hypothesis.strategies import DrawFn, SearchStrategy


def test_self_join() -> None:
    west = pl.DataFrame(
        {
            "t_id": [404, 498, 676, 742],
            "time": [100, 140, 80, 90],
            "cost": [6, 11, 10, 5],
            "cores": [4, 2, 1, 4],
        }
    )

    actual = west.ie_join(
        west, on=[pl.col("time") > pl.col("time"), pl.col("cost") < pl.col("cost")]
    )

    expected = pl.DataFrame(
        {
            "t_id": [742, 404],
            "time": [90, 100],
            "cost": [5, 6],
            "cores": [4, 4],
            "t_id_right": [676, 676],
            "time_right": [80, 80],
            "cost_right": [10, 10],
            "cores_right": [1, 1],
        }
    )
    assert_frame_equal(actual, expected, check_row_order=False, check_exact=True)


def test_basic_ie_join() -> None:
    east = pl.DataFrame(
        {
            "id": [100, 101, 102],
            "dur": [140, 100, 90],
            "rev": [12, 12, 5],
            "cores": [2, 8, 4],
        }
    )
    west = pl.DataFrame(
        {
            "t_id": [404, 498, 676, 742],
            "time": [100, 140, 80, 90],
            "cost": [6, 11, 10, 5],
            "cores": [4, 2, 1, 4],
        }
    )

    actual = east.ie_join(
        west, on=[pl.col("dur") < pl.col("time"), pl.col("rev") > pl.col("cost")]
    )

    expected = pl.DataFrame(
        {
            "id": [101],
            "dur": [100],
            "rev": [12],
            "cores": [8],
            "t_id": [498],
            "time": [140],
            "cost": [11],
            "cores_right": [2],
        }
    )
    assert_frame_equal(actual, expected, check_row_order=False, check_exact=True)


@given(
    offset=st.integers(-6, 5),
    length=st.integers(0, 6),
)
def test_ie_join_with_slice(offset: int, length: int) -> None:
    east = pl.DataFrame(
        {
            "id": [100, 101, 102],
            "dur": [120, 140, 160],
            "rev": [12, 14, 16],
            "cores": [2, 8, 4],
        }
    ).lazy()
    west = pl.DataFrame(
        {
            "t_id": [404, 498, 676, 742],
            "time": [90, 130, 150, 170],
            "cost": [9, 13, 15, 16],
            "cores": [4, 2, 1, 4],
        }
    ).lazy()

    actual = (
        east.ie_join(
            west, on=[pl.col("dur") < pl.col("time"), pl.col("rev") < pl.col("cost")]
        )
        .slice(offset, length)
        .collect()
    )

    expected_full = pl.DataFrame(
        {
            "id": [101, 101, 100, 100, 100],
            "dur": [140, 140, 120, 120, 120],
            "rev": [14, 14, 12, 12, 12],
            "cores": [8, 8, 2, 2, 2],
            "t_id": [676, 742, 498, 676, 742],
            "time": [150, 170, 130, 150, 170],
            "cost": [15, 16, 13, 15, 16],
            "cores_right": [1, 4, 2, 1, 4],
        }
    )
    expected = expected_full.slice(offset, length)

    assert_frame_equal(actual, expected)


def _inequality_expression(col1: str, op: str, col2: str) -> pl.Expr:
    if op == "<":
        return pl.col(col1) < pl.col(col2)
    elif op == "<=":
        return pl.col(col1) <= pl.col(col2)
    elif op == ">":
        return pl.col(col1) > pl.col(col2)
    elif op == ">=":
        return pl.col(col1) >= pl.col(col2)
    else:
        message = f"Invalid operator '{op}'"
        raise ValueError(message)


def operators() -> SearchStrategy[str]:
    valid_operators = ["<", "<=", ">", ">="]
    return st.sampled_from(valid_operators)


@st.composite
def east_df(
    draw: DrawFn, with_nulls: bool = False, use_floats: bool = False
) -> pl.DataFrame:
    height = draw(st.integers(min_value=0, max_value=20))

    if use_floats:
        dur_strategy: SearchStrategy[Any] = st.floats(allow_nan=True)
        rev_strategy: SearchStrategy[Any] = st.floats(allow_nan=True)
        dur_dtype: type[pl.DataType] = pl.Float32
        rev_dtype: type[pl.DataType] = pl.Float32
    else:
        dur_strategy = st.integers(min_value=100, max_value=105)
        rev_strategy = st.integers(min_value=9, max_value=13)
        dur_dtype = pl.Int64
        rev_dtype = pl.Int64

    if with_nulls:
        dur_strategy = dur_strategy | st.none()
        rev_strategy = rev_strategy | st.none()

    cores_strategy = st.integers(min_value=1, max_value=10)

    ids = np.arange(0, height)
    dur = draw(st.lists(dur_strategy, min_size=height, max_size=height))
    rev = draw(st.lists(rev_strategy, min_size=height, max_size=height))
    cores = draw(st.lists(cores_strategy, min_size=height, max_size=height))

    return pl.DataFrame(
        [
            pl.Series("id", ids, dtype=pl.Int64),
            pl.Series("dur", dur, dtype=dur_dtype),
            pl.Series("rev", rev, dtype=rev_dtype),
            pl.Series("cores", cores, dtype=pl.Int64),
        ]
    )


@st.composite
def west_df(
    draw: DrawFn, with_nulls: bool = False, use_floats: bool = False
) -> pl.DataFrame:
    height = draw(st.integers(min_value=0, max_value=20))

    if use_floats:
        time_strategy: SearchStrategy[Any] = st.floats(allow_nan=True)
        cost_strategy: SearchStrategy[Any] = st.floats(allow_nan=True)
        time_dtype: type[pl.DataType] = pl.Float32
        cost_dtype: type[pl.DataType] = pl.Float32
    else:
        time_strategy = st.integers(min_value=100, max_value=105)
        cost_strategy = st.integers(min_value=9, max_value=13)
        time_dtype = pl.Int64
        cost_dtype = pl.Int64

    if with_nulls:
        time_strategy = time_strategy | st.none()
        cost_strategy = cost_strategy | st.none()

    cores_strategy = st.integers(min_value=1, max_value=10)

    t_id = np.arange(100, 100 + height)
    time = draw(st.lists(time_strategy, min_size=height, max_size=height))
    cost = draw(st.lists(cost_strategy, min_size=height, max_size=height))
    cores = draw(st.lists(cores_strategy, min_size=height, max_size=height))

    return pl.DataFrame(
        [
            pl.Series("t_id", t_id, dtype=pl.Int64),
            pl.Series("time", time, dtype=time_dtype),
            pl.Series("cost", cost, dtype=cost_dtype),
            pl.Series("cores", cores, dtype=pl.Int64),
        ]
    )


@given(
    east=east_df(),
    west=west_df(),
    op1=operators(),
    op2=operators(),
)
def test_ie_join(east: pl.DataFrame, west: pl.DataFrame, op1: str, op2: str) -> None:
    expr0 = _inequality_expression("dur", op1, "time")
    expr1 = _inequality_expression("rev", op2, "cost")

    actual = east.ie_join(west, on=[expr0, expr1])

    expected = east.join(west, how="cross").filter(expr0 & expr1)
    assert_frame_equal(actual, expected, check_row_order=False, check_exact=True)


@given(
    east=east_df(with_nulls=True),
    west=west_df(with_nulls=True),
    op1=operators(),
    op2=operators(),
)
def test_ie_join_with_nulls(
    east: pl.DataFrame, west: pl.DataFrame, op1: str, op2: str
) -> None:
    expr0 = _inequality_expression("dur", op1, "time")
    expr1 = _inequality_expression("rev", op2, "cost")

    actual = east.ie_join(west, on=[expr0, expr1])

    expected = east.join(west, how="cross").filter(expr0 & expr1)
    assert_frame_equal(actual, expected, check_row_order=False, check_exact=True)


@given(
    east=east_df(use_floats=True),
    west=west_df(use_floats=True),
    op1=operators(),
    op2=operators(),
)
def test_ie_join_with_floats(
    east: pl.DataFrame, west: pl.DataFrame, op1: str, op2: str
) -> None:
    expr0 = _inequality_expression("dur", op1, "time")
    expr1 = _inequality_expression("rev", op2, "cost")

    actual = east.ie_join(west, on=[expr0, expr1])

    expected = east.join(west, how="cross").filter(expr0 & expr1)
    assert_frame_equal(actual, expected, check_row_order=False, check_exact=True)
