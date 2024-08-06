from __future__ import annotations

from typing import TYPE_CHECKING

import hypothesis.strategies as st
import numpy as np
from hypothesis import given

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from hypothesis.strategies import DrawFn


def test_self_join() -> None:
    west = pl.DataFrame(
        {
            "t_id": [404, 498, 676, 742],
            "time": [100, 140, 80, 90],
            "cost": [6, 11, 10, 5],
            "cores": [4, 2, 1, 4],
        }
    )

    actual = west.ie_join(west, "time", ">", "time", "cost", "<", "cost")

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

    actual = east.ie_join(west, "dur", "<", "time", "rev", ">", "cost")

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


def _filter_expression(col1: str, op: str, col2: str) -> pl.Expr:
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


def operators() -> st.SearchStrategy[str]:
    valid_operators = ["<", "<=", ">", ">="]
    return st.sampled_from(valid_operators)


@st.composite
def east_df(draw: DrawFn) -> pl.DataFrame:
    height = draw(st.integers(min_value=0, max_value=20))

    dur_strategy = st.integers(min_value=100, max_value=105)
    rev_strategy = st.integers(min_value=9, max_value=13)
    cores_strategy = st.integers(min_value=1, max_value=10)

    ids = np.arange(0, height)
    dur = draw(st.lists(dur_strategy, min_size=height, max_size=height))
    rev = draw(st.lists(rev_strategy, min_size=height, max_size=height))
    cores = draw(st.lists(cores_strategy, min_size=height, max_size=height))

    return pl.DataFrame(
        [
            pl.Series("id", ids, dtype=pl.Int64),
            pl.Series("dur", dur, dtype=pl.Int64),
            pl.Series("rev", rev, dtype=pl.Int64),
            pl.Series("cores", cores, dtype=pl.Int64),
        ]
    )


@st.composite
def west_df(draw: DrawFn) -> pl.DataFrame:
    height = draw(st.integers(min_value=0, max_value=20))

    time_strategy = st.integers(min_value=100, max_value=105)
    cost_strategy = st.integers(min_value=9, max_value=13)
    cores_strategy = st.integers(min_value=1, max_value=10)

    t_id = np.arange(100, 100 + height)
    time = draw(st.lists(time_strategy, min_size=height, max_size=height))
    cost = draw(st.lists(cost_strategy, min_size=height, max_size=height))
    cores = draw(st.lists(cores_strategy, min_size=height, max_size=height))

    return pl.DataFrame(
        [
            pl.Series("t_id", t_id, dtype=pl.Int64),
            pl.Series("time", time, dtype=pl.Int64),
            pl.Series("cost", cost, dtype=pl.Int64),
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
    actual = east.ie_join(west, "dur", op1, "time", "rev", op2, "cost")

    expected = east.join(west, how="cross").filter(
        _filter_expression("dur", op1, "time") & _filter_expression("rev", op2, "cost")
    )
    assert_frame_equal(actual, expected, check_row_order=False, check_exact=True)
