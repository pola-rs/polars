from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

import polars as pl
from polars.testing import assert_frame_equal
from polars.testing.parametric.strategies import series

if TYPE_CHECKING:
    from hypothesis.strategies import DrawFn, SearchStrategy


@pytest.mark.parametrize(
    ("pred_1", "pred_2"),
    [
        (pl.col("time") > pl.col("time_right"), pl.col("cost") < pl.col("cost_right")),
        (pl.col("time_right") < pl.col("time"), pl.col("cost_right") > pl.col("cost")),
    ],
)
def test_self_join(pred_1: pl.Expr, pred_2: pl.Expr) -> None:
    west = pl.DataFrame(
        {
            "t_id": [404, 498, 676, 742],
            "time": [100, 140, 80, 90],
            "cost": [6, 11, 10, 5],
            "cores": [4, 2, 1, 4],
        }
    )

    actual = west.join_where(west, pred_1, pred_2)

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

    actual = east.join_where(
        west,
        pl.col("dur") < pl.col("time"),
        pl.col("rev") > pl.col("cost"),
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
        east.join_where(
            west,
            pl.col("dur") < pl.col("time"),
            pl.col("rev") < pl.col("cost"),
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
    # The ordering of the result is arbitrary, so we can
    # only verify that each row of the slice is present in the full expected result.
    assert len(actual) == len(expected_full.slice(offset, length))

    expected_rows = set(expected_full.iter_rows())
    for row in actual.iter_rows():
        assert row in expected_rows, f"{row} not in expected rows"


def test_ie_join_with_expressions() -> None:
    east = pl.DataFrame(
        {
            "id": [100, 101, 102],
            "dur": [70, 50, 45],
            "rev": [12, 12, 5],
            "cores": [2, 8, 4],
        }
    )
    west = pl.DataFrame(
        {
            "t_id": [404, 498, 676, 742],
            "time": [100, 140, 80, 90],
            "cost": [12, 22, 20, 10],
            "cores": [4, 2, 1, 4],
        }
    )

    actual = east.join_where(
        west,
        (pl.col("dur") * 2) < pl.col("time"),
        pl.col("rev") > (pl.col("cost").cast(pl.Int32) // 2).cast(pl.Int64),
    )

    expected = pl.DataFrame(
        {
            "id": [101],
            "dur": [50],
            "rev": [12],
            "cores": [8],
            "t_id": [498],
            "time": [140],
            "cost": [22],
            "cores_right": [2],
        }
    )
    assert_frame_equal(actual, expected, check_row_order=False, check_exact=True)


@pytest.mark.parametrize(
    "range_constraint",
    [
        [
            # can write individual components
            pl.col("time") >= pl.col("start_time"),
            pl.col("time") < pl.col("end_time"),
        ],
        [
            # or a single `is_between` expression
            pl.col("time").is_between("start_time", "end_time", closed="left")
        ],
    ],
)
def test_join_where_predicates(range_constraint: list[pl.Expr]) -> None:
    left = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "group": [0, 0, 0, 1, 1, 1],
            "time": [
                datetime(2024, 8, 26, 15, 34, 30),
                datetime(2024, 8, 26, 15, 35, 30),
                datetime(2024, 8, 26, 15, 36, 30),
                datetime(2024, 8, 26, 15, 37, 30),
                datetime(2024, 8, 26, 15, 38, 0),
                datetime(2024, 8, 26, 15, 39, 0),
            ],
        }
    )
    right = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "group": [0, 1, 1],
            "start_time": [
                datetime(2024, 8, 26, 15, 34, 0),
                datetime(2024, 8, 26, 15, 35, 0),
                datetime(2024, 8, 26, 15, 38, 0),
            ],
            "end_time": [
                datetime(2024, 8, 26, 15, 36, 0),
                datetime(2024, 8, 26, 15, 37, 0),
                datetime(2024, 8, 26, 15, 39, 0),
            ],
        }
    )

    actual = left.join_where(right, *range_constraint).select("id", "id_right")

    expected = pl.DataFrame(
        {
            "id": [0, 1, 1, 2, 4],
            "id_right": [0, 0, 1, 1, 2],
        }
    )
    assert_frame_equal(actual, expected, check_row_order=False, check_exact=True)

    q = (
        left.lazy()
        .join_where(
            right.lazy(),
            pl.col("group_right") == pl.col("group"),
            *range_constraint,
        )
        .select("id", "id_right", "group")
        .sort("id")
    )

    explained = q.explain()
    assert "INNER JOIN" in explained
    assert "FILTER" in explained
    actual = q.collect()

    expected = (
        left.join(right, how="cross")
        .filter(pl.col("group") == pl.col("group_right"), *range_constraint)
        .select("id", "id_right", "group")
        .sort("id")
    )
    assert_frame_equal(actual, expected, check_exact=True)

    q = (
        left.lazy()
        .join_where(
            right.lazy(),
            pl.col("group") != pl.col("group_right"),
            *range_constraint,
        )
        .select("id", "id_right", "group")
        .sort("id")
    )

    explained = q.explain()
    assert "IEJOIN" in explained
    assert "FILTER" in explained
    actual = q.collect()

    expected = (
        left.join(right, how="cross")
        .filter(pl.col("group") != pl.col("group_right"), *range_constraint)
        .select("id", "id_right", "group")
        .sort("id")
    )
    assert_frame_equal(actual, expected, check_exact=True)

    q = (
        left.lazy()
        .join_where(
            right.lazy(),
            pl.col("group") != pl.col("group_right"),
        )
        .select("id", "group", "group_right")
        .sort("id")
        .select("group", "group_right")
    )

    explained = q.explain()
    assert "NESTED LOOP" in explained
    actual = q.collect()
    assert actual.to_dict(as_series=False) == {
        "group": [0, 0, 0, 0, 0, 0, 1, 1, 1],
        "group_right": [1, 1, 1, 1, 1, 1, 0, 0, 0],
    }


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

    actual = east.join_where(west, expr0 & expr1)

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

    actual = east.join_where(west, expr0 & expr1)

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

    actual = east.join_where(west, expr0, expr1)

    expected = east.join(west, how="cross").filter(expr0 & expr1)
    assert_frame_equal(actual, expected, check_row_order=False, check_exact=True)


def test_raise_on_ambiguous_name() -> None:
    df = pl.DataFrame({"id": [1, 2]})
    with pytest.raises(
        pl.exceptions.InvalidOperationError,
        match="'join_where' predicate only refers to columns from a single table",
    ):
        df.join_where(df, pl.col("id") >= pl.col("id"))


def test_raise_invalid_input_join_where() -> None:
    df = pl.DataFrame({"id": [1, 2]})
    with pytest.raises(
        pl.exceptions.InvalidOperationError,
        match="expected join keys/predicates",
    ):
        df.join_where(df)


def test_ie_join_use_keys_multiple() -> None:
    a = pl.LazyFrame({"a": [1, 2, 3], "x": [7, 2, 1]})
    b = pl.LazyFrame({"b": [2, 2, 2], "x": [7, 1, 3]})

    assert a.join_where(
        b,
        pl.col.a >= pl.col.b,
        pl.col.a <= pl.col.b,
    ).collect().sort("x_right").to_dict(as_series=False) == {
        "a": [2, 2, 2],
        "x": [2, 2, 2],
        "b": [2, 2, 2],
        "x_right": [1, 3, 7],
    }


@given(
    left=series(
        dtype=pl.Int64,
        strategy=st.integers(min_value=0, max_value=10) | st.none(),
        max_size=10,
    ),
    right=series(
        dtype=pl.Int64,
        strategy=st.integers(min_value=-10, max_value=10) | st.none(),
        max_size=10,
    ),
    op=operators(),
)
def test_single_inequality(left: pl.Series, right: pl.Series, op: str) -> None:
    expr = _inequality_expression("x", op, "y")

    left_df = pl.DataFrame(
        {
            "id": np.arange(len(left)),
            "x": left,
        }
    )
    right_df = pl.DataFrame(
        {
            "id": np.arange(len(right)),
            "y": right,
        }
    )

    actual = left_df.join_where(right_df, expr)

    expected = left_df.join(right_df, how="cross").filter(expr)
    assert_frame_equal(actual, expected, check_row_order=False, check_exact=True)


@given(
    offset=st.integers(-6, 5),
    length=st.integers(0, 6),
)
def test_single_inequality_with_slice(offset: int, length: int) -> None:
    left = pl.DataFrame(
        {
            "id": list(range(8)),
            "x": [0, 1, 1, 2, 3, 5, 5, 7],
        }
    )
    right = pl.DataFrame(
        {
            "id": list(range(6)),
            "y": [-1, 2, 4, 4, 6, 9],
        }
    )

    expr = pl.col("x") > pl.col("y")
    actual = left.join_where(right, expr).slice(offset, length)

    expected_full = left.join(right, how="cross").filter(expr)

    assert len(actual) == len(expected_full.slice(offset, length))

    expected_rows = set(expected_full.iter_rows())
    for row in actual.iter_rows():
        assert row in expected_rows, f"{row} not in expected rows"


def test_ie_join_projection_pd_19005() -> None:
    lf = pl.LazyFrame({"a": [1, 2], "b": [3, 4]}).with_row_index()
    q = (
        lf.join_where(
            lf,
            pl.col.index < pl.col.index_right,
            pl.col.index.cast(pl.Int64) + pl.col.a > pl.col.a_right,
        )
        .group_by(pl.col.index)
        .agg(pl.col.index_right)
    )

    out = q.collect()
    assert out.schema == pl.Schema(
        [("index", pl.get_index_type()), ("index_right", pl.List(pl.get_index_type()))]
    )
    assert out.shape == (0, 2)


def test_raise_invalid_predicate() -> None:
    left = pl.LazyFrame({"a": [1, 2]}).with_row_index()
    right = pl.LazyFrame({"b": [1, 2]}).with_row_index()

    with pytest.raises(
        pl.exceptions.InvalidOperationError,
        match="'join_where' predicate only refers to columns from a single table",
    ):
        left.join_where(right, pl.col.index >= pl.col.a).collect()


def test_join_on_strings() -> None:
    df = pl.LazyFrame(
        {
            "a": ["a", "b", "c"],
            "b": ["b", "b", "b"],
        }
    )

    q = df.join_where(df, pl.col("a").ge(pl.col("a_right")))

    assert "NESTED LOOP JOIN" in q.explain()
    # Note: Output is flaky without sort when POLARS_MAX_THREADS=1
    assert q.collect().sort(pl.all()).to_dict(as_series=False) == {
        "a": ["a", "b", "b", "c", "c", "c"],
        "b": ["b", "b", "b", "b", "b", "b"],
        "a_right": ["a", "a", "b", "a", "b", "c"],
        "b_right": ["b", "b", "b", "b", "b", "b"],
    }


def test_join_partial_column_name_overlap_19119() -> None:
    left = pl.LazyFrame({"a": [1], "b": [2]})
    right = pl.LazyFrame({"a": [2], "d": [0]})

    q = left.join_where(right, pl.col("a") > pl.col("d"))

    assert q.collect().to_dict(as_series=False) == {
        "a": [1],
        "b": [2],
        "a_right": [2],
        "d": [0],
    }


def test_join_predicate_pushdown_19580() -> None:
    left = pl.LazyFrame(
        {
            "a": [1, 2, 3, 1],
            "b": [1, 2, 3, 4],
            "c": [2, 3, 4, 5],
        }
    )

    right = pl.LazyFrame({"a": [1, 3], "c": [2, 4], "d": [6, 3]})

    q = left.join_where(
        right,
        pl.col("b") < pl.col("c_right"),
        pl.col("a") < pl.col("a_right"),
        pl.col("a") < pl.col("d"),
    )

    expect = (
        left.join(right, how="cross")
        .collect()
        .filter(
            (pl.col("a") < pl.col("d"))
            & (pl.col("b") < pl.col("c_right"))
            & (pl.col("a") < pl.col("a_right"))
        )
    )

    assert_frame_equal(expect, q.collect(), check_row_order=False)


def test_join_where_literal_20061() -> None:
    df_left = pl.DataFrame(
        {"id": [1, 2, 3], "value_left": [10, 20, 30], "flag": [1, 0, 1]}
    )

    df_right = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "value_right": [5, 5, 25],
            "flag": [1, 0, 1],
        }
    )

    assert df_left.join_where(
        df_right,
        pl.col("value_left") > pl.col("value_right"),
        pl.col("flag_right").cast(pl.Int32) == 1,
    ).sort("id").to_dict(as_series=False) == {
        "id": [1, 2, 3, 3],
        "value_left": [10, 20, 30, 30],
        "flag": [1, 0, 1, 1],
        "id_right": [1, 1, 1, 3],
        "value_right": [5, 5, 5, 25],
        "flag_right": [1, 1, 1, 1],
    }


def test_boolean_predicate_join_where() -> None:
    urls = pl.LazyFrame({"url": "abcd.com/page"})
    categories = pl.LazyFrame({"base_url": "abcd.com", "category": "landing page"})
    assert (
        "NESTED LOOP JOIN"
        in urls.join_where(
            categories, pl.col("url").str.starts_with(pl.col("base_url"))
        ).explain()
    )
