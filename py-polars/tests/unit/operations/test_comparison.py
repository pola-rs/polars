from __future__ import annotations

import math
import warnings

import pytest

import polars as pl
from polars.testing import assert_frame_equal


def test_comparison_order_null_broadcasting() -> None:
    # see more: 8183
    exprs = [
        pl.col("v") < pl.col("null"),
        pl.col("null") < pl.col("v"),
        pl.col("v") <= pl.col("null"),
        pl.col("null") <= pl.col("v"),
        pl.col("v") > pl.col("null"),
        pl.col("null") > pl.col("v"),
        pl.col("v") >= pl.col("null"),
        pl.col("null") >= pl.col("v"),
    ]

    kwargs = {f"out{i}": e for i, e in zip(range(len(exprs)), exprs)}

    # single value, hits broadcasting branch
    df = pl.DataFrame({"v": [42], "null": [None]})
    assert all((df.select(**kwargs).null_count() == 1).rows()[0])

    # multiple values, hits default branch
    df = pl.DataFrame({"v": [42, 42], "null": [None, None]})
    assert all((df.select(**kwargs).null_count() == 2).rows()[0])


def test_comparison_nulls_single() -> None:
    df1 = pl.DataFrame(
        {
            "a": pl.Series([None], dtype=pl.Utf8),
            "b": pl.Series([None], dtype=pl.Int64),
            "c": pl.Series([None], dtype=pl.Boolean),
        }
    )
    df2 = pl.DataFrame(
        {
            "a": pl.Series([None], dtype=pl.Utf8),
            "b": pl.Series([None], dtype=pl.Int64),
            "c": pl.Series([None], dtype=pl.Boolean),
        }
    )
    assert (df1 == df2).row(0) == (None, None, None)
    assert (df1 != df2).row(0) == (None, None, None)


def test_comparison_series_expr() -> None:
    df = pl.DataFrame({"a": pl.Series([1, 2, 3]), "b": pl.Series([2, 1, 3])})

    assert_frame_equal(
        df.select(
            (df["a"] == pl.col("b")).alias("eq"),  # False, False, True
            (df["a"] != pl.col("b")).alias("ne"),  # True, True, False
            (df["a"] < pl.col("b")).alias("lt"),  # True, False, False
            (df["a"] <= pl.col("b")).alias("le"),  # True, False, True
            (df["a"] > pl.col("b")).alias("gt"),  # False, True, False
            (df["a"] >= pl.col("b")).alias("ge"),  # False, True, True
        ),
        pl.DataFrame(
            {
                "eq": [False, False, True],
                "ne": [True, True, False],
                "lt": [True, False, False],
                "le": [True, False, True],
                "gt": [False, True, False],
                "ge": [False, True, True],
            }
        ),
    )


def test_comparison_expr_expr() -> None:
    df = pl.DataFrame({"a": pl.Series([1, 2, 3]), "b": pl.Series([2, 1, 3])})

    assert_frame_equal(
        df.select(
            (pl.col("a") == pl.col("b")).alias("eq"),  # False, False, True
            (pl.col("a") != pl.col("b")).alias("ne"),  # True, True, False
            (pl.col("a") < pl.col("b")).alias("lt"),  # True, False, False
            (pl.col("a") <= pl.col("b")).alias("le"),  # True, False, True
            (pl.col("a") > pl.col("b")).alias("gt"),  # False, True, False
            (pl.col("a") >= pl.col("b")).alias("ge"),  # False, True, True
        ),
        pl.DataFrame(
            {
                "eq": [False, False, True],
                "ne": [True, True, False],
                "lt": [True, False, False],
                "le": [True, False, True],
                "gt": [False, True, False],
                "ge": [False, True, True],
            }
        ),
    )


def test_comparison_expr_series() -> None:
    df = pl.DataFrame({"a": pl.Series([1, 2, 3]), "b": pl.Series([2, 1, 3])})

    assert_frame_equal(
        df.select(
            (pl.col("a") == df["b"]).alias("eq"),  # False, False, True
            (pl.col("a") != df["b"]).alias("ne"),  # True, True, False
            (pl.col("a") < df["b"]).alias("lt"),  # True, False, False
            (pl.col("a") <= df["b"]).alias("le"),  # True, False, True
            (pl.col("a") > df["b"]).alias("gt"),  # False, True, False
            (pl.col("a") >= df["b"]).alias("ge"),  # False, True, True
        ),
        pl.DataFrame(
            {
                "eq": [False, False, True],
                "ne": [True, True, False],
                "lt": [True, False, False],
                "le": [True, False, True],
                "gt": [False, True, False],
                "ge": [False, True, True],
            }
        ),
    )


def test_offset_handling_arg_where_7863() -> None:
    df_check = pl.DataFrame({"a": [0, 1]})
    df_check.select((pl.lit(0).append(pl.col("a")).append(0)) != 0)
    assert (
        df_check.select((pl.lit(0).append(pl.col("a")).append(0)) != 0)
        .select(pl.col("literal").arg_true())
        .item()
        == 2
    )


def test_missing_equality_on_bools() -> None:
    df = pl.DataFrame(
        {
            "a": [True, None, False],
        }
    )

    assert df.select(pl.col("a").ne_missing(True))["a"].to_list() == [False, True, True]
    assert df.select(pl.col("a").ne_missing(False))["a"].to_list() == [
        True,
        True,
        False,
    ]


def reference_ordering_propagating(lhs: float | None, rhs: float | None) -> str | None:
    if lhs is None or rhs is None:
        return None

    if math.isnan(lhs) and math.isnan(rhs):
        return "="

    if math.isnan(lhs) or lhs > rhs:
        return ">"

    if math.isnan(rhs) or lhs < rhs:
        return "<"

    return "="


INTERESTING_FLOAT_VALUES = [
    0.0,
    -0.0,
    -1.0,
    1.0,
    -float("nan"),
    float("nan"),
    -float("inf"),
    float("inf"),
    None,
]


@pytest.mark.parametrize("lhs", INTERESTING_FLOAT_VALUES)
@pytest.mark.parametrize("rhs", INTERESTING_FLOAT_VALUES)
def test_total_ordering_float_series(lhs: float | None, rhs: float | None) -> None:
    ref = reference_ordering_propagating(lhs, rhs)

    # Add dummy variable so we don't broadcast or do full-null optimization.
    df = pl.DataFrame(
        {"l": [lhs, 0.0], "r": [rhs, 0.0]}, schema={"l": pl.Float64, "r": pl.Float64}
    )

    assert_frame_equal(
        df.select(
            (pl.col("l") == pl.col("r")).alias("eq"),
            (pl.col("l") != pl.col("r")).alias("ne"),
            (pl.col("l") < pl.col("r")).alias("lt"),
            (pl.col("l") <= pl.col("r")).alias("le"),
            (pl.col("l") > pl.col("r")).alias("gt"),
            (pl.col("l") >= pl.col("r")).alias("ge"),
        ),
        pl.DataFrame(
            {
                "eq": [ref and ref == "=", True],  # "ref and X" propagates ref is None
                "ne": [ref and ref != "=", False],
                "lt": [ref and ref == "<", False],
                "le": [ref and (ref == "<" or ref == "="), True],
                "gt": [ref and ref == ">", False],
                "ge": [ref and (ref == ">" or ref == "="), True],
            }
        ),
    )


@pytest.mark.parametrize("lhs", INTERESTING_FLOAT_VALUES)
@pytest.mark.parametrize("rhs", INTERESTING_FLOAT_VALUES)
def test_total_ordering_float_series_broadcast(
    lhs: float | None, rhs: float | None
) -> None:
    # We do want to test None comparisons.
    warnings.filterwarnings("ignore", category=UserWarning)

    ref = reference_ordering_propagating(lhs, rhs)

    # Add dummy variable so we don't broadcast inherently.
    df = pl.DataFrame(
        {"l": [lhs, lhs], "r": [rhs, rhs]}, schema={"l": pl.Float64, "r": pl.Float64}
    )

    ans_first = df.select(
        (pl.col("l") == pl.col("r").first()).alias("eq"),
        (pl.col("l") != pl.col("r").first()).alias("ne"),
        (pl.col("l") < pl.col("r").first()).alias("lt"),
        (pl.col("l") <= pl.col("r").first()).alias("le"),
        (pl.col("l") > pl.col("r").first()).alias("gt"),
        (pl.col("l") >= pl.col("r").first()).alias("ge"),
    )

    ans_scalar = df.select(
        (pl.col("l") == rhs).alias("eq"),
        (pl.col("l") != rhs).alias("ne"),
        (pl.col("l") < rhs).alias("lt"),
        (pl.col("l") <= rhs).alias("le"),
        (pl.col("l") > rhs).alias("gt"),
        (pl.col("l") >= rhs).alias("ge"),
    )

    ans_correct = pl.DataFrame(
        {
            "eq": [ref and ref == "="] * 2,  # "ref and X" propagates ref is None
            "ne": [ref and ref != "="] * 2,
            "lt": [ref and ref == "<"] * 2,
            "le": [ref and (ref == "<" or ref == "=")] * 2,
            "gt": [ref and ref == ">"] * 2,
            "ge": [ref and (ref == ">" or ref == "=")] * 2,
        },
        schema={c: pl.Boolean for c in ["eq", "ne", "lt", "le", "gt", "ge"]},
    )

    assert_frame_equal(ans_first, ans_correct)
    assert_frame_equal(ans_scalar, ans_correct)


def reference_ordering_missing(lhs: float | None, rhs: float | None) -> str:
    if lhs is None and rhs is None:
        return "="

    if lhs is None:
        return "<"

    if rhs is None:
        return ">"

    if math.isnan(lhs) and math.isnan(rhs):
        return "="

    if math.isnan(lhs) or lhs > rhs:
        return ">"

    if math.isnan(rhs) or lhs < rhs:
        return "<"

    return "="


@pytest.mark.parametrize("lhs", INTERESTING_FLOAT_VALUES)
@pytest.mark.parametrize("rhs", INTERESTING_FLOAT_VALUES)
def test_total_ordering_float_series_missing(
    lhs: float | None, rhs: float | None
) -> None:
    ref = reference_ordering_missing(lhs, rhs)

    # Add dummy variable so we don't broadcast or do full-null optimization.
    df = pl.DataFrame(
        {"l": [lhs, 0.0], "r": [rhs, 0.0]}, schema={"l": pl.Float64, "r": pl.Float64}
    )

    assert_frame_equal(
        df.select(
            pl.col("l").eq_missing(pl.col("r")).alias("eq"),
            pl.col("l").ne_missing(pl.col("r")).alias("ne"),
        ),
        pl.DataFrame(
            {
                "eq": [ref == "=", True],
                "ne": [ref != "=", False],
            }
        ),
    )


@pytest.mark.parametrize("lhs", INTERESTING_FLOAT_VALUES)
@pytest.mark.parametrize("rhs", INTERESTING_FLOAT_VALUES)
def test_total_ordering_float_series_missing_broadcast(
    lhs: float | None, rhs: float | None
) -> None:
    ref = reference_ordering_missing(lhs, rhs)

    # Add dummy variable so we don't broadcast inherently.
    df = pl.DataFrame(
        {"l": [lhs, lhs], "r": [rhs, rhs]}, schema={"l": pl.Float64, "r": pl.Float64}
    )

    ans_first = df.select(
        pl.col("l").eq_missing(pl.col("r").first()).alias("eq"),
        pl.col("l").ne_missing(pl.col("r").first()).alias("ne"),
    )

    ans_scalar = df.select(
        pl.col("l").eq_missing(rhs).alias("eq"),
        pl.col("l").ne_missing(rhs).alias("ne"),
    )

    ans_correct = pl.DataFrame(
        {
            "eq": [ref == "="] * 2,
            "ne": [ref != "="] * 2,
        },
        schema={c: pl.Boolean for c in ["eq", "ne"]},
    )

    assert_frame_equal(ans_first, ans_correct)
    assert_frame_equal(ans_scalar, ans_correct)
