from __future__ import annotations

import math
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from contextlib import AbstractContextManager as ContextManager

    from polars._typing import PolarsDataType


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
            "a": pl.Series([None], dtype=pl.String),
            "b": pl.Series([None], dtype=pl.Int64),
            "c": pl.Series([None], dtype=pl.Boolean),
        }
    )
    df2 = pl.DataFrame(
        {
            "a": pl.Series([None], dtype=pl.String),
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


def test_struct_equality_18870() -> None:
    s = pl.Series([{"a": 1}, None])

    # eq
    result = s.eq(s).to_list()
    expected = [True, None]
    assert result == expected

    # ne
    result = s.ne(s).to_list()
    expected = [False, None]
    assert result == expected

    # eq_missing
    result = s.eq_missing(s).to_list()
    expected = [True, True]
    assert result == expected

    # ne_missing
    result = s.ne_missing(s).to_list()
    expected = [False, False]
    assert result == expected


def test_struct_nested_equality() -> None:
    df = pl.DataFrame(
        {
            "a": [{"foo": 0, "bar": "1"}, {"foo": None, "bar": "1"}, None],
            "b": [{"foo": 0, "bar": "1"}] * 3,
        }
    )

    # eq
    ans = df.select(pl.col("a").eq(pl.col("b")))
    expected = pl.DataFrame({"a": [True, False, None]})
    assert_frame_equal(ans, expected)

    # ne
    ans = df.select(pl.col("a").ne(pl.col("b")))
    expected = pl.DataFrame({"a": [False, True, None]})
    assert_frame_equal(ans, expected)


def isnan(x: Any) -> bool:
    return isinstance(x, float) and math.isnan(x)


def reference_ordering_propagating(lhs: Any, rhs: Any) -> str | None:
    # normal < nan, nan == nan, nulls propagate
    if lhs is None or rhs is None:
        return None

    if isnan(lhs) and isnan(rhs):
        return "="

    if isnan(lhs) or lhs > rhs:
        return ">"

    if isnan(rhs) or lhs < rhs:
        return "<"

    return "="


def reference_ordering_missing(lhs: Any, rhs: Any) -> str:
    # null < normal < nan, nan == nan, null == null
    if lhs is None and rhs is None:
        return "="

    if lhs is None:
        return "<"

    if rhs is None:
        return ">"

    if isnan(lhs) and isnan(rhs):
        return "="

    if isnan(lhs) or lhs > rhs:
        return ">"

    if isnan(rhs) or lhs < rhs:
        return "<"

    return "="


def verify_total_ordering(
    lhs: Any, rhs: Any, dummy: Any, ldtype: PolarsDataType, rdtype: PolarsDataType
) -> None:
    ref = reference_ordering_propagating(lhs, rhs)
    refmiss = reference_ordering_missing(lhs, rhs)

    # Add dummy variable so we don't broadcast or do full-null optimization.
    assert dummy is not None
    df = pl.DataFrame(
        {"l": [lhs, dummy], "r": [rhs, dummy]}, schema={"l": ldtype, "r": rdtype}
    )

    ans = df.select(
        (pl.col("l") == pl.col("r")).alias("eq"),
        (pl.col("l") != pl.col("r")).alias("ne"),
        (pl.col("l") < pl.col("r")).alias("lt"),
        (pl.col("l") <= pl.col("r")).alias("le"),
        (pl.col("l") > pl.col("r")).alias("gt"),
        (pl.col("l") >= pl.col("r")).alias("ge"),
        pl.col("l").eq_missing(pl.col("r")).alias("eq_missing"),
        pl.col("l").ne_missing(pl.col("r")).alias("ne_missing"),
    )

    ans_correct_dict = {
        "eq": [ref and ref == "="],  # "ref and X" propagates ref is None
        "ne": [ref and ref != "="],
        "lt": [ref and ref == "<"],
        "le": [ref and (ref == "<" or ref == "=")],
        "gt": [ref and ref == ">"],
        "ge": [ref and (ref == ">" or ref == "=")],
        "eq_missing": [refmiss == "="],
        "ne_missing": [refmiss != "="],
    }
    ans_correct = pl.DataFrame(
        ans_correct_dict, schema=dict.fromkeys(ans_correct_dict, pl.Boolean)
    )

    assert_frame_equal(ans[:1], ans_correct)


def verify_total_ordering_broadcast(
    lhs: Any, rhs: Any, dummy: Any, ldtype: PolarsDataType, rdtype: PolarsDataType
) -> None:
    ref = reference_ordering_propagating(lhs, rhs)
    refmiss = reference_ordering_missing(lhs, rhs)

    # Add dummy variable so we don't broadcast inherently.
    assert dummy is not None
    df = pl.DataFrame(
        {"l": [lhs, dummy], "r": [rhs, dummy]}, schema={"l": ldtype, "r": rdtype}
    )

    ans_first = df.select(
        (pl.col("l") == pl.col("r").first()).alias("eq"),
        (pl.col("l") != pl.col("r").first()).alias("ne"),
        (pl.col("l") < pl.col("r").first()).alias("lt"),
        (pl.col("l") <= pl.col("r").first()).alias("le"),
        (pl.col("l") > pl.col("r").first()).alias("gt"),
        (pl.col("l") >= pl.col("r").first()).alias("ge"),
        pl.col("l").eq_missing(pl.col("r").first()).alias("eq_missing"),
        pl.col("l").ne_missing(pl.col("r").first()).alias("ne_missing"),
    )

    ans_scalar = df.select(
        (pl.col("l") == rhs).alias("eq"),
        (pl.col("l") != rhs).alias("ne"),
        (pl.col("l") < rhs).alias("lt"),
        (pl.col("l") <= rhs).alias("le"),
        (pl.col("l") > rhs).alias("gt"),
        (pl.col("l") >= rhs).alias("ge"),
        (pl.col("l").eq_missing(rhs)).alias("eq_missing"),
        (pl.col("l").ne_missing(rhs)).alias("ne_missing"),
    )

    ans_correct_dict = {
        "eq": [ref and ref == "="],  # "ref and X" propagates ref is None
        "ne": [ref and ref != "="],
        "lt": [ref and ref == "<"],
        "le": [ref and (ref == "<" or ref == "=")],
        "gt": [ref and ref == ">"],
        "ge": [ref and (ref == ">" or ref == "=")],
        "eq_missing": [refmiss == "="],
        "ne_missing": [refmiss != "="],
    }
    ans_correct = pl.DataFrame(
        ans_correct_dict, schema=dict.fromkeys(ans_correct_dict, pl.Boolean)
    )

    assert_frame_equal(ans_first[:1], ans_correct)
    assert_frame_equal(ans_scalar[:1], ans_correct)


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


@pytest.mark.slow
@pytest.mark.parametrize("lhs", INTERESTING_FLOAT_VALUES)
@pytest.mark.parametrize("rhs", INTERESTING_FLOAT_VALUES)
def test_total_ordering_float_series(lhs: float | None, rhs: float | None) -> None:
    verify_total_ordering(lhs, rhs, 0.0, pl.Float32, pl.Float32)
    verify_total_ordering(lhs, rhs, 0.0, pl.Float64, pl.Float32)
    context: pytest.WarningsRecorder | ContextManager[None] = (
        pytest.warns(UserWarning) if rhs is None else nullcontext()
    )
    with context:
        verify_total_ordering_broadcast(lhs, rhs, 0.0, pl.Float32, pl.Float32)
        verify_total_ordering_broadcast(lhs, rhs, 0.0, pl.Float64, pl.Float32)


INTERESTING_STRING_VALUES = [
    "",
    "foo",
    "bar",
    "fooo",
    "fooooooooooo",
    "foooooooooooo",
    "fooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooom",
    "foooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo",
    "fooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooop",
    None,
]


@pytest.mark.slow
@pytest.mark.parametrize("lhs", INTERESTING_STRING_VALUES)
@pytest.mark.parametrize("rhs", INTERESTING_STRING_VALUES)
def test_total_ordering_string_series(lhs: str | None, rhs: str | None) -> None:
    verify_total_ordering(lhs, rhs, "", pl.String, pl.String)
    context: pytest.WarningsRecorder | ContextManager[None] = (
        pytest.warns(UserWarning) if rhs is None else nullcontext()
    )
    with context:
        verify_total_ordering_broadcast(lhs, rhs, "", pl.String, pl.String)


@pytest.mark.slow
@pytest.mark.parametrize("lhs", INTERESTING_STRING_VALUES)
@pytest.mark.parametrize("rhs", INTERESTING_STRING_VALUES)
@pytest.mark.parametrize("fresh_cat", [False, True])
def test_total_ordering_cat_series(
    lhs: str | None, rhs: str | None, fresh_cat: bool
) -> None:
    if fresh_cat:
        c = [pl.Categorical(pl.Categories.random()) for _ in range(6)]
    else:
        c = [pl.Categorical() for _ in range(6)]
    verify_total_ordering(lhs, rhs, "", c[0], c[0])
    verify_total_ordering(lhs, rhs, "", pl.String, c[1])
    verify_total_ordering(lhs, rhs, "", c[2], pl.String)
    context: pytest.WarningsRecorder | ContextManager[None] = (
        pytest.warns(UserWarning) if rhs is None else nullcontext()
    )
    with context:
        verify_total_ordering_broadcast(lhs, rhs, "", c[3], c[3])
        verify_total_ordering_broadcast(lhs, rhs, "", pl.String, c[4])
        verify_total_ordering_broadcast(lhs, rhs, "", c[5], pl.String)


@pytest.mark.slow
@pytest.mark.parametrize("str_lhs", INTERESTING_STRING_VALUES)
@pytest.mark.parametrize("str_rhs", INTERESTING_STRING_VALUES)
def test_total_ordering_binary_series(str_lhs: str | None, str_rhs: str | None) -> None:
    lhs = None if str_lhs is None else str_lhs.encode("utf-8")
    rhs = None if str_rhs is None else str_rhs.encode("utf-8")
    verify_total_ordering(lhs, rhs, b"", pl.Binary, pl.Binary)
    context: pytest.WarningsRecorder | ContextManager[None] = (
        pytest.warns(UserWarning) if rhs is None else nullcontext()
    )
    with context:
        verify_total_ordering_broadcast(lhs, rhs, b"", pl.Binary, pl.Binary)


@pytest.mark.parametrize("lhs", [None, False, True])
@pytest.mark.parametrize("rhs", [None, False, True])
def test_total_ordering_bool_series(lhs: bool | None, rhs: bool | None) -> None:
    verify_total_ordering(lhs, rhs, False, pl.Boolean, pl.Boolean)
    context: pytest.WarningsRecorder | ContextManager[None] = (
        pytest.warns(UserWarning) if rhs is None else nullcontext()
    )
    with context:
        verify_total_ordering_broadcast(lhs, rhs, False, pl.Boolean, pl.Boolean)


def test_cat_compare_with_bool() -> None:
    data = pl.DataFrame([pl.Series("col1", ["a", "b"], dtype=pl.Categorical)])

    with pytest.raises(ComputeError, match="cannot compare categorical with bool"):
        data.filter(pl.col("col1") == True)  # noqa: E712


def test_schema_ne_missing_9256() -> None:
    df = pl.DataFrame({"a": [0, 1, None], "b": [True, False, True]})

    assert df.select(pl.col("a").ne_missing(0).or_(pl.col("b")))["a"].all()


def test_nested_binary_literal_super_type_12227() -> None:
    # The `.alias` is important here to trigger the bug.
    result = pl.select(x=1).select((pl.lit(0) + ((pl.col("x") > 0) * 0.1)).alias("x"))
    assert result.item() == 0.1

    result = pl.select((pl.lit(0) + (pl.lit(0) == pl.lit(0)) * pl.lit(0.1)) + pl.lit(0))
    assert result.item() == 0.1


def test_struct_broadcasting_comparison() -> None:
    df = pl.DataFrame({"foo": [{"a": 1}, {"a": 2}, {"a": 1}]})
    assert df.select(eq=pl.col.foo == pl.col.foo.last()).to_dict(as_series=False) == {
        "eq": [True, False, True]
    }


@pytest.mark.parametrize("dtype", [pl.List(pl.Int64), pl.Array(pl.Int64, 1)])
def test_compare_list_broadcast_empty_first_chunk_20165(dtype: pl.DataType) -> None:
    s = pl.concat(2 * [pl.Series([[1]], dtype=dtype)]).filter([False, True])

    assert s.len() == 1
    assert s.n_chunks() == 2

    assert_series_equal(
        pl.select(pl.lit(pl.Series([[1], [2]]), dtype=dtype) == pl.lit(s)).to_series(),
        pl.Series([True, False]),
    )
