from __future__ import annotations

import math
from contextlib import nullcontext
from datetime import datetime, timedelta
from itertools import combinations
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.datatypes.group import INTEGER_DTYPES
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

    kwargs = {f"out{i}": e for i, e in zip(range(len(exprs)), exprs, strict=True)}

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
        pytest.warns(
            UserWarning,
            match=r"Consider using `\.is_null\(\)` or `\.is_not_null\(\)`",
        )
        if rhs is None
        else nullcontext()
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
        pytest.warns(
            UserWarning,
            match=r"Consider using `\.is_null\(\)` or `\.is_not_null\(\)`",
        )
        if rhs is None
        else nullcontext()
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
        pytest.warns(
            UserWarning,
            match=r"Consider using `\.is_null\(\)` or `\.is_not_null\(\)`",
        )
        if rhs is None
        else nullcontext()
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
        pytest.warns(
            UserWarning,
            match=r"Consider using `\.is_null\(\)` or `\.is_not_null\(\)`",
        )
        if rhs is None
        else nullcontext()
    )
    with context:
        verify_total_ordering_broadcast(lhs, rhs, b"", pl.Binary, pl.Binary)


@pytest.mark.parametrize("lhs", [None, False, True])
@pytest.mark.parametrize("rhs", [None, False, True])
def test_total_ordering_bool_series(lhs: bool | None, rhs: bool | None) -> None:
    verify_total_ordering(lhs, rhs, False, pl.Boolean, pl.Boolean)
    context: pytest.WarningsRecorder | ContextManager[None] = (
        pytest.warns(
            UserWarning,
            match=r"Consider using `\.is_null\(\)` or `\.is_not_null\(\)`",
        )
        if rhs is None
        else nullcontext()
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


def test_date_duration_comparison_error_25517() -> None:
    date = pl.Series("date", [1], pl.Date)
    duration = pl.Series("duration", [1], pl.Duration("ns"))

    with pytest.raises(ComputeError, match="cannot compare date with duration"):
        _ = date > duration

    with pytest.raises(ComputeError, match="cannot compare date with duration"):
        _ = duration > date

    with pytest.raises(ComputeError, match="cannot compare date with duration"):
        _ = date == duration


@pytest.mark.parametrize(
    ("dtype_lhs", "dtype_rhs"),
    combinations(sorted(INTEGER_DTYPES, key=lambda v: str(v)), 2),
)
@pytest.mark.parametrize("swap", [True, False])
def test_comparison_literal_behavior_matches_nonliteral_behavior(
    dtype_lhs: PolarsDataType,
    dtype_rhs: PolarsDataType,
    swap: bool,
) -> None:
    if swap:
        dtype_lhs, dtype_rhs = dtype_rhs, dtype_lhs

    int_boundaries = [
        -(1 << 127),
        -(1 << 63),
        -(1 << 31),
        -(1 << 15),
        -(1 << 7),
        0,
        (1 << 7) - 1,
        (1 << 8) - 1,
        (1 << 15) - 1,
        (1 << 16) - 1,
        (1 << 31) - 1,
        (1 << 32) - 1,
        (1 << 63) - 1,
        (1 << 64) - 1,
        (1 << 127) - 1,
        (1 << 128) - 1,
        None,
    ]

    lhs_s = pl.Series("l", int_boundaries, dtype=dtype_lhs, strict=False)
    rhs_s = pl.Series("r", int_boundaries, dtype=dtype_rhs, strict=False)

    rmin: Any = rhs_s.min()
    rmax: Any = rhs_s.max()

    lmin: Any = lhs_s.min()
    lmax: Any = lhs_s.max()

    assert rmin is not None
    assert rmax is not None
    assert lmin is not None
    assert lmax is not None

    df = pl.DataFrame([lhs_s, rhs_s])

    def build_comparison_query(
        lf: pl.LazyFrame,
        l: pl.Expr,  # noqa: E741
        r: pl.Expr,
    ) -> pl.LazyFrame:
        return lf.select(
            **dict(ne=l != r, lt=l < r, lteq=l <= r, eq=l == r, gteq=l >= r, gt=l > r),  # noqa: C408
            **dict(  # noqa: C408
                rne=r != l, rlt=r < l, rlteq=r <= l, req=r == l, rgteq=r >= l, rgt=r > l
            ),
            eq_missing=l.eq_missing(r),
            ne_missing=l.ne_missing(r),
            req_missing=r.eq_missing(l),
            rne_missing=r.ne_missing(l),
        )

    def test(lit_val: pl.Expr, col_expr: pl.Expr) -> None:
        lf = df.lazy()
        q = build_comparison_query(lf, col_expr, lit_val)
        q_use_rcol = build_comparison_query(
            lf.with_columns(lit_val=lit_val), col_expr, pl.col("lit_val")
        )

        assert_frame_equal(q.collect(), q_use_rcol.collect())

    if rmax > lmax:
        test(pl.lit(lmax + 1, dtype=dtype_rhs), pl.col("l"))
        test(pl.lit(None, dtype=dtype_rhs), pl.col("l"))
    elif lmax > rmax:
        test(pl.lit(rmax + 1, dtype=dtype_lhs), pl.col("r"))
        test(pl.lit(None, dtype=dtype_lhs), pl.col("r"))

    if rmin < lmin:
        test(pl.lit(lmin - 1, dtype=dtype_rhs), pl.col("l"))
        test(pl.lit(None, dtype=dtype_rhs), pl.col("l"))
    elif lmin < rmin:
        test(pl.lit(rmin - 1, dtype=dtype_lhs), pl.col("r"))
        test(pl.lit(None, dtype=dtype_lhs), pl.col("r"))


def test_comparison_literal_downcast_flooring_datetime_ns() -> None:
    dt = datetime(2026, 1, 1)
    unit_phys_s = pl.Series("datetime[ns]", [dt], dtype=pl.Datetime("ns")).to_physical()

    adjust_s = pl.Series(
        [
            1_000,
            1_000_000,
            1_000_000_000,
            86_400_000_000_000,  # NS_IN_DAY
        ]
    )

    df = (
        pl.concat(
            [
                unit_phys_s - 1,
                unit_phys_s,
                unit_phys_s + (adjust_s - 1),
                unit_phys_s + adjust_s,
            ]
        )
        .sort()
        .cast(pl.Datetime("ns"))
        .to_frame()
        .with_row_index()
    )

    result_vs_col = df.with_columns(
        us=pl.lit(dt, dtype=pl.Datetime("us")),
        ms=pl.lit(dt, dtype=pl.Datetime("ms")),
        date=pl.lit(dt.date(), dtype=pl.Date),
    ).select(
        pl.col("datetime[ns]").cast(pl.String),
        eq_us=pl.col("datetime[ns]") == pl.col("us"),
        eq_ms=pl.col("datetime[ns]") == pl.col("ms"),
        eq_date=pl.col("datetime[ns]") == pl.col("date"),
    )

    F = False
    truth_table = pl.DataFrame(
        [
            ("2025-12-31 23:59:59.999999999", F, F, F),
            ("2026-01-01 00:00:00.000000000", True, True, True),
            ("2026-01-01 00:00:00.000000999", True, True, F),
            ("2026-01-01 00:00:00.000001000", F, True, F),
            ("2026-01-01 00:00:00.000999999", F, True, F),
            ("2026-01-01 00:00:00.001000000", F, F, F),
            ("2026-01-01 00:00:00.999999999", F, F, F),
            ("2026-01-01 00:00:01.000000000", F, F, F),
            ("2026-01-01 23:59:59.999999999", F, F, F),
            ("2026-01-02 00:00:00.000000000", F, F, F),
        ],
        orient="row",
        schema=result_vs_col.schema,
    )

    assert_frame_equal(result_vs_col, truth_table)

    # us -> ns downcast
    q = df.lazy().with_columns(
        eq=pl.col("datetime[ns]") == pl.lit(dt, dtype=pl.Datetime("us")),
        lteq=pl.col("datetime[ns]") <= pl.lit(dt, dtype=pl.Datetime("us")),
    )

    plan = q.explain()
    assert plan.count(".000000999") == 2

    assert_frame_equal(
        q.filter("eq")
        .select(pl.implode("index"), pl.last("datetime[ns]").cast(pl.String))
        .collect(),
        pl.DataFrame(
            {"index": [[1, 2]], "datetime[ns]": "2026-01-01 00:00:00.000000999"},
            schema={"index": pl.List(pl.get_index_type()), "datetime[ns]": pl.String},
        ),
    )

    assert_frame_equal(
        q.filter("lteq")
        .select(pl.implode("index"), pl.last("datetime[ns]").cast(pl.String))
        .collect(),
        pl.DataFrame(
            {"index": [[0, 1, 2]], "datetime[ns]": "2026-01-01 00:00:00.000000999"},
            schema={"index": pl.List(pl.get_index_type()), "datetime[ns]": pl.String},
        ),
    )

    # ms -> ns downcast
    q = df.lazy().with_columns(
        eq=pl.col("datetime[ns]") == pl.lit(dt, dtype=pl.Datetime("ms")),
        lteq=pl.col("datetime[ns]") <= pl.lit(dt, dtype=pl.Datetime("ms")),
    )

    plan = q.explain()
    assert plan.count(".000999999") == 2

    assert_frame_equal(
        q.filter("eq")
        .select(pl.implode("index"), pl.last("datetime[ns]").cast(pl.String))
        .collect(),
        pl.DataFrame(
            {"index": [[1, 2, 3, 4]], "datetime[ns]": "2026-01-01 00:00:00.000999999"},
            schema={"index": pl.List(pl.get_index_type()), "datetime[ns]": pl.String},
        ),
    )

    assert_frame_equal(
        q.filter("lteq")
        .select(pl.implode("index"), pl.last("datetime[ns]").cast(pl.String))
        .collect(),
        pl.DataFrame(
            {
                "index": [[0, 1, 2, 3, 4]],
                "datetime[ns]": "2026-01-01 00:00:00.000999999",
            },
            schema={"index": pl.List(pl.get_index_type()), "datetime[ns]": pl.String},
        ),
    )


def test_comparison_literal_downcast_flooring_datetime_us() -> None:
    dt = datetime(2026, 1, 1)
    unit_phys_s = pl.Series("datetime[us]", [dt], dtype=pl.Datetime("us")).to_physical()

    adjust_s = pl.Series(
        [
            1_000,
            1_000_000,
            86_400_000_000,  # US_IN_DAY
        ]
    )

    df = (
        pl.concat(
            [
                unit_phys_s - 1,
                unit_phys_s,
                unit_phys_s + (adjust_s - 1),
                unit_phys_s + adjust_s,
            ]
        )
        .sort()
        .cast(pl.Datetime("us"))
        .to_frame()
        .with_row_index()
    )

    result_vs_col = df.with_columns(
        ms=pl.lit(dt, dtype=pl.Datetime("ms")),
        date=pl.lit(dt.date(), dtype=pl.Date),
    ).select(
        pl.col("datetime[us]").cast(pl.String),
        eq_ms=pl.col("datetime[us]") == pl.col("ms"),
        eq_date=pl.col("datetime[us]") == pl.col("date"),
    )

    F = False
    truth_table = pl.DataFrame(
        [
            ("2025-12-31 23:59:59.999999", F, F),
            ("2026-01-01 00:00:00.000000", True, True),
            ("2026-01-01 00:00:00.000999", True, F),
            ("2026-01-01 00:00:00.001000", F, F),
            ("2026-01-01 00:00:00.999999", F, F),
            ("2026-01-01 00:00:01.000000", F, F),
            ("2026-01-01 23:59:59.999999", F, F),
            ("2026-01-02 00:00:00.000000", F, F),
        ],
        orient="row",
        schema=result_vs_col.schema,
    )

    assert_frame_equal(result_vs_col, truth_table)

    # ms -> us downcast
    q = df.lazy().with_columns(
        eq=pl.col("datetime[us]") == pl.lit(dt, dtype=pl.Datetime("ms")),
        lteq=pl.col("datetime[us]") <= pl.lit(dt, dtype=pl.Datetime("ms")),
    )

    plan = q.explain()
    assert plan.count(".000999") == 2

    assert_frame_equal(
        q.filter("eq")
        .select(pl.implode("index"), pl.last("datetime[us]").cast(pl.String))
        .collect(),
        pl.DataFrame(
            {"index": [[1, 2]], "datetime[us]": "2026-01-01 00:00:00.000999"},
            schema={"index": pl.List(pl.get_index_type()), "datetime[us]": pl.String},
        ),
    )

    assert_frame_equal(
        q.filter("lteq")
        .select(pl.implode("index"), pl.last("datetime[us]").cast(pl.String))
        .collect(),
        pl.DataFrame(
            {"index": [[0, 1, 2]], "datetime[us]": "2026-01-01 00:00:00.000999"},
            schema={"index": pl.List(pl.get_index_type()), "datetime[us]": pl.String},
        ),
    )


def test_comparison_literal_downcast_flooring_duration_ns() -> None:
    dt = timedelta()
    unit_phys_s = pl.Series("duration[ns]", [0], dtype=pl.Int64)

    adjust_s = pl.Series(
        [
            1_000,
            1_000_000,
            1_000_000_000,
            86_400_000_000_000,  # NS_IN_DAY
        ]
    )

    df = (
        pl.concat(
            [
                unit_phys_s - 1,
                unit_phys_s,
                unit_phys_s + (adjust_s - 1),
                unit_phys_s + adjust_s,
            ]
        )
        .sort()
        .cast(pl.Duration("ns"))
        .to_frame()
        .with_row_index()
    )

    result_vs_col = df.with_columns(
        us=pl.lit(dt, dtype=pl.Duration("us")),
        ms=pl.lit(dt, dtype=pl.Duration("ms")),
    ).select(
        pl.col("duration[ns]").dt.to_string(),
        eq_us=pl.col("duration[ns]") == pl.col("us"),
        eq_ms=pl.col("duration[ns]") == pl.col("ms"),
    )

    F = False
    truth_table = pl.DataFrame(
        [
            ("-PT0.000000001S", True, True),
            ("PT0S", True, True),
            ("PT0.000000999S", True, True),
            ("PT0.000001S", F, True),
            ("PT0.000999999S", F, True),
            ("PT0.001S", F, F),
            ("PT0.999999999S", F, F),
            ("PT1S", F, F),
            ("PT23H59M59.999999999S", F, F),
            ("P1D", F, F),
        ],
        orient="row",
        schema=result_vs_col.schema,
    )

    assert_frame_equal(result_vs_col, truth_table)

    # us -> ns downcast
    q = df.lazy().with_columns(
        eq=pl.col("duration[ns]") == pl.lit(dt, dtype=pl.Datetime("us")),
        lteq=pl.col("duration[ns]") <= pl.lit(dt, dtype=pl.Datetime("us")),
    )

    plan = q.explain()
    assert plan.count("999ns") == 2

    assert_frame_equal(
        q.filter("eq")
        .select(pl.implode("index"), pl.last("duration[ns]").dt.to_string())
        .collect(),
        pl.DataFrame(
            {"index": [[1, 2]], "duration[ns]": "PT0.000000999S"},
            schema={"index": pl.List(pl.get_index_type()), "duration[ns]": pl.String},
        ),
    )

    assert_frame_equal(
        q.filter("lteq")
        .select(pl.implode("index"), pl.last("duration[ns]").dt.to_string())
        .collect(),
        pl.DataFrame(
            {"index": [[0, 1, 2]], "duration[ns]": "PT0.000000999S"},
            schema={"index": pl.List(pl.get_index_type()), "duration[ns]": pl.String},
        ),
    )

    # ms -> ns downcast
    q = df.lazy().with_columns(
        eq=pl.col("duration[ns]") == pl.lit(dt, dtype=pl.Datetime("ms")),
        lteq=pl.col("duration[ns]") <= pl.lit(dt, dtype=pl.Datetime("ms")),
    )

    plan = q.explain()
    assert plan.count("999999ns") == 2

    assert_frame_equal(
        q.filter("eq")
        .select(pl.implode("index"), pl.last("duration[ns]").dt.to_string())
        .collect(),
        pl.DataFrame(
            {"index": [[1, 2, 3, 4]], "duration[ns]": "PT0.000999999S"},
            schema={"index": pl.List(pl.get_index_type()), "duration[ns]": pl.String},
        ),
    )

    assert_frame_equal(
        q.filter("lteq")
        .select(pl.implode("index"), pl.last("duration[ns]").dt.to_string())
        .collect(),
        pl.DataFrame(
            {
                "index": [[0, 1, 2, 3, 4]],
                "duration[ns]": "PT0.000999999S",
            },
            schema={"index": pl.List(pl.get_index_type()), "duration[ns]": pl.String},
        ),
    )


def test_comparison_literal_downcast_flooring_duration_us() -> None:
    dt = timedelta()
    unit_phys_s = pl.Series("duration[us]", [0], dtype=pl.Int64)

    adjust_s = pl.Series(
        [
            1_000,
            1_000_000,
            86_400_000_000,  # US_IN_DAY
        ]
    )

    df = (
        pl.concat(
            [
                unit_phys_s - 1,
                unit_phys_s,
                unit_phys_s + (adjust_s - 1),
                unit_phys_s + adjust_s,
            ]
        )
        .sort()
        .cast(pl.Duration("us"))
        .to_frame()
        .with_row_index()
    )

    result_vs_col = df.with_columns(
        ms=pl.lit(dt, dtype=pl.Duration("ms")),
    ).select(
        pl.col("duration[us]").dt.to_string(),
        eq_ms=pl.col("duration[us]") == pl.col("ms"),
    )

    F = False
    truth_table = pl.DataFrame(
        [
            ("-PT0.000001S", True),
            ("PT0S", True),
            ("PT0.000999S", True),
            ("PT0.001S", F),
            ("PT0.999999S", F),
            ("PT1S", F),
            ("PT23H59M59.999999S", F),
            ("P1D", F),
        ],
        orient="row",
        schema=result_vs_col.schema,
    )

    assert_frame_equal(result_vs_col, truth_table)

    # ms -> us downcast
    q = df.lazy().with_columns(
        eq=pl.col("duration[us]") == pl.lit(dt, dtype=pl.Duration("ms")),
        lteq=pl.col("duration[us]") <= pl.lit(dt, dtype=pl.Duration("ms")),
    )

    plan = q.explain()
    assert plan.count("999µs") == 2

    assert_frame_equal(
        q.filter("eq")
        .select(pl.implode("index"), pl.last("duration[us]").dt.to_string())
        .collect(),
        pl.DataFrame(
            {"index": [[1, 2]], "duration[us]": "PT0.000999S"},
            schema={"index": pl.List(pl.get_index_type()), "duration[us]": pl.String},
        ),
    )

    assert_frame_equal(
        q.filter("lteq")
        .select(pl.implode("index"), pl.last("duration[us]").dt.to_string())
        .collect(),
        pl.DataFrame(
            {"index": [[0, 1, 2]], "duration[us]": "PT0.000999S"},
            schema={"index": pl.List(pl.get_index_type()), "duration[us]": pl.String},
        ),
    )


def test_comparison_literal_downcast_rewrites() -> None:
    lf = pl.LazyFrame(
        schema={
            "i16": pl.Int16,
            "u16": pl.UInt16,
            "datetime[ns]": pl.Datetime("ns"),
            "str": pl.String,
        }
    )

    def assert_rewrite(expr: pl.Expr, search_str: str) -> None:
        plan = lf.select(expr).explain()
        assert search_str in plan

    # Date<>Datetime casts Date to Datetime, out of range values become NULL.
    assert_rewrite(
        pl.col("datetime[ns]") == pl.date(2999, 1, 1),
        'null.repeat([col("datetime[ns]").len()',
    )

    assert_rewrite(
        pl.col("datetime[ns]") == pl.lit(datetime(2026, 1, 1), dtype=pl.Datetime("ms")),
        "is_between([2026-01-01 00:00:00, 2026-01-01 00:00:00.000999999])",
    )

    assert_rewrite(
        pl.col("datetime[ns]")
        <= pl.lit(datetime(2026, 1, 1, microsecond=1000), dtype=pl.Datetime("ms")),
        "<= (2026-01-01 00:00:00.001999999)",
    )

    assert_rewrite(
        pl.col("i16") == pl.lit(10, dtype=pl.Int32),
        'col("i16")) == (10)',
    )

    assert_rewrite(
        pl.col("i16") < (-(1 << 16) - 1),
        ".is_not_null()).then(false).otherwise(null)",
    )

    assert_rewrite(
        pl.col("i16") >= (-(1 << 16) - 1),
        ".is_not_null()).then(true).otherwise(null)",
    )

    assert_rewrite(
        pl.col("i16") < pl.lit(None, dtype=pl.Int16),
        'null.repeat([col("i16").len()])',
    )

    assert_rewrite(
        pl.col("i16").eq_missing(pl.lit(None, dtype=pl.Int16)),
        'col("i16").is_null()',
    )

    assert_rewrite(
        pl.col("i16").eq_missing(None),
        'col("i16").is_null()',
    )

    assert_rewrite(
        pl.col("str").ne_missing(None),
        'col("str").is_not_null()',
    )

    assert_rewrite(
        pl.col("str").ne_missing(None),
        'col("str").is_not_null()',
    )

    assert_rewrite(
        pl.col("i16").eq_missing(-(1 << 16) - 1),
        'false.repeat([col("i16").len()])',
    )

    assert_rewrite(
        pl.col("i16").ne_missing(1 << 16),
        'true.repeat([col("i16").len()])',
    )

    assert_rewrite(
        pl.col("u16").is_between(-1, 10),
        'col("u16")) <= (10)',
    )

    assert_rewrite(
        pl.col("u16").is_between(10, 1 << 16, closed="right"),
        'col("u16")) > (10)',
    )

    assert_rewrite(
        pl.col("u16").is_between(1 << 16, 1 << 17),
        'when(col("u16").is_not_null()).then(false).otherwise(null)',
    )
