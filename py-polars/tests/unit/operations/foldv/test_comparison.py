"""Comparison tests: foldv must behave exactly like native Polars.

Each test runs a fold body through the JIT and through native Polars, which
evaluates the same body one row at a time while carrying the state between rows.
The two must agree on dtype, values, nulls, NaN, signed zero, and errors, in both
eager and lazy queries. Add comparison tests for every operation the JIT supports;
a difference from Polars is a bug unless it is documented as intentional.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

import polars as pl

NAN = float("nan")
INF = float("inf")
STATE = "__foldv_reference_state__"

INTS = pl.DataFrame({"l": pl.Series([1, None, 3, None, -5], dtype=pl.Int64)})
FLOATS = pl.DataFrame({"f": [1.0, NAN, None, -0.0, 0.0, INF, -INF, NAN]})
BOOLS = pl.DataFrame(
    {
        "a": [True, True, True, False, False, False, None, None, None],
        "b": [True, False, None, True, False, None, True, False, None],
    }
)
CASTS = pl.DataFrame(
    {
        "u64": pl.Series([0, 2**63 - 1, 2**63, 2**64 - 1, None], dtype=pl.UInt64),
        "u32": pl.Series([0, 2**31 - 1, 2**31, 2**32 - 1, None], dtype=pl.UInt32),
        "i64": pl.Series([0, -1, 2**31, -(2**31) - 1, None], dtype=pl.Int64),
        "f64": [-0.5, -2147483648.5, 2147483647.9, NAN, None],
        "f64b": [-1.0, 4294967295.5, INF, -INF, 1e300],
        "bool": [True, False, None, True, False],
    }
)


def _reference(df: pl.DataFrame, body: Any, initial_state: pl.Expr) -> list[Any]:
    state = pl.select(initial_state.alias(STATE)).to_series()
    output = []
    for row in range(df.height):
        frame = df.slice(row, 1).with_columns(state)
        state = frame.select(body(pl.col(STATE)).cast(state.dtype).alias(STATE))
        state = state.to_series()
        output.append(state[0])
    return output


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) or math.isnan(right):
            return math.isnan(left) and math.isnan(right)
        return left == right and math.copysign(1, left) == math.copysign(1, right)
    return left == right and (left is None) == (right is None)


def _assert_matches_polars(df: pl.DataFrame, body: Any, initial_state: pl.Expr) -> None:
    expected = _reference(df, body, initial_state)
    expression = pl.foldv(body, initial_state=initial_state)
    for result in (df.select(expression), df.lazy().select(expression).collect()):
        values = result.to_series().to_list()
        assert len(values) == len(expected)
        assert all(map(_same, values, expected)), f"{values} != {expected}"


def _i64(value: int | None) -> pl.Expr:
    return pl.lit(value, dtype=pl.Int64)


ARITHMETIC = {
    "add": lambda left, right: left + right,
    "subtract": lambda left, right: left - right,
    "multiply": lambda left, right: left * right,
}

COMPARISONS = {
    "eq": lambda left, right: left == right,
    "ne": lambda left, right: left != right,
    "lt": lambda left, right: left < right,
    "le": lambda left, right: left <= right,
    "gt": lambda left, right: left > right,
    "ge": lambda left, right: left >= right,
    "eq_missing": lambda left, right: left.eq_missing(right),
    "ne_missing": lambda left, right: left.ne_missing(right),
}


@pytest.mark.parametrize("op", ARITHMETIC)
@pytest.mark.parametrize(
    ("df", "column", "initial_state"),
    [
        (INTS, "l", _i64(1)),
        (INTS, "l", _i64(None)),
        (FLOATS, "f", pl.lit(1.0, dtype=pl.Float64)),
        (FLOATS.cast(pl.Float32), "f", pl.lit(1.0, dtype=pl.Float32)),
    ],
    ids=["int64", "int64-null-initial", "float64", "float32"],
)
def test_foldv_arithmetic_matches_polars_null_and_nan_semantics(
    op: str, df: pl.DataFrame, column: str, initial_state: pl.Expr
) -> None:
    function = ARITHMETIC[op]
    _assert_matches_polars(
        df, lambda state: function(state, pl.col(column)), initial_state
    )


@pytest.mark.parametrize("op", COMPARISONS)
@pytest.mark.parametrize("rhs", [NAN, 0.0, -0.0, 1.0, None], ids=str)
@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64])
def test_foldv_float_comparisons_match_polars(
    op: str, rhs: float | None, dtype: pl.DataType
) -> None:
    # Comparison results drive a branch and are also stored as a nullable Boolean state.
    compare = COMPARISONS[op]
    df = FLOATS.cast(dtype)
    predicate = compare(pl.col("f"), pl.lit(rhs, dtype=dtype))
    _assert_matches_polars(
        df, lambda state: pl.when(predicate).then(state + 1).otherwise(state), _i64(0)
    )
    _assert_matches_polars(df, lambda state: predicate, pl.lit(True))


@pytest.mark.parametrize("op", COMPARISONS)
def test_foldv_integer_comparisons_match_polars(op: str) -> None:
    predicate = COMPARISONS[op](pl.col("l"), _i64(3))
    _assert_matches_polars(
        INTS, lambda state: pl.when(predicate).then(state + 1).otherwise(state), _i64(0)
    )
    _assert_matches_polars(INTS, lambda state: predicate, pl.lit(True))


@pytest.mark.parametrize(
    "body",
    [
        lambda state: pl.col("a") & pl.col("b"),
        lambda state: pl.col("a") | pl.col("b"),
        lambda state: ~pl.col("a"),
        lambda state: state & pl.col("b"),
        lambda state: state | pl.col("a"),
    ],
    ids=["and", "or", "not", "state-and", "state-or"],
)
@pytest.mark.parametrize(
    "initial_state",
    [True, False, None],
    ids=["true", "false", "null"],
)
def test_foldv_boolean_operators_match_polars_kleene_logic(
    body: Any, initial_state: bool | None
) -> None:
    _assert_matches_polars(BOOLS, body, pl.lit(initial_state, dtype=pl.Boolean))


@pytest.mark.parametrize(
    "body",
    [
        lambda state: (
            pl.when(state.is_null()).then(pl.col("l")).otherwise(state + pl.col("l"))
        ),
        lambda state: (
            pl.when(pl.col("l").is_not_null()).then(state + 1).otherwise(state)
        ),
        lambda state: pl.when(pl.col("l") > 2).then(None).otherwise(pl.col("l")),
        lambda state: pl.when(pl.col("l") > 0).then(pl.col("l")).otherwise(None),
        lambda state: (
            pl.when(pl.col("l") > 0)
            .then(pl.when(state.is_null()).then(pl.col("l")).otherwise(None))
            .otherwise(state)
        ),
        lambda state: state.fill_null(pl.col("l")) + pl.col("l").fill_null(0),
        lambda state: (
            pl.when(pl.lit(None).is_null()).then(state + pl.col("l")).otherwise(state)
        ),
    ],
    ids=[
        "recover",
        "is-not-null",
        "null-then",
        "null-otherwise",
        "nested",
        "fill-null",
        "untyped-null-is-null",
    ],
)
@pytest.mark.parametrize("initial_state", [0, None], ids=["valid", "null"])
def test_foldv_state_transitions_match_polars(
    body: Any, initial_state: int | None
) -> None:
    _assert_matches_polars(INTS, body, _i64(initial_state))


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("u64", pl.Int64),
        ("u64", pl.Int32),
        ("u64", pl.UInt32),
        ("u32", pl.Int32),
        ("u32", pl.Int64),
        ("i64", pl.Int32),
        ("i64", pl.UInt32),
        ("i64", pl.UInt64),
        ("i64", pl.Boolean),
        ("f64", pl.Int32),
        ("f64", pl.UInt32),
        ("f64", pl.Int64),
        ("f64", pl.UInt64),
        ("f64", pl.Float32),
        ("f64", pl.Boolean),
        ("f64b", pl.UInt32),
        ("f64b", pl.Int64),
        ("f64b", pl.UInt64),
        ("f64b", pl.Float32),
        ("bool", pl.Int64),
        ("bool", pl.Float64),
    ],
)
@pytest.mark.parametrize("mode", ["strict", "non-strict", "wrapping"])
def test_foldv_casts_match_polars_null_and_overflow_semantics(
    source: str, target: pl.DataType, mode: str
) -> None:
    def body(state: pl.Expr) -> pl.Expr:
        if mode == "wrapping":
            return pl.col(source).cast(target, wrap_numerical=True)
        return pl.col(source).cast(target, strict=mode == "strict")

    initial_state = pl.lit(False if target == pl.Boolean else 0, dtype=target)
    try:
        CASTS.select(body(pl.col(source)))
    except pl.exceptions.InvalidOperationError:
        # A strict cast that fails in Polars must also fail in foldv.
        expression = pl.foldv(body, initial_state=initial_state)
        with pytest.raises(
            pl.exceptions.InvalidOperationError, match="failed in foldv"
        ):
            CASTS.select(expression)
        return
    _assert_matches_polars(CASTS, body, initial_state)


def test_foldv_matches_polars_for_sliced_and_chunked_nullable_inputs() -> None:
    values = pl.DataFrame(
        {"l": pl.Series([None if i % 3 == 0 else i for i in range(40)], dtype=pl.Int64)}
    )
    chunked = pl.concat(
        [values.slice(0, 7), values.slice(7, 13), values.slice(20, 20)],
        rechunk=False,
    )
    flags = pl.DataFrame({"b": [None if i % 5 == 0 else i % 2 == 0 for i in range(40)]})

    def skip_nulls(state: pl.Expr) -> pl.Expr:
        return pl.when(pl.col("l").is_null()).then(state).otherwise(state + pl.col("l"))

    _assert_matches_polars(values.slice(5, 20), skip_nulls, _i64(0))
    _assert_matches_polars(chunked, skip_nulls, _i64(0))
    _assert_matches_polars(
        flags.slice(3, 30),
        lambda state: pl.when(pl.col("b")).then(state + 1).otherwise(state),
        _i64(0),
    )


@pytest.mark.parametrize(
    ("df", "body", "dtype"),
    [
        (
            INTS,
            lambda state: (
                pl.when(state.is_null())
                .then(pl.col("l"))
                .otherwise(state + pl.col("l"))
            ),
            pl.Int64,
        ),
        (FLOATS, lambda state: state + pl.col("f"), pl.Float64),
        (BOOLS, lambda state: state | pl.col("a"), pl.Boolean),
        (INTS, lambda state: state.fill_null(pl.col("l")) * 2, pl.Int64),
    ],
    ids=["recover-int", "float", "boolean-kleene", "fill-null"],
)
def test_foldv_infers_an_untyped_null_initial_state_like_polars(
    df: pl.DataFrame, body: Any, dtype: pl.DataType
) -> None:
    # Like `pl.lit(None) + pl.col("a")`, an untyped null takes the body's type.
    native = df.lazy().select(body(pl.lit(None)).alias("r")).collect_schema()["r"]
    assert native == dtype

    expression = pl.foldv(body, initial_state=None)
    assert df.lazy().select(expression).collect_schema().dtypes() == [dtype]
    for result in (df.select(expression), df.lazy().select(expression).collect()):
        assert result.dtypes == [dtype]
        expected = _reference(df, body, pl.lit(None, dtype=dtype))
        assert all(map(_same, result.to_series().to_list(), expected))


def _untyped_null() -> pl.Expr:
    return pl.lit(None)


@pytest.mark.parametrize(
    ("df", "body", "initial_state"),
    [
        (INTS, lambda state: _untyped_null() + pl.col("l"), _i64(0)),
        (INTS, lambda state: state * _untyped_null() + pl.col("l"), _i64(0)),
        (
            INTS,
            lambda state: (
                pl.when(pl.col("l") > _untyped_null()).then(state + 1).otherwise(state)
            ),
            _i64(0),
        ),
        (
            INTS,
            lambda state: (
                pl.when(_untyped_null() == pl.col("l")).then(state + 1).otherwise(state)
            ),
            _i64(0),
        ),
        (
            INTS,
            lambda state: (
                pl.when(pl.col("l").eq_missing(_untyped_null()))
                .then(state + 1)
                .otherwise(state)
            ),
            _i64(0),
        ),
        (
            INTS,
            lambda state: (
                pl.when(pl.col("l").ne_missing(_untyped_null()))
                .then(state + 1)
                .otherwise(state)
            ),
            _i64(0),
        ),
        (
            FLOATS,
            lambda state: (
                pl.when(pl.col("f").eq_missing(_untyped_null()))
                .then(state + 1)
                .otherwise(state)
            ),
            _i64(0),
        ),
        (INTS, lambda state: pl.col("l") == _untyped_null(), pl.lit(True)),
        (BOOLS, lambda state: pl.col("a") & _untyped_null(), pl.lit(True)),
        (BOOLS, lambda state: _untyped_null() | pl.col("a"), pl.lit(True)),
        (INTS, lambda state: _untyped_null().cast(pl.Int64) + pl.col("l"), _i64(0)),
        (
            INTS,
            lambda state: (
                pl.when(_untyped_null().is_not_null())
                .then(state)
                .otherwise(state + pl.col("l"))
            ),
            _i64(0),
        ),
        (
            INTS,
            lambda state: pl.when(pl.col("l") > 0).then(None).otherwise(None),
            _i64(0),
        ),
        (
            INTS,
            lambda state: state.fill_null(_untyped_null()) + pl.col("l"),
            _i64(None),
        ),
        (
            INTS,
            lambda state: _untyped_null().fill_null(pl.col("l")) + state,
            _i64(0),
        ),
    ],
    ids=[
        "null-plus-column",
        "state-times-null",
        "column-gt-null",
        "null-eq-column",
        "eq-missing-null",
        "ne-missing-null",
        "float-eq-missing-null",
        "boolean-state-eq-null",
        "and-null",
        "null-or",
        "null-cast",
        "null-is-not-null",
        "null-then-null-otherwise",
        "fill-null-with-null",
        "null-fill-null",
    ],
)
def test_foldv_untyped_null_literals_match_polars(
    df: pl.DataFrame, body: Any, initial_state: pl.Expr
) -> None:
    # An untyped null takes the type of whatever it is combined with, as in Polars.
    _assert_matches_polars(df, body, initial_state)


@pytest.mark.parametrize(
    ("df", "column", "body", "initial_state"),
    [
        (INTS, "l", lambda state, value: state + _untyped_null(), _i64(0)),
        (FLOATS, "f", lambda state, value: state - _untyped_null(), pl.lit(1.0)),
        (BOOLS, "a", lambda state, value: state & _untyped_null(), pl.lit(True)),
        (BOOLS, "a", lambda state, value: state | _untyped_null(), pl.lit(False)),
    ],
    ids=["int-add", "float-subtract", "boolean-and", "boolean-or"],
)
def test_foldv_untyped_null_with_only_the_state_matches_polars(
    df: pl.DataFrame, column: str, body: Any, initial_state: pl.Expr
) -> None:
    # `pl.foldv` needs a referenced column for its length; `Expr.foldv` uses its input.
    expected = _reference(df, lambda state: body(state, pl.col(column)), initial_state)
    expression = pl.col(column).foldv(body, initial_state=initial_state)
    for result in (df.select(expression), df.lazy().select(expression).collect()):
        values = result.to_series().to_list()
        assert len(values) == len(expected)
        assert all(map(_same, values, expected)), f"{values} != {expected}"


@pytest.mark.parametrize(
    "transform",
    [
        lambda expr: expr.sort(),
        lambda expr: expr.head(3),
        lambda expr: expr.drop_nulls(),
        lambda expr: expr * 2 + 1,
    ],
    ids=["sort", "head", "drop-nulls", "arithmetic"],
)
def test_expr_foldv_over_an_evaluated_input_matches_polars(transform: Any) -> None:
    # Polars evaluates the input first, so the fold sees exactly the values it computes.
    values = INTS.select(transform(pl.col("l")))

    def body(state: pl.Expr, value: pl.Expr) -> pl.Expr:
        return pl.when(value.is_null()).then(state).otherwise(state + value)

    expected = _reference(values, lambda state: body(state, pl.col("l")), _i64(0))
    expression = transform(pl.col("l")).foldv(body, initial_state=_i64(0))
    for result in (INTS.select(expression), INTS.lazy().select(expression).collect()):
        assert result.to_series().to_list() == expected


NAN_CHECKS = {
    "is_nan": pl.Expr.is_nan,
    "is_not_nan": pl.Expr.is_not_nan,
    "is_finite": pl.Expr.is_finite,
    "is_infinite": pl.Expr.is_infinite,
}


@pytest.mark.parametrize("check", NAN_CHECKS)
@pytest.mark.parametrize(
    ("df", "column"),
    [(FLOATS, "f"), (FLOATS.cast(pl.Float32), "f"), (INTS, "l")],
    ids=["float64", "float32", "int64"],
)
def test_foldv_nan_checks_match_polars(
    check: str, df: pl.DataFrame, column: str
) -> None:
    # Checks drive a branch and are also stored as a nullable Boolean state.
    predicate = NAN_CHECKS[check](pl.col(column))
    _assert_matches_polars(
        df, lambda state: pl.when(predicate).then(state + 1).otherwise(state), _i64(0)
    )
    _assert_matches_polars(df, lambda state: predicate, pl.lit(True))


@pytest.mark.parametrize("check", NAN_CHECKS)
def test_foldv_nan_checks_reject_booleans_like_polars(check: str) -> None:
    predicate = NAN_CHECKS[check](pl.col("a"))
    with pytest.raises(pl.exceptions.InvalidOperationError):
        BOOLS.select(predicate)
    expression = pl.foldv(
        lambda state: pl.when(predicate).then(state + 1).otherwise(state),
        initial_state=_i64(0),
    )
    with pytest.raises(pl.exceptions.InvalidOperationError, match=check):
        BOOLS.select(expression)


@pytest.mark.parametrize(
    ("df", "body", "initial_state"),
    [
        (FLOATS, lambda state: state + pl.col("f").fill_nan(0.0), pl.lit(0.0)),
        (
            FLOATS.cast(pl.Float32),
            lambda state: state + pl.col("f").fill_nan(0.0),
            pl.lit(0.0, dtype=pl.Float32),
        ),
        (
            FLOATS,
            lambda state: (
                pl.when(state.is_null())
                .then(pl.col("f").fill_nan(None))
                .otherwise(state)
            ),
            pl.lit(None, dtype=pl.Float64),
        ),
        (INTS, lambda state: state + pl.col("l").fill_nan(0.0), pl.lit(0.0)),
        (INTS, lambda state: state + pl.col("l").fill_nan(None), _i64(0)),
    ],
    ids=["float64", "float32", "float64-null", "int64", "int64-null"],
)
def test_foldv_fill_nan_matches_polars(
    df: pl.DataFrame, body: Any, initial_state: pl.Expr
) -> None:
    # `fill_nan` keeps nulls; an integer filled with a float becomes Float64.
    _assert_matches_polars(df, body, initial_state)


@pytest.mark.parametrize(
    "literal",
    [5_000_000_000, -1, 1, 2**64 - 1, 1.5, NAN],
    ids=["5e9", "-1", "1", "2^64-1", "1.5", "nan"],
)
@pytest.mark.parametrize(
    "target",
    [pl.Int32, pl.Int64, pl.UInt32, pl.UInt64, pl.Float32, pl.Boolean],
)
@pytest.mark.parametrize("mode", ["strict", "non-strict", "wrapping"])
def test_foldv_literal_casts_match_polars(
    literal: float, target: pl.DataType, mode: str
) -> None:
    # A cast of a literal applies the cast's overflow rule to the literal's own value.
    if mode == "wrapping":
        cast = pl.lit(literal).cast(target, wrap_numerical=True)
    else:
        cast = pl.lit(literal).cast(target, strict=mode == "strict")

    def body(state: pl.Expr) -> pl.Expr:
        return pl.when(pl.col("l").is_null()).then(state).otherwise(cast)

    initial_state = pl.lit(False if target == pl.Boolean else 0, dtype=target)
    try:
        _reference(INTS, body, initial_state)
    except pl.exceptions.InvalidOperationError:
        # A strict cast that fails in Polars must also fail in foldv.
        with pytest.raises(pl.exceptions.InvalidOperationError):
            INTS.select(pl.foldv(body, initial_state=initial_state))
        return
    _assert_matches_polars(INTS, body, initial_state)


WIDENING = pl.DataFrame(
    {
        "u32": pl.Series([1, None, 3], dtype=pl.UInt32),
        "i32": pl.Series([1, -2, None], dtype=pl.Int32),
        "u64": pl.Series([1, 2, None], dtype=pl.UInt64),
        "f32": pl.Series([1.5, None, 2.5], dtype=pl.Float32),
    }
)


@pytest.mark.parametrize(
    ("body", "dtype"),
    [
        (lambda state: state + (pl.col("u32") + -1), pl.Int64),
        (lambda state: state + (pl.col("i32") + 5_000_000_000), pl.Int64),
        (lambda state: state + (pl.col("u32") + 1), pl.UInt32),
        (lambda state: state + (pl.col("u64") + -1), pl.Int64),
        (lambda state: state + (pl.col("u64") + 2**63), pl.UInt64),
        (lambda state: state + (pl.col("f32") + 1), pl.Float32),
        (
            lambda state: pl.when(pl.col("u32") > -1).then(state + 1).otherwise(state),
            pl.Int64,
        ),
        (
            lambda state: (
                pl.when(pl.col("i32") > 0).then(5_000_000_000).otherwise(pl.col("i32"))
            ),
            pl.Int64,
        ),
    ],
    ids=[
        "u32-plus-negative",
        "i32-plus-5e9",
        "u32-plus-1",
        "u64-plus-negative",
        "u64-plus-2^63",
        "f32-plus-1",
        "u32-gt-negative",
        "when-then-5e9",
    ],
)
def test_foldv_untyped_integer_literals_widen_like_polars(
    body: Any, dtype: pl.DataType
) -> None:
    # An untyped integer adopts the other operand's type only when its value fits there.
    _assert_matches_polars(WIDENING, body, pl.lit(0, dtype=dtype))


def test_foldv_rejects_a_literal_that_needs_int128() -> None:
    # Polars widens `i32 + 2**63` to Int128, which foldv does not support.
    expression = pl.foldv(
        lambda state: state + (pl.col("i32") + 2**63), initial_state=_i64(0)
    )
    with pytest.raises(pl.exceptions.InvalidOperationError, match="i128"):
        WIDENING.select(expression)


@pytest.mark.parametrize(
    "body",
    [
        lambda state: (
            pl.when(pl.col("i32") > 0)
            .then(pl.col("i32"))
            .otherwise(pl.lit(None, dtype=pl.Int64))
        ),
        lambda state: (
            pl.when(pl.col("i32") > 0)
            .then(pl.lit(None, dtype=pl.Int64))
            .otherwise(pl.lit(None, dtype=pl.Int64))
        ),
        lambda state: state + pl.lit(None, dtype=pl.Int64) + pl.col("i32"),
        lambda state: (
            pl.when(pl.lit(None, dtype=pl.Float64).is_nan())
            .then(state)
            .otherwise(state + pl.col("i32"))
        ),
        lambda state: pl.lit(None, dtype=pl.Int64).fill_null(pl.col("i32")) + state,
    ],
    ids=["otherwise", "both-branches", "plus-column", "is-nan", "fill-null"],
)
@pytest.mark.parametrize("initial_state", [0, None], ids=["valid", "null"])
def test_foldv_typed_null_literals_match_polars(
    body: Any, initial_state: int | None
) -> None:
    # A typed null keeps its own dtype instead of taking the other operand's.
    _assert_matches_polars(WIDENING, body, _i64(initial_state))


TEXTS = pl.DataFrame(
    {
        "s": ["a", "", None, "héllo", "b", "日本", "a", "z"],
        "t": ["a", "b", "x", None, "é", "日本", "", "y"],
    }
)


@pytest.mark.parametrize("op", COMPARISONS)
@pytest.mark.parametrize(
    "rhs",
    [pl.lit("b"), pl.lit(""), pl.col("t"), pl.lit(None, dtype=pl.String), pl.lit(None)],
    ids=["literal", "empty-literal", "column", "typed-null", "untyped-null"],
)
def test_foldv_string_comparisons_match_polars(op: str, rhs: pl.Expr) -> None:
    # Comparisons drive a branch and are also stored as a nullable Boolean state.
    predicate = COMPARISONS[op](pl.col("s"), rhs)
    _assert_matches_polars(
        TEXTS,
        lambda state: pl.when(predicate).then(state + 1).otherwise(state),
        _i64(0),
    )
    _assert_matches_polars(TEXTS, lambda state: predicate, pl.lit(True))


@pytest.mark.parametrize(
    "body",
    [
        lambda state: state + pl.col("s"),
        lambda state: state + (pl.col("s") + pl.col("t")),
        lambda state: state + "·" + pl.col("t"),
        lambda state: state + pl.concat_str([pl.col("s"), pl.col("t")], separator=", "),
        lambda state: pl.when(pl.col("s") > state).then(pl.col("s")).otherwise(state),
        lambda state: (
            pl.when(state.is_null()).then(pl.col("t")).otherwise(state + pl.col("s"))
        ),
        lambda state: state.fill_null("∅") + pl.col("s").fill_null("_"),
        lambda state: (
            pl.when(pl.col("s") == "b").then(None).otherwise(state + pl.col("t"))
        ),
    ],
    ids=[
        "concatenate-column",
        "concatenate-two-columns",
        "concatenate-literal",
        "concat-str",
        "replace-with-input",
        "restart-on-null",
        "fill-null",
        "null-branch",
    ],
)
@pytest.mark.parametrize(
    "initial_state",
    [pl.lit(""), pl.lit("start→"), pl.lit(None, dtype=pl.String)],
    ids=["empty", "text", "null"],
)
def test_foldv_string_states_match_polars(body: Any, initial_state: pl.Expr) -> None:
    _assert_matches_polars(TEXTS, body, initial_state)


@pytest.mark.parametrize(
    "body",
    [
        lambda state: state + pl.col("s").str.len_bytes(),
        lambda state: state + pl.col("s").str.len_chars(),
        lambda state: state + (pl.col("s") + pl.col("t")).str.len_chars(),
        lambda state: (
            pl.when(pl.col("s").str.len_bytes() > pl.col("t").str.len_bytes())
            .then(state + 1)
            .otherwise(state)
        ),
    ],
    ids=["len-bytes", "len-chars", "len-chars-of-concatenation", "len-bytes-branch"],
)
def test_foldv_string_lengths_match_polars(body: Any) -> None:
    _assert_matches_polars(TEXTS, body, _i64(0))


def test_foldv_untyped_null_initial_state_becomes_a_string_like_polars() -> None:
    # Like `pl.lit(None) + pl.col("s")`, an untyped null state takes the String type.
    def body(state: pl.Expr) -> pl.Expr:
        return pl.when(state.is_null()).then(pl.col("s")).otherwise(state + pl.col("s"))

    expression = pl.foldv(body, initial_state=None)
    assert TEXTS.lazy().select(expression).collect_schema().dtypes() == [pl.String]
    expected = _reference(TEXTS, body, pl.lit(None, dtype=pl.String))
    assert TEXTS.select(expression).to_series().to_list() == expected


DIVISION = {
    "true_divide": lambda left, right: left / right,
    "floor_divide": lambda left, right: left // right,
    "modulus": lambda left, right: left % right,
}
NUMERIC_DTYPES = [pl.Int32, pl.Int64, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64]


def _division_values(dtype: pl.DataType) -> list[int | float | None]:
    if dtype.is_float():
        return [None, NAN, -INF, INF, -7.5, -3.0, -0.0, 0.0, 3.0, 7.5, 1e-30]
    bits = 32 if dtype in (pl.Int32, pl.UInt32) else 64
    if dtype.is_unsigned_integer():
        return [None, 0, 1, 3, 7, 2 ** (bits - 1), 2**bits - 1]
    return [None, -(2 ** (bits - 1)), -7, -3, -1, 0, 1, 3, 7, 2 ** (bits - 1) - 1]


@pytest.mark.parametrize("op", DIVISION)
@pytest.mark.parametrize("left_dtype", NUMERIC_DTYPES)
@pytest.mark.parametrize("right_dtype", NUMERIC_DTYPES)
def test_foldv_division_matches_polars(
    op: str, left_dtype: pl.DataType, right_dtype: pl.DataType
) -> None:
    # Every operand pair, including nulls and zero divisors, must be safe to evaluate.
    pairs = [
        (left, right)
        for left in _division_values(left_dtype)
        for right in _division_values(right_dtype)
    ]
    df = pl.DataFrame(
        {
            "a": pl.Series([left for left, _ in pairs], dtype=left_dtype),
            "b": pl.Series([right for _, right in pairs], dtype=right_dtype),
        }
    )
    function = DIVISION[op]
    body = lambda state: function(pl.col("a"), pl.col("b"))  # noqa: E731
    expected = df.select(body(None)).to_series()
    initial_state = pl.lit(1, dtype=expected.dtype)
    expression = pl.foldv(body, initial_state=initial_state).alias("f")
    assert df.lazy().select(expression).collect_schema()["f"] == expected.dtype
    if expected.dtype == pl.Int128:
        # Mixed UInt64/signed integers require an optional width from phase 18.
        with pytest.raises(pl.exceptions.InvalidOperationError, match="dtype i128"):
            df.select(expression)
        with pytest.raises(pl.exceptions.InvalidOperationError, match="dtype i128"):
            df.lazy().select(expression).collect()
        return
    _assert_matches_polars(df, body, initial_state)


@pytest.mark.parametrize("op", DIVISION)
@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
@pytest.mark.parametrize("null_initial", [False, True])
def test_foldv_division_carries_and_recovers_state(
    op: str, dtype: pl.DataType, null_initial: bool
) -> None:
    df = pl.DataFrame({"b": pl.Series([3, 0, None, 2, 1], dtype=dtype)})
    state_dtype = pl.Float64 if op == "true_divide" and not dtype.is_float() else dtype
    _assert_matches_polars(
        df,
        lambda state: DIVISION[op](state.fill_null(12), pl.col("b")),
        pl.lit(
            None if null_initial else 24,
            dtype=state_dtype,
        ),
    )


@pytest.mark.parametrize("op", DIVISION)
@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
@pytest.mark.parametrize("literal_first", [False, True])
def test_foldv_division_literals_match_polars(
    op: str, dtype: pl.DataType, literal_first: bool
) -> None:
    df = pl.DataFrame({"a": pl.Series(_division_values(dtype), dtype=dtype)})
    for divisor in [3, -3, 0, 3.0, -0.0, None]:

        def body(state: pl.Expr, divisor: int | float | None = divisor) -> pl.Expr:
            left, right = pl.col("a"), pl.lit(divisor)
            return (
                DIVISION[op](right, left)
                if literal_first
                else DIVISION[op](left, right)
            )

        # Polars can report UInt64 in its schema for `-3 // u64`, but executes as Int64.
        result_dtype = df.select(body(pl.col(STATE))).to_series().dtype
        _assert_matches_polars(
            df,
            body,
            pl.lit(0, dtype=result_dtype),
        )


NUMERIC_UNARY = {
    "negate": lambda value: -value,
    "abs": lambda value: value.abs(),
    "sign": lambda value: value.sign(),
    "floor": lambda value: value.floor(),
    "ceil": lambda value: value.ceil(),
    "round": lambda value: value.round(),
    "sqrt": lambda value: value.sqrt(),
}


def _assert_numeric_body(df: pl.DataFrame, body: Any) -> None:
    # Use the executed dtype; native Polars' schema for literals can remain unknown.
    expected = df.head(1).select(body(pl.col("a"))).to_series()
    expression = pl.foldv(
        lambda state: body(pl.col("a")), initial_state=pl.lit(0, dtype=expected.dtype)
    ).alias("f")
    assert df.lazy().select(expression).collect_schema()["f"] == expected.dtype
    _assert_matches_polars(
        df, lambda state: body(pl.col("a")), pl.lit(0, dtype=expected.dtype)
    )


@pytest.mark.parametrize("op", NUMERIC_UNARY)
@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_foldv_numeric_unary_matches_polars(op: str, dtype: pl.DataType) -> None:
    df = pl.DataFrame({"a": pl.Series(_division_values(dtype), dtype=dtype)})
    function = NUMERIC_UNARY[op]
    if op == "negate" and dtype.is_unsigned_integer():
        with pytest.raises(pl.exceptions.InvalidOperationError):
            df.select(function(pl.col("a")))
        with pytest.raises(pl.exceptions.InvalidOperationError, match="negation"):
            df.select(
                pl.foldv(
                    lambda state: function(pl.col("a")),
                    initial_state=pl.lit(0, dtype=dtype),
                )
            )
        return
    _assert_numeric_body(df, function)


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64])
@pytest.mark.parametrize(
    "decimals", [0, 1, 2, 8, 46, 47, 299, 300, 308, 325, 326, 2**32 - 1]
)
@pytest.mark.parametrize("mode", ["half_to_even", "half_away_from_zero"])
def test_foldv_round_matches_polars(
    dtype: pl.DataType, decimals: int, mode: Any
) -> None:
    values = [
        None,
        NAN,
        INF,
        -INF,
        -3.5,
        -2.5,
        -1.5,
        -0.5,
        -0.0,
        0.0,
        0.5,
        1.5,
        2.5,
        3.5,
        1.005,
        2.675,
        1e30,
        1e-30,
    ]
    if dtype == pl.Float64:
        values += [1e308, 1e-308, 5e-324, -5e-324]
    df = pl.DataFrame({"a": pl.Series(values, dtype=dtype)})
    _assert_numeric_body(df, lambda value: value.round(decimals, mode=mode))


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
@pytest.mark.parametrize("exponent", [0, 1, 2, 3, 9, 10, 11, 0.5, -1.0, NAN, INF, None])
def test_foldv_pow_literal_matches_polars(dtype: pl.DataType, exponent: Any) -> None:
    # Large integer powers deliberately overflow their fixed-width results.
    df = pl.DataFrame({"a": pl.Series(_division_values(dtype), dtype=dtype)})
    if dtype.is_integer() and exponent == 11:
        # Native generic integer pow panics on overflow in debug builds.
        df = pl.DataFrame({"a": pl.Series([None, 0, 1, 2], dtype=dtype)})
    if exponent is None:
        exponent = pl.lit(None, dtype=pl.Int64)
    _assert_numeric_body(df, lambda value: value.pow(exponent))


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
@pytest.mark.parametrize("exponent_dtype", NUMERIC_DTYPES)
def test_foldv_pow_column_matches_polars(
    dtype: pl.DataType, exponent_dtype: pl.DataType
) -> None:
    exponents = [None, 0, 1, 2, 3, 9, 10]
    if exponent_dtype.is_float():
        exponents += [0.5, -1.0, NAN, INF]
    bases = _division_values(dtype)
    pairs = [(base, exp) for base in bases for exp in exponents]
    df = pl.DataFrame(
        {
            "a": pl.Series([base for base, _ in pairs], dtype=dtype),
            "e": pl.Series([exp for _, exp in pairs], dtype=exponent_dtype),
        }
    )
    _assert_numeric_body(df, lambda value: value.pow(pl.col("e")))


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
@pytest.mark.parametrize(
    "bounds", [(1, 3), (3, 1), (None, 3), (1, None), (None, None), (-1, 3.5)]
)
def test_foldv_clip_matches_polars(dtype: pl.DataType, bounds: tuple[Any, Any]) -> None:
    df = pl.DataFrame({"a": pl.Series(_division_values(dtype), dtype=dtype)})
    if dtype.is_unsigned_integer() and bounds[0] == -1:
        with pytest.raises(pl.exceptions.InvalidOperationError):
            df.select(pl.col("a").clip(*bounds))
        with pytest.raises(pl.exceptions.InvalidOperationError, match="conversion"):
            df.select(
                pl.foldv(
                    lambda state: pl.col("a").clip(*bounds),
                    initial_state=pl.lit(0, dtype=dtype),
                )
            )
        return
    _assert_numeric_body(df, lambda value: value.clip(*bounds))


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64])
def test_foldv_clip_null_and_nan_bounds_match_polars(dtype: pl.DataType) -> None:
    values = [None, NAN, INF, -INF, -3.0, -0.0, 0.0, 3.0]
    pairs = [(value, lo, hi) for value in values for lo in values for hi in values]
    df = pl.DataFrame(
        {
            "a": pl.Series([v for v, _, _ in pairs], dtype=dtype),
            "lo": pl.Series([lo for _, lo, _ in pairs], dtype=dtype),
            "hi": pl.Series([hi for _, _, hi in pairs], dtype=dtype),
        }
    )
    _assert_numeric_body(df, lambda value: value.clip(pl.col("lo"), pl.col("hi")))


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
@pytest.mark.parametrize("right_dtype", NUMERIC_DTYPES)
@pytest.mark.parametrize("maximum", [False, True])
def test_foldv_horizontal_min_max_matches_polars(
    dtype: pl.DataType, right_dtype: pl.DataType, maximum: bool
) -> None:
    pairs = [
        (left, right)
        for left in _division_values(dtype)
        for right in _division_values(right_dtype)
    ]
    df = pl.DataFrame(
        {
            "a": pl.Series([left for left, _ in pairs], dtype=dtype),
            "b": pl.Series([right for _, right in pairs], dtype=right_dtype),
        }
    )
    function = pl.max_horizontal if maximum else pl.min_horizontal
    # Signed/UInt64 pairs can require optional Int128 in full Python builds.
    result_dtype = df.select(function("a", "b")).to_series().dtype
    if result_dtype == pl.Int128:
        with pytest.raises(pl.exceptions.InvalidOperationError, match="i128"):
            df.select(
                pl.foldv(
                    lambda state: function("a", "b"),
                    initial_state=pl.lit(0, dtype=result_dtype),
                )
            )
        return
    _assert_numeric_body(df, lambda value: function(value, pl.col("b")))


@pytest.mark.parametrize("function", [pl.min_horizontal, pl.max_horizontal])
@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_foldv_running_min_max_recovers_from_null(
    function: Any, dtype: pl.DataType
) -> None:
    df = pl.DataFrame({"a": pl.Series(_division_values(dtype), dtype=dtype)})
    _assert_matches_polars(
        df, lambda state: function(state, pl.col("a")), pl.lit(None, dtype=dtype)
    )


def test_foldv_numeric_strict_casts_match_polars() -> None:
    for body in [
        lambda state: pl.col("a").pow(pl.col("e")),
        lambda state: pl.col("a").clip(pl.col("e"), 5),
    ]:
        df = pl.DataFrame(
            {"a": pl.Series([None, 2, 3], dtype=pl.Int32), "e": [2**32, -1, 2]}
        )
        with pytest.raises(pl.exceptions.InvalidOperationError):
            _reference(df, body, _i64(0))
        with pytest.raises(pl.exceptions.InvalidOperationError, match="conversion"):
            df.select(pl.foldv(body, initial_state=pl.lit(0, dtype=pl.Int32)))


@pytest.mark.parametrize("op", NUMERIC_UNARY)
@pytest.mark.parametrize("dtype", [pl.Int32, pl.Int64, pl.Float32, pl.Float64])
@pytest.mark.parametrize("null_initial", [False, True])
def test_foldv_numeric_unary_carries_state(
    op: str, dtype: pl.DataType, null_initial: bool
) -> None:
    if op == "sqrt" and dtype.is_integer():
        dtype = pl.Float64
    df = pl.DataFrame({"a": pl.Series([1, None, 3, 2, 5], dtype=dtype)})
    _assert_matches_polars(
        df,
        lambda state: NUMERIC_UNARY[op](state.fill_null(0) + pl.col("a").fill_null(0)),
        pl.lit(None if null_initial else 1, dtype=dtype),
    )


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_foldv_clip_and_pow_carry_state(dtype: pl.DataType) -> None:
    df = pl.DataFrame({"a": pl.Series([1, None, 3, 2, 5], dtype=dtype)})
    _assert_matches_polars(
        df,
        lambda state: (
            (state.fill_null(0) + pl.col("a").fill_null(0)).pow(3).clip(0, 20)
        ),
        pl.lit(None, dtype=dtype),
    )


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_foldv_horizontal_multiple_inputs(dtype: pl.DataType) -> None:
    df = pl.DataFrame({"a": pl.Series(_division_values(dtype), dtype=dtype)})
    for function in [pl.min_horizontal, pl.max_horizontal]:
        _assert_numeric_body(
            df,
            lambda value, function=function: function(
                value, pl.lit(2, dtype=dtype), None
            ),
        )
        _assert_numeric_body(df, lambda value, function=function: function(value))


@pytest.mark.parametrize("dtype", [pl.Int32, pl.Int64, pl.UInt32, pl.UInt64])
def test_foldv_pow_generic_wraps_integer_overflow(dtype: pl.DataType) -> None:
    # Native generic pow panics on overflow in debug builds and wraps in release.
    bits = 32 if dtype in (pl.Int32, pl.UInt32) else 64
    maximum = 2 ** (bits - int(dtype.is_signed_integer())) - 1
    df = pl.DataFrame({"a": pl.Series([maximum, None, 2], dtype=dtype)})
    result = df.select(
        pl.col("a").foldv(
            lambda state, value: value.pow(11), initial_state=pl.lit(0, dtype=dtype)
        )
    )
    expected = [pow(maximum, 11, 2**bits), None, 2**11]
    if dtype.is_signed_integer():
        expected = [
            v - 2**bits if v is not None and v >= 2 ** (bits - 1) else v
            for v in expected
        ]
    assert result.to_series().to_list() == expected


def test_foldv_numeric_errors_are_clear() -> None:
    df = pl.DataFrame({"a": [1, 2], "e": [-1, 2]})
    with pytest.raises(pl.exceptions.InvalidOperationError, match="conversion"):
        df.select(pl.foldv(lambda state: pl.col("a").pow(pl.col("e")), initial_state=0))
    with pytest.raises(pl.exceptions.InvalidOperationError, match=r"root\.min"):
        df.select(
            pl.foldv(lambda state: pl.col("a").clip(pl.lit("1"), 3), initial_state=0)
        )
    with pytest.raises(pl.exceptions.InvalidOperationError, match="exponent"):
        df.select(pl.foldv(lambda state: pl.col("a").pow(None), initial_state=0))


def test_foldv_numeric_error_in_unselected_branch_is_not_evaluated() -> None:
    df = pl.DataFrame({"a": [1, 2], "e": [-1, -1]})
    expression = pl.foldv(
        lambda state: (
            pl.when(pl.col("a") > 0)
            .then(state + pl.col("a"))
            .otherwise(pl.col("a").pow(pl.col("e")))
        ),
        initial_state=0,
    )
    assert df.select(expression).to_series().to_list() == [1, 3]
