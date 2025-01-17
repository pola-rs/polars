from __future__ import annotations

import operator
from typing import Any, Callable

import pytest

import polars as pl
from polars.exceptions import InvalidOperationError, ShapeError
from polars.testing import assert_frame_equal, assert_series_equal
from tests.unit.operations.arithmetic.utils import (
    BROADCAST_SERIES_COMBINATIONS,
    EXEC_OP_COMBINATIONS,
)


@pytest.mark.parametrize(
    "list_side", ["left", "left3", "both", "right3", "right", "none"]
)
@pytest.mark.parametrize(
    "broadcast_series",
    BROADCAST_SERIES_COMBINATIONS,
)
@pytest.mark.parametrize("exec_op", EXEC_OP_COMBINATIONS)
@pytest.mark.slow
def test_list_arithmetic_values(
    list_side: str,
    broadcast_series: Callable[
        [pl.Series, pl.Series, pl.Series], tuple[pl.Series, pl.Series, pl.Series]
    ],
    exec_op: Callable[[pl.Series, pl.Series, Any], pl.Series],
) -> None:
    """
    Tests value correctness.

    This test checks for output value correctness (a + b == c) across different
    codepaths, by wrapping the values (a, b, c) in different combinations of
    list / primitive columns.
    """
    import operator as op

    dtypes: list[Any] = [pl.Null, pl.Null, pl.Null]
    dtype: Any = pl.Null

    def materialize_list(v: Any) -> pl.Series:
        return pl.Series(
            [[None, v, None]],
            dtype=pl.List(dtype),
        )

    def materialize_list3(v: Any) -> pl.Series:
        return pl.Series(
            [[[[None, v], None], None]],
            dtype=pl.List(pl.List(pl.List(dtype))),
        )

    def materialize_primitive(v: Any) -> pl.Series:
        return pl.Series([v], dtype=dtype)

    def materialize_series(
        l: Any,  # noqa: E741
        r: Any,
        o: Any,
    ) -> tuple[pl.Series, pl.Series, pl.Series]:
        nonlocal dtype

        dtype = dtypes[0]
        l = {  # noqa: E741
            "left":   materialize_list,
            "left3":  materialize_list3,
            "both":   materialize_list,
            "right":  materialize_primitive,
            "right3": materialize_primitive,
            "none":   materialize_primitive,
        }[list_side](l)  # fmt: skip

        dtype = dtypes[1]
        r = {
            "left":   materialize_primitive,
            "left3":  materialize_primitive,
            "both":   materialize_list,
            "right":  materialize_list,
            "right3": materialize_list3,
            "none":   materialize_primitive,
        }[list_side](r)  # fmt: skip

        dtype = dtypes[2]
        o = {
            "left":   materialize_list,
            "left3":  materialize_list3,
            "both":   materialize_list,
            "right":  materialize_list,
            "right3": materialize_list3,
            "none":   materialize_primitive,
        }[list_side](o)  # fmt: skip

        assert l.len() == 1
        assert r.len() == 1
        assert o.len() == 1

        return broadcast_series(l, r, o)

    # Signed
    dtypes = [pl.Int8, pl.Int8, pl.Int8]

    l, r, o = materialize_series(2, 3, 5)  # noqa: E741
    assert_series_equal(exec_op(l, r, op.add), o)

    l, r, o = materialize_series(-5, 127, 124)  # noqa: E741
    assert_series_equal(exec_op(l, r, op.sub), o)

    l, r, o = materialize_series(-5, 127, -123)  # noqa: E741
    assert_series_equal(exec_op(l, r, op.mul), o)

    l, r, o = materialize_series(-5, 3, -2)  # noqa: E741
    assert_series_equal(exec_op(l, r, op.floordiv), o)

    l, r, o = materialize_series(-5, 3, 1)  # noqa: E741
    assert_series_equal(exec_op(l, r, op.mod), o)

    dtypes = [pl.UInt8, pl.UInt8, pl.Float64]
    l, r, o = materialize_series(2, 128, 0.015625)  # noqa: E741
    assert_series_equal(exec_op(l, r, op.truediv), o)

    # Unsigned
    dtypes = [pl.UInt8, pl.UInt8, pl.UInt8]

    l, r, o = materialize_series(2, 3, 5)  # noqa: E741
    assert_series_equal(exec_op(l, r, op.add), o)

    l, r, o = materialize_series(2, 3, 255)  # noqa: E741
    assert_series_equal(exec_op(l, r, op.sub), o)

    l, r, o = materialize_series(2, 128, 0)  # noqa: E741
    assert_series_equal(exec_op(l, r, op.mul), o)

    l, r, o = materialize_series(5, 2, 2)  # noqa: E741
    assert_series_equal(exec_op(l, r, op.floordiv), o)

    l, r, o = materialize_series(5, 2, 1)  # noqa: E741
    assert_series_equal(exec_op(l, r, op.mod), o)

    dtypes = [pl.UInt8, pl.UInt8, pl.Float64]
    l, r, o = materialize_series(2, 128, 0.015625)  # noqa: E741
    assert_series_equal(exec_op(l, r, op.truediv), o)

    # Floats. Note we pick Float32 to ensure there is no accidental upcasting
    # to Float64.
    dtypes = [pl.Float32, pl.Float32, pl.Float32]
    l, r, o = materialize_series(1.7, 2.3, 4.0)  # noqa: E741
    assert_series_equal(exec_op(l, r, op.add), o)

    l, r, o = materialize_series(1.7, 2.3, -0.5999999999999999)  # noqa: E741
    assert_series_equal(exec_op(l, r, op.sub), o)

    l, r, o = materialize_series(1.7, 2.3, 3.9099999999999997)  # noqa: E741
    assert_series_equal(exec_op(l, r, op.mul), o)

    l, r, o = materialize_series(7.0, 3.0, 2.0)  # noqa: E741
    assert_series_equal(exec_op(l, r, op.floordiv), o)

    l, r, o = materialize_series(-5.0, 3.0, 1.0)  # noqa: E741
    assert_series_equal(exec_op(l, r, op.mod), o)

    l, r, o = materialize_series(2.0, 128.0, 0.015625)  # noqa: E741
    assert_series_equal(exec_op(l, r, op.truediv), o)

    #
    # Tests for zero behavior
    #

    # Integer

    dtypes = [pl.UInt8, pl.UInt8, pl.UInt8]

    l, r, o = materialize_series(1, 0, None)  # noqa: E741
    assert_series_equal(exec_op(l, r, op.floordiv), o)
    assert_series_equal(exec_op(l, r, op.mod), o)

    l, r, o = materialize_series(0, 0, None)  # noqa: E741
    assert_series_equal(exec_op(l, r, op.floordiv), o)
    assert_series_equal(exec_op(l, r, op.mod), o)

    dtypes = [pl.UInt8, pl.UInt8, pl.Float64]

    l, r, o = materialize_series(1, 0, float("inf"))  # noqa: E741
    assert_series_equal(exec_op(l, r, op.truediv), o)

    l, r, o = materialize_series(0, 0, float("nan"))  # noqa: E741
    assert_series_equal(exec_op(l, r, op.truediv), o)

    # Float

    dtypes = [pl.Float32, pl.Float32, pl.Float32]

    l, r, o = materialize_series(1, 0, float("inf"))  # noqa: E741
    assert_series_equal(exec_op(l, r, op.floordiv), o)

    l, r, o = materialize_series(1, 0, float("nan"))  # noqa: E741
    assert_series_equal(exec_op(l, r, op.mod), o)

    l, r, o = materialize_series(1, 0, float("inf"))  # noqa: E741
    assert_series_equal(exec_op(l, r, op.truediv), o)

    l, r, o = materialize_series(0, 0, float("nan"))  # noqa: E741
    assert_series_equal(exec_op(l, r, op.floordiv), o)

    l, r, o = materialize_series(0, 0, float("nan"))  # noqa: E741
    assert_series_equal(exec_op(l, r, op.mod), o)

    l, r, o = materialize_series(0, 0, float("nan"))  # noqa: E741
    assert_series_equal(exec_op(l, r, op.truediv), o)

    #
    # Tests for NULL behavior
    #

    for dtype, truediv_dtype in [  # type: ignore[misc]
        [pl.Int8, pl.Float64],
        [pl.Float32, pl.Float32],
    ]:
        for vals in [
            [None, None, None],
            [0, None, None],
            [None, 0, None],
            [0, None, None],
            [None, 0, None],
            [3, None, None],
            [None, 3, None],
        ]:
            dtypes = 3 * [dtype]

            l, r, o = materialize_series(*vals)  # type: ignore[misc]  # noqa: E741
            assert_series_equal(exec_op(l, r, op.add), o)
            assert_series_equal(exec_op(l, r, op.sub), o)
            assert_series_equal(exec_op(l, r, op.mul), o)
            assert_series_equal(exec_op(l, r, op.floordiv), o)
            assert_series_equal(exec_op(l, r, op.mod), o)
            dtypes[2] = truediv_dtype  # type: ignore[has-type]
            l, r, o = materialize_series(*vals)  # type: ignore[misc]  # noqa: E741
            assert_series_equal(exec_op(l, r, op.truediv), o)

    # Type upcasting for Boolean and Null

    # Check boolean upcasting
    dtypes = [pl.Boolean, pl.UInt8, pl.UInt8]

    l, r, o = materialize_series(True, 3, 4)  # noqa: E741
    assert_series_equal(exec_op(l, r, op.add), o)

    l, r, o = materialize_series(True, 3, 254)  # noqa: E741
    assert_series_equal(exec_op(l, r, op.sub), o)

    l, r, o = materialize_series(True, 3, 3)  # noqa: E741
    assert_series_equal(exec_op(l, r, op.mul), o)

    l, r, o = materialize_series(True, 3, 0)  # noqa: E741
    if list_side != "none":
        # TODO: FIXME: We get an error on non-lists with this:
        # "floor_div operation not supported for dtype `bool`"
        assert_series_equal(exec_op(l, r, op.floordiv), o)

    l, r, o = materialize_series(True, 3, 1)  # noqa: E741
    assert_series_equal(exec_op(l, r, op.mod), o)

    dtypes = [pl.Boolean, pl.UInt8, pl.Float64]
    l, r, o = materialize_series(True, 128, 0.0078125)  # noqa: E741
    assert_series_equal(exec_op(l, r, op.truediv), o)

    # Check Null upcasting
    dtypes = [pl.Null, pl.UInt8, pl.UInt8]
    l, r, o = materialize_series(None, 3, None)  # noqa: E741
    assert_series_equal(exec_op(l, r, op.add), o)
    assert_series_equal(exec_op(l, r, op.sub), o)
    assert_series_equal(exec_op(l, r, op.mul), o)
    if list_side != "none":
        assert_series_equal(exec_op(l, r, op.floordiv), o)
    assert_series_equal(exec_op(l, r, op.mod), o)

    dtypes = [pl.Null, pl.UInt8, pl.Float64]
    l, r, o = materialize_series(None, 3, None)  # noqa: E741
    assert_series_equal(exec_op(l, r, op.truediv), o)


@pytest.mark.parametrize("exec_op", EXEC_OP_COMBINATIONS)
def test_list_add_supertype(
    exec_op: Callable[[pl.Series, pl.Series, Any], pl.Series],
) -> None:
    import operator as op

    a = pl.Series("a", [[1], [2]], dtype=pl.List(pl.Int8))
    b = pl.Series("b", [[1], [999]], dtype=pl.List(pl.Int64))

    assert_series_equal(
        exec_op(a, b, op.add),
        pl.Series("a", [[2], [1001]], dtype=pl.List(pl.Int64)),
    )


@pytest.mark.parametrize("exec_op", EXEC_OP_COMBINATIONS)
@pytest.mark.parametrize(
    "broadcast_series",
    BROADCAST_SERIES_COMBINATIONS,
)
@pytest.mark.slow
def test_list_numeric_op_validity_combination(
    broadcast_series: Callable[
        [pl.Series, pl.Series, pl.Series], tuple[pl.Series, pl.Series, pl.Series]
    ],
    exec_op: Callable[[pl.Series, pl.Series, Any], pl.Series],
) -> None:
    import operator as op

    a = pl.Series("a", [[1], [2], None, [None], [11], [1111]], dtype=pl.List(pl.Int32))
    b = pl.Series("b", [[1], [3], [11], [1111], None, [None]], dtype=pl.List(pl.Int64))
    # expected result
    e = pl.Series("a", [[2], [5], None, [None], None, [None]], dtype=pl.List(pl.Int64))

    assert_series_equal(
        exec_op(a, b, op.add),
        e,
    )

    a = pl.Series("a", [[1]], dtype=pl.List(pl.Int32))
    b = pl.Series("b", [None], dtype=pl.Int64)
    e = pl.Series("a", [[None]], dtype=pl.List(pl.Int64))

    a, b, e = broadcast_series(a, b, e)
    assert_series_equal(exec_op(a, b, op.add), e)

    a = pl.Series("a", [None], dtype=pl.List(pl.Int32))
    b = pl.Series("b", [1], dtype=pl.Int64)
    e = pl.Series("a", [None], dtype=pl.List(pl.Int64))

    a, b, e = broadcast_series(a, b, e)
    assert_series_equal(exec_op(a, b, op.add), e)

    a = pl.Series("a", [None], dtype=pl.List(pl.Int32))
    b = pl.Series("b", [0], dtype=pl.Int64)
    e = pl.Series("a", [None], dtype=pl.List(pl.Int64))

    a, b, e = broadcast_series(a, b, e)
    assert_series_equal(exec_op(a, b, op.floordiv), e)


def test_list_add_alignment() -> None:
    a = pl.Series("a", [[1, 1], [1, 1, 1]])
    b = pl.Series("b", [[1, 1, 1], [1, 1]])

    df = pl.DataFrame([a, b])

    with pytest.raises(ShapeError):
        df.select(x=pl.col("a") + pl.col("b"))

    # Test masking and slicing
    a = pl.Series("a", [[1, 1, 1], [1], [1, 1], [1, 1, 1]])
    b = pl.Series("b", [[1, 1], [1], [1, 1, 1], [1]])
    c = pl.Series("c", [1, 1, 1, 1])
    p = pl.Series("p", [True, True, False, False])

    df = pl.DataFrame([a, b, c, p]).filter("p").slice(1)

    for rhs in [pl.col("b"), pl.lit(1), pl.col("c"), pl.lit([1])]:
        assert_series_equal(
            df.select(x=pl.col("a") + rhs).to_series(), pl.Series("x", [[2]])
        )

    df = df.vstack(df)

    for rhs in [pl.col("b"), pl.lit(1), pl.col("c"), pl.lit([1])]:
        assert_series_equal(
            df.select(x=pl.col("a") + rhs).to_series(), pl.Series("x", [[2], [2]])
        )


@pytest.mark.parametrize("exec_op", EXEC_OP_COMBINATIONS)
@pytest.mark.slow
def test_list_add_empty_lists(
    exec_op: Callable[[pl.Series, pl.Series, Any], pl.Series],
) -> None:
    l = pl.Series(  # noqa: E741
        "x",
        [[[[]], []], []],
    )
    r = pl.Series([1])

    assert_series_equal(
        exec_op(l, r, operator.add),
        pl.Series("x", [[[[]], []], []], dtype=pl.List(pl.List(pl.List(pl.Int64)))),
    )

    l = pl.Series(  # noqa: E741
        "x",
        [[[[]], None], []],
    )
    r = pl.Series([1])

    assert_series_equal(
        exec_op(l, r, operator.add),
        pl.Series("x", [[[[]], None], []], dtype=pl.List(pl.List(pl.List(pl.Int64)))),
    )


@pytest.mark.parametrize("exec_op", EXEC_OP_COMBINATIONS)
def test_list_to_list_arithmetic_double_nesting_raises_error(
    exec_op: Callable[[pl.Series, pl.Series, Any], pl.Series],
) -> None:
    s = pl.Series(dtype=pl.List(pl.List(pl.Int32)))

    with pytest.raises(
        InvalidOperationError,
        match="cannot add two list columns with non-numeric inner types",
    ):
        exec_op(s, s, operator.add)


@pytest.mark.parametrize("exec_op", EXEC_OP_COMBINATIONS)
def test_list_add_height_mismatch(
    exec_op: Callable[[pl.Series, pl.Series, Any], pl.Series],
) -> None:
    s = pl.Series([[1], [2], [3]], dtype=pl.List(pl.Int32))

    # TODO: Make the error type consistently a ShapeError
    with pytest.raises(
        (ShapeError, InvalidOperationError),
        match="length",
    ):
        exec_op(s, pl.Series([1, 1]), operator.add)


@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.sub,
        operator.mul,
        operator.floordiv,
        operator.mod,
        operator.truediv,
    ],
)
@pytest.mark.parametrize("exec_op", EXEC_OP_COMBINATIONS)
@pytest.mark.slow
def test_list_date_to_numeric_arithmetic_raises_error(
    op: Callable[[Any], Any], exec_op: Callable[[pl.Series, pl.Series, Any], pl.Series]
) -> None:
    l = pl.Series([1], dtype=pl.Date)  # noqa: E741
    r = pl.Series([[1]], dtype=pl.List(pl.Int32))

    exec_op(l.to_physical(), r, op)

    # TODO(_): Ideally this always raises InvalidOperationError. The TypeError
    # is being raised by checks on the Python side that should be moved to Rust.
    with pytest.raises((InvalidOperationError, TypeError)):
        exec_op(l, r, op)


@pytest.mark.parametrize(
    ("expected", "expr", "column_names"),
    [
        ([[2, 4], [6]], lambda a, b: a + b, ("a", "a")),
        ([[0, 0], [0]], lambda a, b: a - b, ("a", "a")),
        ([[1, 4], [9]], lambda a, b: a * b, ("a", "a")),
        ([[1.0, 1.0], [1.0]], lambda a, b: a / b, ("a", "a")),
        ([[0, 0], [0]], lambda a, b: a % b, ("a", "a")),
        (
            [[3, 4], [7]],
            lambda a, b: a + b,
            ("a", "uint8"),
        ),
    ],
)
def test_list_arithmetic_same_size(
    expected: Any,
    expr: Callable[[pl.Series | pl.Expr, pl.Series | pl.Expr], pl.Series],
    column_names: tuple[str, str],
) -> None:
    df = pl.DataFrame(
        [
            pl.Series("a", [[1, 2], [3]]),
            pl.Series("uint8", [[2, 2], [4]], dtype=pl.List(pl.UInt8())),
            pl.Series("nested", [[[1, 2]], [[3]]]),
            pl.Series(
                "nested_uint8", [[[1, 2]], [[3]]], dtype=pl.List(pl.List(pl.UInt8()))
            ),
        ]
    )
    # Expr-based arithmetic:
    assert_frame_equal(
        df.select(expr(pl.col(column_names[0]), pl.col(column_names[1]))),
        pl.Series(column_names[0], expected).to_frame(),
    )
    # Direct arithmetic on the Series:
    assert_series_equal(
        expr(df[column_names[0]], df[column_names[1]]),
        pl.Series(column_names[0], expected),
    )


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([[1, 2, 3]], [[1, None, 5]], [[2, None, 8]]),
        ([[2], None, [5]], [None, [3], [2]], [None, None, [7]]),
    ],
)
def test_list_arithmetic_nulls(a: list[Any], b: list[Any], expected: list[Any]) -> None:
    series_a = pl.Series(a)
    series_b = pl.Series(b)
    series_expected = pl.Series(expected)

    # Same dtype:
    assert_series_equal(series_a + series_b, series_expected)

    # Different dtype:
    assert_series_equal(
        series_a._recursive_cast_to_dtype(pl.Int32())
        + series_b._recursive_cast_to_dtype(pl.Int64()),
        series_expected._recursive_cast_to_dtype(pl.Int64()),
    )


def test_list_arithmetic_error_cases() -> None:
    # Different series length:
    with pytest.raises(InvalidOperationError, match="different lengths"):
        _ = pl.Series("a", [[1, 2], [1, 2], [1, 2]]) / pl.Series("b", [[1, 2], [3, 4]])
    with pytest.raises(InvalidOperationError, match="different lengths"):
        _ = pl.Series("a", [[1, 2], [1, 2], [1, 2]]) / pl.Series("b", [[1, 2], None])

    # Different list length:
    with pytest.raises(ShapeError, match="lengths differed at index 0: 2 != 1"):
        _ = pl.Series("a", [[1, 2], [1, 2], [1, 2]]) / pl.Series("b", [[1]])

    with pytest.raises(ShapeError, match="lengths differed at index 0: 2 != 1"):
        _ = pl.Series("a", [[1, 2], [2, 3]]) / pl.Series("b", [[1], None])


@pytest.mark.parametrize("exec_op", EXEC_OP_COMBINATIONS)
def test_list_arithmetic_invalid_dtypes(
    exec_op: Callable[[pl.Series, pl.Series, Any], pl.Series],
) -> None:
    import operator as op

    a = pl.Series([[1, 2]])
    b = pl.Series(["hello"])

    # Wrong types:
    with pytest.raises(
        InvalidOperationError, match="add operation not supported for dtypes"
    ):
        exec_op(a, b, op.add)

    a = pl.Series("a", [[1]])
    b = pl.Series("b", [[[1]]])

    # list<->list is restricted to 1 level of nesting
    with pytest.raises(
        InvalidOperationError,
        match="cannot add two list columns with non-numeric inner types",
    ):
        exec_op(a, b, op.add)

    # Ensure dtype is validated to be `List` at all nesting levels instead of panicking.
    a = pl.Series([[[1]], [[1]]], dtype=pl.List(pl.Array(pl.Int64, 1)))
    b = pl.Series([1], dtype=pl.Int64)

    with pytest.raises(
        InvalidOperationError, match="dtype was not list on all nesting levels"
    ):
        exec_op(a, b, op.add)

    with pytest.raises(
        InvalidOperationError, match="dtype was not list on all nesting levels"
    ):
        exec_op(b, a, op.add)


@pytest.mark.parametrize(
    ("expected", "expr", "column_names"),
    [
        # All 5 arithmetic operations:
        ([[3, 4], [6]], lambda a, b: a + b, ("list", "int64")),
        ([[-1, 0], [0]], lambda a, b: a - b, ("list", "int64")),
        ([[2, 4], [9]], lambda a, b: a * b, ("list", "int64")),
        ([[0.5, 1.0], [1.0]], lambda a, b: a / b, ("list", "int64")),
        ([[1, 0], [0]], lambda a, b: a % b, ("list", "int64")),
        # Different types:
        (
            [[3, 4], [7]],
            lambda a, b: a + b,
            ("list", "uint8"),
        ),
        # Extra nesting + different types:
        (
            [[[3, 4]], [[8]]],
            lambda a, b: a + b,
            ("nested", "int64"),
        ),
        # Primitive numeric on the left; only addition and multiplication are
        # supported:
        ([[3, 4], [6]], lambda a, b: a + b, ("int64", "list")),
        ([[2, 4], [9]], lambda a, b: a * b, ("int64", "list")),
        # Primitive numeric on the left with different types:
        (
            [[3, 4], [7]],
            lambda a, b: a + b,
            ("uint8", "list"),
        ),
        (
            [[2, 4], [12]],
            lambda a, b: a * b,
            ("uint8", "list"),
        ),
    ],
)
def test_list_and_numeric_arithmetic_same_size(
    expected: Any,
    expr: Callable[[pl.Series | pl.Expr, pl.Series | pl.Expr], pl.Series],
    column_names: tuple[str, str],
) -> None:
    df = pl.DataFrame(
        [
            pl.Series("list", [[1, 2], [3]]),
            pl.Series("int64", [2, 3], dtype=pl.Int64()),
            pl.Series("uint8", [2, 4], dtype=pl.UInt8()),
            pl.Series("nested", [[[1, 2]], [[5]]]),
        ]
    )
    # Expr-based arithmetic:
    assert_frame_equal(
        df.select(expr(pl.col(column_names[0]), pl.col(column_names[1]))),
        pl.Series(column_names[0], expected).to_frame(),
    )
    # Direct arithmetic on the Series:
    assert_series_equal(
        expr(df[column_names[0]], df[column_names[1]]),
        pl.Series(column_names[0], expected),
    )


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        # Null on numeric on the right:
        ([[1, 2], [3]], [1, None], [[2, 3], [None]]),
        # Null on list on the left:
        ([[[1, 2]], [[3]]], [None, 1], [[[None, None]], [[4]]]),
        # Extra nesting:
        ([[[2, None]], [[3, 6]]], [3, 4], [[[5, None]], [[7, 10]]]),
    ],
)
def test_list_and_numeric_arithmetic_nulls(
    a: list[Any], b: list[Any], expected: list[Any]
) -> None:
    series_a = pl.Series(a)
    series_b = pl.Series(b)
    series_expected = pl.Series(expected, dtype=series_a.dtype)

    # Same dtype:
    assert_series_equal(series_a + series_b, series_expected)

    # Different dtype:
    assert_series_equal(
        series_a._recursive_cast_to_dtype(pl.Int32())
        + series_b._recursive_cast_to_dtype(pl.Int64()),
        series_expected._recursive_cast_to_dtype(pl.Int64()),
    )

    # Swap sides:
    assert_series_equal(series_b + series_a, series_expected)
    assert_series_equal(
        series_b._recursive_cast_to_dtype(pl.Int32())
        + series_a._recursive_cast_to_dtype(pl.Int64()),
        series_expected._recursive_cast_to_dtype(pl.Int64()),
    )


def test_list_and_numeric_arithmetic_error_cases() -> None:
    # Different series length:
    with pytest.raises(
        InvalidOperationError, match="series of different lengths: got 3 and 2"
    ):
        _ = pl.Series("a", [[1, 2], [3, 4], [5, 6]]) + pl.Series("b", [1, 2])
    with pytest.raises(
        InvalidOperationError, match="series of different lengths: got 3 and 2"
    ):
        _ = pl.Series("a", [[1, 2], [3, 4], [5, 6]]) / pl.Series("b", [1, None])

    # Wrong types:
    with pytest.raises(
        InvalidOperationError, match="add operation not supported for dtypes"
    ):
        _ = pl.Series("a", [[1, 2], [3, 4]]) + pl.Series("b", ["hello", "world"])


@pytest.mark.parametrize("broadcast", [True, False])
@pytest.mark.parametrize("dtype", [pl.Int64(), pl.Float64()])
def test_list_arithmetic_div_ops_zero_denominator(
    broadcast: bool, dtype: pl.DataType
) -> None:
    # Notes
    # * truediv (/) on integers upcasts to Float64
    # * Otherwise, we test floordiv (//) and module/rem (%)
    #   * On integers, 0-denominator is expected to output NULL
    #   * On floats, 0-denominator has different outputs, e.g. NaN, Inf, depending
    #     on a few factors (e.g. whether the numerator is also 0).

    s = pl.Series([[0], [1], [None], None]).cast(pl.List(dtype))

    n = 1 if broadcast else s.len()

    # list<->primitive

    # truediv
    assert_series_equal(
        pl.Series([1]).new_from_index(0, n) / s,
        pl.Series([[float("inf")], [1.0], [None], None], dtype=pl.List(pl.Float64)),
    )

    assert_series_equal(
        s / pl.Series([1]).new_from_index(0, n),
        pl.Series([[0.0], [1.0], [None], None], dtype=pl.List(pl.Float64)),
    )

    # floordiv
    assert_series_equal(
        pl.Series([1]).new_from_index(0, n) // s,
        (
            pl.Series([[None], [1], [None], None], dtype=s.dtype)
            if not dtype.is_float()
            else pl.Series([[float("inf")], [1.0], [None], None], dtype=s.dtype)
        ),
    )

    assert_series_equal(
        s // pl.Series([0]).new_from_index(0, n),
        (
            pl.Series([[None], [None], [None], None], dtype=s.dtype)
            if not dtype.is_float()
            else pl.Series(
                [[float("nan")], [float("inf")], [None], None], dtype=s.dtype
            )
        ),
    )

    # rem
    assert_series_equal(
        pl.Series([1]).new_from_index(0, n) % s,
        (
            pl.Series([[None], [0], [None], None], dtype=s.dtype)
            if not dtype.is_float()
            else pl.Series([[float("nan")], [0.0], [None], None], dtype=s.dtype)
        ),
    )

    assert_series_equal(
        s % pl.Series([0]).new_from_index(0, n),
        (
            pl.Series([[None], [None], [None], None], dtype=s.dtype)
            if not dtype.is_float()
            else pl.Series(
                [[float("nan")], [float("nan")], [None], None], dtype=s.dtype
            )
        ),
    )

    # list<->list

    # truediv
    assert_series_equal(
        pl.Series([[1]]).new_from_index(0, n) / s,
        pl.Series([[float("inf")], [1.0], [None], None], dtype=pl.List(pl.Float64)),
    )

    assert_series_equal(
        s / pl.Series([[0]]).new_from_index(0, n),
        pl.Series(
            [[float("nan")], [float("inf")], [None], None], dtype=pl.List(pl.Float64)
        ),
    )

    # floordiv
    assert_series_equal(
        pl.Series([[1]]).new_from_index(0, n) // s,
        (
            pl.Series([[None], [1], [None], None], dtype=s.dtype)
            if not dtype.is_float()
            else pl.Series([[float("inf")], [1.0], [None], None], dtype=s.dtype)
        ),
    )

    assert_series_equal(
        s // pl.Series([[0]]).new_from_index(0, n),
        (
            pl.Series([[None], [None], [None], None], dtype=s.dtype)
            if not dtype.is_float()
            else pl.Series(
                [[float("nan")], [float("inf")], [None], None], dtype=s.dtype
            )
        ),
    )

    # rem
    assert_series_equal(
        pl.Series([[1]]).new_from_index(0, n) % s,
        (
            pl.Series([[None], [0], [None], None], dtype=s.dtype)
            if not dtype.is_float()
            else pl.Series([[float("nan")], [0.0], [None], None], dtype=s.dtype)
        ),
    )

    assert_series_equal(
        s % pl.Series([[0]]).new_from_index(0, n),
        (
            pl.Series([[None], [None], [None], None], dtype=s.dtype)
            if not dtype.is_float()
            else pl.Series(
                [[float("nan")], [float("nan")], [None], None], dtype=s.dtype
            )
        ),
    )


def test_list_to_primitive_arithmetic() -> None:
    # Input data
    # * List type: List(List(List(Int16))) (triple-nested)
    # * Numeric type: Int32
    #
    # Tests run
    #                      Broadcast  Operation
    #                      | L | R |
    # * list<->primitive   |   |   |  floor_div
    # * primitive<->list   |   |   |  floor_div
    # * list<->primitive   |   | * |  subtract
    # * primitive<->list   | * |   |  subtract
    # * list<->primitive   | * |   |  subtract
    # * primitive<->list   |   | * |  subtract
    #
    # Notes
    # * In floor_div, we check that results from a 0 denominator are masked out
    # * We choose floor_div and subtract as they emit different results when
    #   sides are swapped

    # Create some non-zero start offsets and masked out rows.
    lhs = (
        pl.Series(
            [
                [[[None, None, None, None, None]]],  # sliced out
                # Nulls at every level XO
                [[[3, 7]], [[-3], [None], [], [], None], [], None],
                [[[1, 2, 3, 4, 5]]],  # masked out
                [[[3, 7]], [[0], [None], [], [], None]],
                [[[3, 7]]],
            ],
            dtype=pl.List(pl.List(pl.List(pl.Int16))),
        )
        .slice(1)
        .to_frame()
        .select(pl.when(pl.int_range(pl.len()) != 1).then(pl.first()))
        .to_series()
    )

    # Note to reader: This is what our LHS looks like
    assert_series_equal(
        lhs,
        pl.Series(
            [
                [[[3, 7]], [[-3], [None], [], [], None], [], None],
                None,
                [[[3, 7]], [[0], [None], [], [], None]],
                [[[3, 7]]],
            ],
            dtype=pl.List(pl.List(pl.List(pl.Int16))),
        ),
    )

    class _:
        # Floor div, no broadcasting
        rhs = pl.Series([5, 1, 0, None], dtype=pl.Int32)

        assert len(lhs) == len(rhs)

        expect = pl.Series(
            [
                [[[0, 1]], [[-1], [None], [], [], None], [], None],
                None,
                [[[None, None]], [[None], [None], [], [], None]],
                [[[None, None]]],
            ],
            dtype=pl.List(pl.List(pl.List(pl.Int32))),
        )

        out = (
            pl.select(l=lhs, r=rhs)
            .select(pl.col("l") // pl.col("r"))
            .to_series()
            .alias("")
        )

        assert_series_equal(out, expect)

        # Flipped

        expect = pl.Series(  # noqa: PIE794
            [
                [[[1, 0]], [[-2], [None], [], [], None], [], None],
                None,
                [[[0, 0]], [[None], [None], [], [], None]],
                [[[None, None]]],
            ],
            dtype=pl.List(pl.List(pl.List(pl.Int32))),
        )

        out = (  # noqa: PIE794
            pl.select(l=lhs, r=rhs)
            .select(pl.col("r") // pl.col("l"))
            .to_series()
            .alias("")
        )

        assert_series_equal(out, expect)

    class _:  # type: ignore[no-redef]
        # Subtraction with broadcasting
        rhs = pl.Series([1], dtype=pl.Int32)

        expect = pl.Series(
            [
                [[[2, 6]], [[-4], [None], [], [], None], [], None],
                None,
                [[[2, 6]], [[-1], [None], [], [], None]],
                [[[2, 6]]],
            ],
            dtype=pl.List(pl.List(pl.List(pl.Int32))),
        )

        out = pl.select(l=lhs).select(pl.col("l") - rhs).to_series().alias("")

        assert_series_equal(out, expect)

        # Flipped

        expect = pl.Series(  # noqa: PIE794
            [
                [[[-2, -6]], [[4], [None], [], [], None], [], None],
                None,
                [[[-2, -6]], [[1], [None], [], [], None]],
                [[[-2, -6]]],
            ],
            dtype=pl.List(pl.List(pl.List(pl.Int32))),
        )

        out = pl.select(l=lhs).select(rhs - pl.col("l")).to_series().alias("")  # noqa: PIE794

        assert_series_equal(out, expect)

    # Test broadcasting of the list side
    lhs = lhs.slice(2, 1)
    # Note to reader: This is what our LHS looks like
    assert_series_equal(
        lhs,
        pl.Series(
            [
                [[[3, 7]], [[0], [None], [], [], None]],
            ],
            dtype=pl.List(pl.List(pl.List(pl.Int16))),
        ),
    )

    assert len(lhs) == 1

    class _:  # type: ignore[no-redef]
        rhs = pl.Series([1, 2, 3, None, 5], dtype=pl.Int32)

        expect = pl.Series(
            [
                [[[2, 6]], [[-1], [None], [], [], None]],
                [[[1, 5]], [[-2], [None], [], [], None]],
                [[[0, 4]], [[-3], [None], [], [], None]],
                [[[None, None]], [[None], [None], [], [], None]],
                [[[-2, 2]], [[-5], [None], [], [], None]],
            ],
            dtype=pl.List(pl.List(pl.List(pl.Int32))),
        )

        out = pl.select(r=rhs).select(lhs - pl.col("r")).to_series().alias("")

        assert_series_equal(out, expect)

        # Flipped

        expect = pl.Series(  # noqa: PIE794
            [
                [[[-2, -6]], [[1], [None], [], [], None]],
                [[[-1, -5]], [[2], [None], [], [], None]],
                [[[0, -4]], [[3], [None], [], [], None]],
                [[[None, None]], [[None], [None], [], [], None]],
                [[[2, -2]], [[5], [None], [], [], None]],
            ],
            dtype=pl.List(pl.List(pl.List(pl.Int32))),
        )

        out = pl.select(r=rhs).select(pl.col("r") - lhs).to_series().alias("")  # noqa: PIE794

        assert_series_equal(out, expect)
