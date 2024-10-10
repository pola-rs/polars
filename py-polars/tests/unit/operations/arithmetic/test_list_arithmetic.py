from __future__ import annotations

import operator
from typing import Any, Callable

import pytest

import polars as pl
from polars.exceptions import InvalidOperationError, ShapeError
from polars.testing import assert_series_equal


def exec_op_with_series(lhs: pl.Series, rhs: pl.Series, op: Any) -> pl.Series:
    v: pl.Series = op(lhs, rhs)
    return v


def build_expr_op_exec(
    type_coercion: bool,
) -> Callable[[pl.Series, pl.Series, Any], pl.Series]:
    def func(lhs: pl.Series, rhs: pl.Series, op: Any) -> pl.Series:
        return (
            pl.select(lhs)
            .lazy()
            .select(op(pl.first(), rhs))
            .collect(type_coercion=type_coercion)
            .to_series()
        )

    return func


def build_series_broadcaster(
    side: str,
) -> Callable[
    [pl.Series, pl.Series, pl.Series], tuple[pl.Series, pl.Series, pl.Series]
]:
    length = 3

    if side == "left":

        def func(
            l: pl.Series,  # noqa: E741
            r: pl.Series,
            o: pl.Series,
        ) -> tuple[pl.Series, pl.Series, pl.Series]:
            return l.new_from_index(0, length), r, o.new_from_index(0, length)
    elif side == "right":

        def func(
            l: pl.Series,  # noqa: E741
            r: pl.Series,
            o: pl.Series,
        ) -> tuple[pl.Series, pl.Series, pl.Series]:
            return l, r.new_from_index(0, length), o.new_from_index(0, length)
    elif side == "both":

        def func(
            l: pl.Series,  # noqa: E741
            r: pl.Series,
            o: pl.Series,
        ) -> tuple[pl.Series, pl.Series, pl.Series]:
            return (
                l.new_from_index(0, length),
                r.new_from_index(0, length),
                o.new_from_index(0, length),
            )
    elif side == "none":

        def func(
            l: pl.Series,  # noqa: E741
            r: pl.Series,
            o: pl.Series,
        ) -> tuple[pl.Series, pl.Series, pl.Series]:
            return l, r, o
    else:
        raise ValueError(side)

    return func


BROADCAST_SERIES_COMBINATIONS = [
    build_series_broadcaster("left"),
    build_series_broadcaster("right"),
    build_series_broadcaster("both"),
    build_series_broadcaster("none"),
]

EXEC_OP_COMBINATIONS = [
    exec_op_with_series,
    build_expr_op_exec(True),
    build_expr_op_exec(False),
]


@pytest.mark.parametrize(
    "list_side", ["left", "left3", "both", "right3", "right", "none"]
)
@pytest.mark.parametrize(
    "broadcast_series",
    BROADCAST_SERIES_COMBINATIONS,
)
@pytest.mark.parametrize("exec_op", EXEC_OP_COMBINATIONS)
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
