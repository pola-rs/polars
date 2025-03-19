from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import pytest

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.testing import assert_series_equal
from tests.unit.operations.arithmetic.utils import (
    BROADCAST_SERIES_COMBINATIONS,
    EXEC_OP_COMBINATIONS,
)

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


@pytest.mark.parametrize(
    "array_side", ["left", "left3", "both", "both3", "right3", "right", "none"]
)
@pytest.mark.parametrize(
    "broadcast_series",
    BROADCAST_SERIES_COMBINATIONS,
)
@pytest.mark.parametrize("exec_op", EXEC_OP_COMBINATIONS)
@pytest.mark.slow
def test_array_arithmetic_values(
    array_side: str,
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

    def materialize_array(v: Any) -> pl.Series:
        return pl.Series(
            [[None, v, None]],
            dtype=pl.Array(dtype, 3),
        )

    def materialize_array3(v: Any) -> pl.Series:
        return pl.Series(
            [[[[None, v], None], None]],
            dtype=pl.Array(pl.Array(pl.Array(dtype, 2), 2), 2),
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
            "left":   materialize_array,
            "left3":  materialize_array3,
            "both":   materialize_array,
            "both3":  materialize_array3,
            "right":  materialize_primitive,
            "right3": materialize_primitive,
            "none":   materialize_primitive,
        }[array_side](l)  # fmt: skip

        dtype = dtypes[1]
        r = {
            "left":   materialize_primitive,
            "left3":  materialize_primitive,
            "both":   materialize_array,
            "both3":  materialize_array3,
            "right":  materialize_array,
            "right3": materialize_array3,
            "none":   materialize_primitive,
        }[array_side](r)  # fmt: skip

        dtype = dtypes[2]
        o = {
            "left":   materialize_array,
            "left3":  materialize_array3,
            "both":   materialize_array,
            "both3":  materialize_array3,
            "right":  materialize_array,
            "right3": materialize_array3,
            "none":   materialize_primitive,
        }[array_side](o)  # fmt: skip

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
    if array_side != "none":
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
    if array_side != "none":
        assert_series_equal(exec_op(l, r, op.floordiv), o)
    assert_series_equal(exec_op(l, r, op.mod), o)

    dtypes = [pl.Null, pl.UInt8, pl.Float64]
    l, r, o = materialize_series(None, 3, None)  # noqa: E741
    assert_series_equal(exec_op(l, r, op.truediv), o)


@pytest.mark.parametrize(
    ("lhs_dtype", "rhs_dtype", "expected_dtype"),
    [
        (pl.Array(pl.Int64, 2), pl.Int64, pl.Array(pl.Float64, 2)),
        (pl.Array(pl.Float32, 2), pl.Float32, pl.Array(pl.Float32, 2)),
        (pl.Array(pl.Duration("us"), 2), pl.Int64, pl.Array(pl.Duration("us"), 2)),
    ],
)
def test_array_truediv_schema(
    lhs_dtype: PolarsDataType, rhs_dtype: PolarsDataType, expected_dtype: PolarsDataType
) -> None:
    schema = {"lhs": lhs_dtype, "rhs": rhs_dtype}
    df = pl.DataFrame({"lhs": [[None, 10]], "rhs": 2}, schema=schema)
    result = df.lazy().select(pl.col("lhs").truediv("rhs")).collect_schema()["lhs"]
    assert result == expected_dtype


def test_array_literal_broadcast() -> None:
    df = pl.DataFrame({"A": [[0.1, 0.2], [0.3, 0.4]]}).cast(pl.Array(float, 2))

    lit = pl.lit([3, 5], pl.Array(float, 2))
    assert df.select(
        mul=pl.all() * lit,
        div=pl.all() / lit,
        add=pl.all() + lit,
        sub=pl.all() - lit,
        div_=lit / pl.all(),
        add_=lit + pl.all(),
        sub_=lit - pl.all(),
        mul_=lit * pl.all(),
    ).to_dict(as_series=False) == {
        "mul": [[0.30000000000000004, 1.0], [0.8999999999999999, 2.0]],
        "div": [[0.03333333333333333, 0.04], [0.09999999999999999, 0.08]],
        "add": [[3.1, 5.2], [3.3, 5.4]],
        "sub": [[-2.9, -4.8], [-2.7, -4.6]],
        "div_": [[30.0, 25.0], [10.0, 12.5]],
        "add_": [[3.1, 5.2], [3.3, 5.4]],
        "sub_": [[2.9, 4.8], [2.7, 4.6]],
        "mul_": [[0.30000000000000004, 1.0], [0.8999999999999999, 2.0]],
    }


def test_array_arith_double_nested_shape() -> None:
    # Ensure the implementation doesn't just naively add the leaf arrays without
    # checking the dimension. In this example both arrays have the leaf stride as
    # 6, however one is (3, 2) while the other is (2, 3).
    a = pl.Series([[[1, 1], [1, 1], [1, 1]]], dtype=pl.Array(pl.Array(pl.Int64, 2), 3))
    b = pl.Series([[[1, 1, 1], [1, 1, 1]]], dtype=pl.Array(pl.Array(pl.Int64, 3), 2))

    with pytest.raises(InvalidOperationError, match="differing dtypes"):
        a + b


@pytest.mark.parametrize("exec_op", EXEC_OP_COMBINATIONS)
@pytest.mark.parametrize(
    "broadcast_series",
    BROADCAST_SERIES_COMBINATIONS,
)
@pytest.mark.slow
def test_array_numeric_op_validity_combination(
    broadcast_series: Callable[
        [pl.Series, pl.Series, pl.Series], tuple[pl.Series, pl.Series, pl.Series]
    ],
    exec_op: Callable[[pl.Series, pl.Series, Any], pl.Series],
) -> None:
    import operator as op

    array_dtype = pl.Array(pl.Int64, 1)

    a = pl.Series("a", [[1], [2], None, [None], [11], [1111]], dtype=array_dtype)
    b = pl.Series("b", [[1], [3], [11], [1111], None, [None]], dtype=array_dtype)
    # expected result
    e = pl.Series("a", [[2], [5], None, [None], None, [None]], dtype=array_dtype)

    assert_series_equal(
        exec_op(a, b, op.add),
        e,
    )

    a = pl.Series("a", [[1]], dtype=array_dtype)
    b = pl.Series("b", [None], dtype=pl.Int64)
    e = pl.Series("a", [[None]], dtype=array_dtype)

    a, b, e = broadcast_series(a, b, e)
    assert_series_equal(exec_op(a, b, op.add), e)

    a = pl.Series("a", [None], dtype=array_dtype)
    b = pl.Series("b", [1], dtype=pl.Int64)
    e = pl.Series("a", [None], dtype=array_dtype)

    a, b, e = broadcast_series(a, b, e)
    assert_series_equal(exec_op(a, b, op.add), e)

    a = pl.Series("a", [None], dtype=array_dtype)
    b = pl.Series("b", [0], dtype=pl.Int64)
    e = pl.Series("a", [None], dtype=array_dtype)

    a, b, e = broadcast_series(a, b, e)
    assert_series_equal(exec_op(a, b, op.floordiv), e)

    # >1 level nested array
    a = pl.Series(
        # row 1: [ [1, NULL], NULL ]
        # row 2: NULL
        [[[1, None], None], None],
        dtype=pl.Array(pl.Array(pl.Int64, 2), 2),
    )
    b = pl.Series(
        [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
        dtype=pl.Array(pl.Array(pl.Int64, 2), 2),
    )
    e = a  # added 0
    assert_series_equal(exec_op(a, b, op.add), e)


def test_array_elementwise_arithmetic_19682() -> None:
    dt = pl.Array(pl.Int64, (2, 3))

    a = pl.Series("a", [[[1, 2, 3], [4, 5, 6]]], dt)
    sc = pl.Series("a", [1])
    zfa = pl.Series("a", [[]], pl.Array(pl.Int64, 0))

    assert_series_equal(a + a, pl.Series("a", [[[2, 4, 6], [8, 10, 12]]], dt))
    assert_series_equal(a + sc, pl.Series("a", [[[2, 3, 4], [5, 6, 7]]], dt))
    assert_series_equal(sc + a, pl.Series("a", [[[2, 3, 4], [5, 6, 7]]], dt))
    assert_series_equal(zfa + zfa, pl.Series("a", [[]], pl.Array(pl.Int64, 0)))


@pytest.mark.parametrize("exec_op", EXEC_OP_COMBINATIONS)
def test_array_add_supertype(
    exec_op: Callable[[pl.Series, pl.Series, Any], pl.Series],
) -> None:
    import operator as op

    a = pl.Series("a", [[1], [2]], dtype=pl.Array(pl.Int8, 1))
    b = pl.Series("b", [[1], [999]], dtype=pl.Array(pl.Int64, 1))

    assert_series_equal(
        exec_op(a, b, op.add),
        pl.Series("a", [[2], [1001]], dtype=pl.Array(pl.Int64, 1)),
    )


@pytest.mark.parametrize("exec_op", EXEC_OP_COMBINATIONS)
def test_array_arithmetic_dtype_mismatch(
    exec_op: Callable[[pl.Series, pl.Series, Any], pl.Series],
) -> None:
    import operator as op

    a = pl.Series("a", [[1], [2]], dtype=pl.Array(pl.Int64, 1))
    b = pl.Series("b", [[1, 1], [999, 999]], dtype=pl.Array(pl.Int64, 2))

    with pytest.raises(InvalidOperationError, match="differing dtypes"):
        exec_op(a, b, op.add)

    a = pl.Series([[[1]], [[1]]], dtype=pl.Array(pl.List(pl.Int64), 1))
    b = pl.Series([1], dtype=pl.Int64)

    with pytest.raises(
        InvalidOperationError, match="dtype was not array on all nesting levels"
    ):
        exec_op(a, a, op.add)

    with pytest.raises(
        InvalidOperationError, match="dtype was not array on all nesting levels"
    ):
        exec_op(a, b, op.add)

    with pytest.raises(
        InvalidOperationError, match="dtype was not array on all nesting levels"
    ):
        exec_op(b, a, op.add)
