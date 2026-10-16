from __future__ import annotations

import operator
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.testing.asserts.frame import assert_frame_equal

if TYPE_CHECKING:
    from collections.abc import Callable


def test_add_parse_str_input_as_literal() -> None:
    df = pl.DataFrame({"a": ["x", "y"]})
    result = df.select(pl.col("a") + "b")
    expected = pl.DataFrame({"a": ["xb", "yb"]})
    assert_frame_equal(result, expected)


def test_truediv_parse_str_input_as_col_name() -> None:
    df = pl.DataFrame({"a": [10, 12], "b": [5, 4]})
    result = df.select(pl.col("a") / "b")
    expected = pl.DataFrame({"a": [2, 3]}, schema={"a": pl.Float64})
    assert_frame_equal(result, expected)


class _Timestamp:
    """Foreign type that knows how to compare itself against expressions."""

    def __init__(self, dt: datetime) -> None:
        self.dt = dt

    def __eq__(self, other: object) -> Any:
        if isinstance(other, pl.Expr):
            return other == self.dt
        return NotImplemented

    def __ne__(self, other: object) -> Any:
        if isinstance(other, pl.Expr):
            return other != self.dt
        return NotImplemented

    def __lt__(self, other: object) -> Any:
        if isinstance(other, pl.Expr):
            return other > self.dt
        return NotImplemented

    def __le__(self, other: object) -> Any:
        if isinstance(other, pl.Expr):
            return other >= self.dt
        return NotImplemented

    def __gt__(self, other: object) -> Any:
        if isinstance(other, pl.Expr):
            return other < self.dt
        return NotImplemented

    def __ge__(self, other: object) -> Any:
        if isinstance(other, pl.Expr):
            return other <= self.dt
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (operator.eq, [False, True, False]),
        (operator.ne, [True, False, True]),
        (operator.lt, [True, False, False]),
        (operator.le, [True, True, False]),
        (operator.gt, [False, False, True]),
        (operator.ge, [False, True, True]),
    ],
)
def test_comparison_defers_to_foreign_type_29977(
    op: Callable[[Any, Any], Any], expected: list[bool]
) -> None:
    df = pl.DataFrame(
        {"t": [datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2022, 1, 1)]}
    )
    expr = op(pl.col("t"), _Timestamp(datetime(2021, 1, 1)))
    assert isinstance(expr, pl.Expr)
    assert df.select(expr).to_series().to_list() == expected


@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        operator.floordiv,
        operator.mod,
        operator.pow,
        operator.and_,
        operator.or_,
        operator.xor,
    ],
)
def test_arithmetic_defers_to_foreign_type_29977(op: Callable[[Any, Any], Any]) -> None:
    class Number:
        def __init__(self, value: int) -> None:
            self.value = value

    def reflected(self: Number, other: Any) -> Any:
        if isinstance(other, pl.Expr):
            return op(other, self.value)
        return NotImplemented

    reflected_name = f"__r{op.__name__.rstrip('_')}__"
    setattr(Number, reflected_name, reflected)

    df = pl.DataFrame({"a": [1, 2, 3]})
    result = df.select(op(pl.col("a"), Number(2)))
    expected = df.select(op(pl.col("a"), 2))
    assert_frame_equal(result, expected)


class _Unsupported:
    """Foreign type that does not know how to compare itself against expressions."""


class _UnsupportedWithLt:
    """Like :class:`_Unsupported` but has some dunders."""

    def __lt__(self, other: object) -> bool:
        return NotImplemented  # type: ignore[no-any-return]

    __hash__ = None  # type: ignore[assignment]


@pytest.mark.parametrize("value", [_Unsupported(), _UnsupportedWithLt()])
@pytest.mark.parametrize(
    "op",
    [
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
        operator.add,
        operator.mul,
        operator.and_,
    ],
)
def test_binop_unsupported_type_raises(
    value: Any, op: Callable[[Any, Any], Any]
) -> None:
    with pytest.raises(TypeError, match=r"not supported|unsupported operand"):
        op(pl.col("a"), value)
    with pytest.raises(TypeError, match=r"not supported|unsupported operand"):
        op(value, pl.col("a"))


@pytest.mark.parametrize("op", [operator.eq, operator.lt, operator.add])
@pytest.mark.parametrize(
    ("value", "match"),
    [
        ([1, "x"], "unexpected value while building Series"),
        ({1: 2}, "cannot create expression literal for value of type dict"),
    ],
)
def test_binop_supported_type_conversion_error_is_raised(
    op: Callable[[Any, Any], Any], value: Any, match: str
) -> None:
    # Errors converting a value of a supported type
    # are not deferred to the other operand.
    with pytest.raises(TypeError, match=match):
        op(pl.col("a"), value)
