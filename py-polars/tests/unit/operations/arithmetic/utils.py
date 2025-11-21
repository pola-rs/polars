from __future__ import annotations

from typing import Any

import polars as pl


def exec_op_with_series(lhs: pl.Series, rhs: pl.Series, op: Any) -> pl.Series:
    v: pl.Series = op(lhs, rhs)
    return v


def exec_op_with_expr(lhs: pl.Series, rhs: pl.Series, op: Any) -> pl.Series:
    return pl.select(lhs).lazy().select(op(pl.first(), rhs)).collect().to_series()


def exec_op_with_expr_no_type_coercion(
    lhs: pl.Series, rhs: pl.Series, op: Any
) -> pl.Series:
    optimizations = pl.QueryOptFlags()
    optimizations._pyoptflags.type_coercion = False
    return (
        pl.select(lhs)
        .lazy()
        .select(op(pl.first(), rhs))
        .collect(optimizations=optimizations)
        .to_series()
    )


BROADCAST_LEN = 3


def broadcast_left(
    l: pl.Series,  # noqa: E741
    r: pl.Series,
    o: pl.Series,
) -> tuple[pl.Series, pl.Series, pl.Series]:
    return l.new_from_index(0, BROADCAST_LEN), r, o.new_from_index(0, BROADCAST_LEN)


def broadcast_right(
    l: pl.Series,  # noqa: E741
    r: pl.Series,
    o: pl.Series,
) -> tuple[pl.Series, pl.Series, pl.Series]:
    return l, r.new_from_index(0, BROADCAST_LEN), o.new_from_index(0, BROADCAST_LEN)


def broadcast_both(
    l: pl.Series,  # noqa: E741
    r: pl.Series,
    o: pl.Series,
) -> tuple[pl.Series, pl.Series, pl.Series]:
    return (
        l.new_from_index(0, BROADCAST_LEN),
        r.new_from_index(0, BROADCAST_LEN),
        o.new_from_index(0, BROADCAST_LEN),
    )


def broadcast_none(
    l: pl.Series,  # noqa: E741
    r: pl.Series,
    o: pl.Series,
) -> tuple[pl.Series, pl.Series, pl.Series]:
    return l, r, o


BROADCAST_SERIES_COMBINATIONS = [
    broadcast_left,
    broadcast_right,
    broadcast_both,
    broadcast_none,
]

EXEC_OP_COMBINATIONS = [
    exec_op_with_series,
    exec_op_with_expr,
    exec_op_with_expr_no_type_coercion,
]
