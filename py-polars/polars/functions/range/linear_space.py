from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, overload

from polars import functions as F
from polars._utils.parse import parse_into_expression
from polars._utils.wrap import wrap_expr

with contextlib.suppress(ImportError):  # Module not available when building docs
    import polars.polars as plr

from typing import Literal

if TYPE_CHECKING:
    from polars import Expr, Series
    from polars._typing import ClosedInterval, IntoExpr, NumericLiteral, TemporalLiteral


@overload
def linear_space(
    start: NumericLiteral | TemporalLiteral | IntoExpr,
    end: NumericLiteral | TemporalLiteral | IntoExpr,
    num_samples: int,
    *,
    closed: ClosedInterval = ...,
    eager: Literal[False] = ...,
) -> Expr: ...


@overload
def linear_space(
    start: NumericLiteral | TemporalLiteral | IntoExpr,
    end: NumericLiteral | TemporalLiteral | IntoExpr,
    num_samples: int,
    *,
    closed: ClosedInterval = ...,
    eager: Literal[True],
) -> Series: ...


@overload
def linear_space(
    start: NumericLiteral | TemporalLiteral | IntoExpr,
    end: NumericLiteral | TemporalLiteral | IntoExpr,
    num_samples: int,
    *,
    closed: ClosedInterval = ...,
    eager: bool,
) -> Expr | Series: ...


def linear_space(
    start: NumericLiteral | TemporalLiteral | IntoExpr,
    end: NumericLiteral | TemporalLiteral | IntoExpr,
    num_samples: int,
    *,
    closed: ClosedInterval = "both",
    eager: bool = False,
) -> Expr | Series:
    """Linearly-spaced elements."""
    start = parse_into_expression(start)
    end = parse_into_expression(end)
    result = wrap_expr(plr.linear_space(start, end, num_samples, closed))

    if eager:
        return F.select(result).to_series()

    return result
