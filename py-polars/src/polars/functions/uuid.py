from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from polars import functions as F
from polars._utils.wrap import wrap_expr

if TYPE_CHECKING:
    from polars import Expr, Series


@overload
def uuid4(n: int | Expr, *, eager: Literal[False] = ...) -> Expr: ...


@overload
def uuid4(n: int | Expr, *, eager: Literal[True]) -> Series: ...


@overload
def uuid4(n: int | Expr, *, eager: bool) -> Expr | Series: ...


def uuid4(n: int | Expr, *, eager: bool = False) -> Expr | Series:
    """
    Generate `n` random UUID version 4 values.

    Parameters
    ----------
    n
        Number of UUID values to generate. This is explicit so expression and eager
        use have identical row-count semantics.
    eager
        Evaluate immediately and return a Series. If false, return an expression.

    Returns
    -------
    Expr or Series
        UUID values with dtype :class:`UUID`.

    Examples
    --------
    >>> ids = pl.uuid4(2, eager=True)
    >>> ids.dtype
    UUID
    >>> ids.uuid.version().to_list()
    [4, 4]
    """
    source = F.int_range(0, n)
    expr = wrap_expr(source._pyexpr.uuid_generate_v4()).alias("uuid4")
    return F.select(expr).to_series() if eager else expr


@overload
def uuid7(n: int | Expr, *, eager: Literal[False] = ...) -> Expr: ...


@overload
def uuid7(n: int | Expr, *, eager: Literal[True]) -> Series: ...


@overload
def uuid7(n: int | Expr, *, eager: bool) -> Expr | Series: ...


def uuid7(n: int | Expr, *, eager: bool = False) -> Expr | Series:
    """
    Generate `n` process-monotonic, time-ordered UUID version 7 values.

    Parameters
    ----------
    n
        Number of UUID values to generate. This is explicit so expression and eager
        use have identical row-count semantics.
    eager
        Evaluate immediately and return a Series. If false, return an expression.

    Returns
    -------
    Expr or Series
        UUID values with dtype :class:`UUID`.

    Examples
    --------
    >>> ids = pl.uuid7(2, eager=True)
    >>> ids.uuid.version().to_list()
    [7, 7]
    >>> ids.is_sorted()
    True
    """
    source = F.int_range(0, n)
    expr = wrap_expr(source._pyexpr.uuid_generate_v7()).alias("uuid7")
    return F.select(expr).to_series() if eager else expr
