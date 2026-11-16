from __future__ import annotations

from typing import TYPE_CHECKING, Any

from polars._utils.parse import parse_into_expression
from polars._utils.various import _NamespaceSuggestMixin
from polars._utils.wrap import wrap_expr

# Tells "no default" apart from `default=None`, which is a null default.
_NO_DEFAULT: Any = object()

if TYPE_CHECKING:
    from collections.abc import Mapping

    from polars import Expr
    from polars._typing import IntoExpr


def _parse_variant_entries(
    entries: Mapping[str, Any] | None, named: dict[str, Any]
) -> dict[str, Any]:
    """Merge variant entries given as a mapping and as keyword arguments."""
    out = dict(entries) if entries is not None else {}
    for name, value in named.items():
        if name in out:
            msg = f"variant {name!r} is given more than once"
            raise ValueError(msg)
        out[name] = value
    return out


class ExprUnionNameSpace(_NamespaceSuggestMixin):
    """Namespace for union related expressions."""

    _accessor = "union"

    def __init__(self, expr: Expr) -> None:
        self._pyexpr = expr._pyexpr

    def tag(self) -> Expr:
        """
        Get the variant of every row.

        Returns an :class:`Enum` of the variant names, in variant order. Null rows
        give null.

        .. engine-support:: in-memory, streaming

        Examples
        --------
        >>> dtype = pl.Union({"i": pl.Int64, "s": pl.String})
        >>> df = pl.DataFrame({"x": [1, None], "y": ["a", "b"]})
        >>> df.select(
        ...     u=pl.when(pl.col("x").is_not_null())
        ...     .then(pl.union_value(dtype, i=pl.col("x")))
        ...     .otherwise(pl.union_value(dtype, s=pl.col("y")))
        ... ).select(pl.col("u").union.tag())
        shape: (2, 1)
        ┌──────┐
        │ u    │
        │ ---  │
        │ enum │
        ╞══════╡
        │ i    │
        │ s    │
        └──────┘
        """
        return wrap_expr(self._pyexpr.union_tag())

    def field(self, name: str) -> Expr:
        """
        Get the values of one variant.

        Rows of other variants and null rows give null.

        .. engine-support:: in-memory, streaming

        Parameters
        ----------
        name
            The name of the variant.

        Examples
        --------
        >>> dtype = pl.Union({"i": pl.Int64, "s": pl.String})
        >>> df = pl.DataFrame({"x": [1, None], "y": ["a", "b"]})
        >>> df.select(
        ...     u=pl.when(pl.col("x").is_not_null())
        ...     .then(pl.union_value(dtype, i=pl.col("x")))
        ...     .otherwise(pl.union_value(dtype, s=pl.col("y")))
        ... ).select(pl.col("u").union.field("i"))
        shape: (2, 1)
        ┌──────┐
        │ u    │
        │ ---  │
        │ i64  │
        ╞══════╡
        │ 1    │
        │ null │
        └──────┘
        """
        return wrap_expr(self._pyexpr.union_field(name))

    def match(
        self,
        arms: Mapping[str, IntoExpr] | None = None,
        /,
        *,
        default: IntoExpr = _NO_DEFAULT,
        **named_arms: IntoExpr,
    ) -> Expr:
        """
        Evaluate an expression that depends on the variant of each row.

        Inside the arm of a variant, :func:`element` is the value of that variant.
        Arms can also refer to other columns.

        .. engine-support:: in-memory, streaming

        Parameters
        ----------
        arms
            A mapping from variant names to arms, for names that are not valid
            keyword arguments.
        default
            The value for variants without an arm. Required unless every variant
            has an arm; `None` gives null.
        **named_arms
            The arm of each variant, by variant name.

        Notes
        -----
        The arms and the default must have the same data type, except that null
        and literals such as `0` take the data type of the other arms. They must
        be elementwise. A null row gives null.

        Examples
        --------
        >>> dtype = pl.Union({"i": pl.Int64, "s": pl.String})
        >>> df = pl.DataFrame({"x": [1, None, 3], "y": ["a", "bc", "d"]})
        >>> df = df.select(
        ...     u=pl.when(pl.col("x").is_not_null())
        ...     .then(pl.union_value(dtype, i=pl.col("x")))
        ...     .otherwise(pl.union_value(dtype, s=pl.col("y")))
        ... )
        >>> df.select(
        ...     pl.col("u").union.match(
        ...         i=pl.element() * 10,
        ...         s=pl.element().str.len_chars().cast(pl.Int64),
        ...     )
        ... )
        shape: (3, 1)
        ┌─────┐
        │ u   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 10  │
        │ 2   │
        │ 30  │
        └─────┘

        Variants without an arm take the default:

        >>> df.select(pl.col("u").union.match(i=pl.element(), default=0))
        shape: (3, 1)
        ┌─────┐
        │ u   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 1   │
        │ 0   │
        │ 3   │
        └─────┘
        """
        entries = _parse_variant_entries(arms, named_arms)
        pyarms = [(name, parse_into_expression(arm)) for name, arm in entries.items()]
        pydefault = (
            None
            if default is _NO_DEFAULT
            else parse_into_expression(default, str_as_lit=False)
        )
        return wrap_expr(self._pyexpr.union_match(pyarms, pydefault))
