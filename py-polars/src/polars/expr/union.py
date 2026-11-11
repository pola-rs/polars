from __future__ import annotations

from typing import TYPE_CHECKING

from polars._utils.various import _NamespaceSuggestMixin
from polars._utils.wrap import wrap_expr

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from polars import Expr


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
