from __future__ import annotations

from typing import TYPE_CHECKING

from polars._utils.wrap import wrap_expr

if TYPE_CHECKING:
    from polars import Expr


class ExprMapNameSpace:
    """Namespace for map related expressions."""

    _accessor = "map"

    def __init__(self, expr: Expr) -> None:
        self._pyexpr = expr._pyexpr

    def entries(self) -> Expr:
        """
        Convert the `Map` to a `List` of `Struct` entries.

        Each entry is a `Struct` with a `key` and a `value` field. Entry order is
        preserved. The inverse of :meth:`Expr.list.to_map`.

        .. engine-support:: in-memory, streaming, distributed

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"m": pl.Series([{"a": 1, "b": 2}], dtype=pl.Map(pl.String, pl.Int64))}
        ... )
        >>> df.select(pl.col("m").map.entries())
        shape: (1, 1)
        ┌────────────────────┐
        │ m                  │
        │ ---                │
        │ list[struct[2]]    │
        ╞════════════════════╡
        │ [{"a",1}, {"b",2}] │
        └────────────────────┘
        """
        return wrap_expr(self._pyexpr.map_entries())
