from __future__ import annotations

from typing import TYPE_CHECKING

from polars._utils.parse import parse_into_expression
from polars._utils.wrap import wrap_expr

if TYPE_CHECKING:
    from polars import Expr
    from polars._typing import IntoExpr


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

    def keys(self) -> Expr:
        """
        Get the keys of every map as a `List`, in entry order.

        .. engine-support:: in-memory, streaming, distributed

        Returns
        -------
        Expr
            Expression of data type :class:`List` of the map's key type.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "m": pl.Series(
        ...             [{"a": 1, "b": 2}, {}, None], dtype=pl.Map(pl.String, pl.Int64)
        ...         )
        ...     }
        ... )
        >>> df.select(pl.col("m").map.keys())
        shape: (3, 1)
        ┌────────────┐
        │ m          │
        │ ---        │
        │ list[str]  │
        ╞════════════╡
        │ ["a", "b"] │
        │ []         │
        │ null       │
        └────────────┘
        """
        return wrap_expr(self._pyexpr.map_keys())

    def values(self) -> Expr:
        """
        Get the values of every map as a `List`, in entry order.

        .. engine-support:: in-memory, streaming, distributed

        Returns
        -------
        Expr
            Expression of data type :class:`List` of the map's value type.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "m": pl.Series(
        ...             [{"a": 1, "b": 2}, {}, None], dtype=pl.Map(pl.String, pl.Int64)
        ...         )
        ...     }
        ... )
        >>> df.select(pl.col("m").map.values())
        shape: (3, 1)
        ┌───────────┐
        │ m         │
        │ ---       │
        │ list[i64] │
        ╞═══════════╡
        │ [1, 2]    │
        │ []        │
        │ null      │
        └───────────┘
        """
        return wrap_expr(self._pyexpr.map_values())

    def len(self) -> Expr:
        """
        Get the number of entries of every map.

        .. engine-support:: in-memory, streaming, distributed

        Returns
        -------
        Expr
            Expression of data type :class:`UInt32`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "m": pl.Series(
        ...             [{"a": 1, "b": 2}, {}, None], dtype=pl.Map(pl.String, pl.Int64)
        ...         )
        ...     }
        ... )
        >>> df.select(pl.col("m").map.len())
        shape: (3, 1)
        ┌──────┐
        │ m    │
        │ ---  │
        │ u32  │
        ╞══════╡
        │ 2    │
        │ 0    │
        │ null │
        └──────┘
        """
        return wrap_expr(self._pyexpr.map_len())

    def contains_key(self, key: IntoExpr) -> Expr:
        """
        Check whether every map holds `key`.

        .. engine-support:: in-memory, streaming, distributed

        Parameters
        ----------
        key
            Key to look up. A single value is broadcast over all maps. It is cast to
            the map's key type, and a key the cast cannot represent exactly, because it
            is out of range or would be rounded, is absent.

        Returns
        -------
        Expr
            Expression of data type :class:`Boolean`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "m": pl.Series(
        ...             [{"a": 1, "b": 2}, {}, None], dtype=pl.Map(pl.String, pl.Int64)
        ...         )
        ...     }
        ... )
        >>> df.select(pl.col("m").map.contains_key("a"))
        shape: (3, 1)
        ┌───────┐
        │ m     │
        │ ---   │
        │ bool  │
        ╞═══════╡
        │ true  │
        │ false │
        │ null  │
        └───────┘
        """
        key_pyexpr = parse_into_expression(key, str_as_lit=True)
        return wrap_expr(self._pyexpr.map_contains_key(key_pyexpr))

    def get(self, key: IntoExpr) -> Expr:
        """
        Look up `key` in every map.

        Maps that do not hold `key` yield null, as do null maps. Use
        :meth:`contains_key` to tell a missing key apart from a key whose value is null.

        .. engine-support:: in-memory, streaming, distributed

        Parameters
        ----------
        key
            Key to look up. A single value is broadcast over all maps. It is cast to
            the map's key type, and a key the cast cannot represent exactly, because it
            is out of range or would be rounded, is absent.

        Returns
        -------
        Expr
            Expression of the map's value type.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "m": pl.Series(
        ...             [{"a": 1, "b": 2}, {}, None], dtype=pl.Map(pl.String, pl.Int64)
        ...         ),
        ...         "k": ["b", "a", "a"],
        ...     }
        ... )
        >>> df.select(
        ...     pl.col("m").map.get("a").alias("a"),
        ...     pl.col("m").map.get(pl.col("k")).alias("by_col"),
        ... )
        shape: (3, 2)
        ┌──────┬────────┐
        │ a    ┆ by_col │
        │ ---  ┆ ---    │
        │ i64  ┆ i64    │
        ╞══════╪════════╡
        │ 1    ┆ 2      │
        │ null ┆ null   │
        │ null ┆ null   │
        └──────┴────────┘
        """
        key_pyexpr = parse_into_expression(key, str_as_lit=True)
        return wrap_expr(self._pyexpr.map_get(key_pyexpr))
