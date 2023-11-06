from __future__ import annotations

from typing import TYPE_CHECKING

from polars.utils._wrap import wrap_expr

if TYPE_CHECKING:
    from polars import Expr


class ExprArrayNameSpace:
    """Namespace for array related expressions."""

    _accessor = "arr"

    def __init__(self, expr: Expr):
        self._pyexpr = expr._pyexpr

    def min(self) -> Expr:
        """
        Compute the min values of the sub-arrays.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={"a": [[1, 2], [4, 3]]},
        ...     schema={"a": pl.Array(inner=pl.Int64, width=2)},
        ... )
        >>> df.select(pl.col("a").arr.min())
        shape: (2, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 1   │
        │ 3   │
        └─────┘

        """
        return wrap_expr(self._pyexpr.arr_min())

    def max(self) -> Expr:
        """
        Compute the max values of the sub-arrays.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={"a": [[1, 2], [4, 3]]},
        ...     schema={"a": pl.Array(inner=pl.Int64, width=2)},
        ... )
        >>> df.select(pl.col("a").arr.max())
        shape: (2, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 2   │
        │ 4   │
        └─────┘

        """
        return wrap_expr(self._pyexpr.arr_max())

    def sum(self) -> Expr:
        """
        Compute the sum values of the sub-arrays.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={"a": [[1, 2], [4, 3]]},
        ...     schema={"a": pl.Array(inner=pl.Int64, width=2)},
        ... )
        >>> df.select(pl.col("a").arr.sum())
        shape: (2, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 3   │
        │ 7   │
        └─────┘

        """
        return wrap_expr(self._pyexpr.arr_sum())

    def unique(self, *, maintain_order: bool = False) -> Expr:
        """
        Get the unique/distinct values in the array.

        Parameters
        ----------
        maintain_order
            Maintain order of data. This requires more work.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [[1, 1, 2]],
        ...     },
        ...     schema={"a": pl.Array(inner=pl.Int64, width=3)},
        ... )
        >>> df.select(pl.col("a").arr.unique())
        shape: (1, 1)
        ┌───────────┐
        │ a         │
        │ ---       │
        │ list[i64] │
        ╞═══════════╡
        │ [1, 2]    │
        └───────────┘

        """
        return wrap_expr(self._pyexpr.arr_unique(maintain_order))

    def to_list(self) -> Expr:
        """
        Convert an Array column into a List column with the same inner data type.

        Returns
        -------
        Expr
            Expression of data type :class:`List`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={"a": [[1, 2], [3, 4]]},
        ...     schema={"a": pl.Array(inner=pl.Int8, width=2)},
        ... )
        >>> df.select(pl.col("a").arr.to_list())
        shape: (2, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ list[i8] │
        ╞══════════╡
        │ [1, 2]   │
        │ [3, 4]   │
        └──────────┘

        """
        return wrap_expr(self._pyexpr.arr_to_list())
