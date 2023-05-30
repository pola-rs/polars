from __future__ import annotations

from typing import TYPE_CHECKING

from polars.utils._wrap import wrap_expr

if TYPE_CHECKING:
    from polars import Expr
    from polars.type_aliases import CategoricalOrdering


class ExprCatNameSpace:
    """Namespace for categorical related expressions."""

    _accessor = "cat"

    def __init__(self, expr: Expr):
        self._pyexpr = expr._pyexpr

    def set_ordering(self, ordering: CategoricalOrdering) -> Expr:
        """
        Determine how this categorical series should be sorted.

        Parameters
        ----------
        ordering : {'physical', 'lexical'}
            Ordering type:

            - 'physical' -> Use the physical representation of the categories to
                determine the order (default).
            - 'lexical' -> Use the string values to determine the ordering.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"cats": ["z", "z", "k", "a", "b"], "vals": [3, 1, 2, 2, 3]}
        ... ).with_columns(
        ...     [
        ...         pl.col("cats").cast(pl.Categorical).cat.set_ordering("lexical"),
        ...     ]
        ... )
        >>> df.sort(["cats", "vals"])
        shape: (5, 2)
        ┌──────┬──────┐
        │ cats ┆ vals │
        │ ---  ┆ ---  │
        │ cat  ┆ i64  │
        ╞══════╪══════╡
        │ a    ┆ 2    │
        │ b    ┆ 3    │
        │ k    ┆ 2    │
        │ z    ┆ 1    │
        │ z    ┆ 3    │
        └──────┴──────┘

        """
        return wrap_expr(self._pyexpr.cat_set_ordering(ordering))
