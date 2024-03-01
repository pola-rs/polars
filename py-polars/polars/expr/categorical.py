from __future__ import annotations

from typing import TYPE_CHECKING

from polars._utils.deprecation import deprecate_function
from polars._utils.wrap import wrap_expr

if TYPE_CHECKING:
    from polars import Expr
    from polars.type_aliases import CategoricalOrdering


class ExprCatNameSpace:
    """Namespace for categorical related expressions."""

    _accessor = "cat"

    def __init__(self, expr: Expr):
        self._pyexpr = expr._pyexpr

    @deprecate_function(
        "Set the ordering directly on the datatype `pl.Categorical('lexical')`"
        " or `pl.Categorical('physical')` or `cast()` to the intended data type."
        " This method will be removed in the next breaking change",
        version="0.19.19",
    )
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
        """
        return wrap_expr(self._pyexpr.cat_set_ordering(ordering))

    def get_categories(self) -> Expr:
        """
        Get the categories stored in this data type.

        Examples
        --------
        >>> df = pl.Series(
        ...     "cats", ["foo", "bar", "foo", "foo", "ham"], dtype=pl.Categorical
        ... ).to_frame()
        >>> df.select(pl.col("cats").cat.get_categories())
        shape: (3, 1)
        ┌──────┐
        │ cats │
        │ ---  │
        │ str  │
        ╞══════╡
        │ foo  │
        │ bar  │
        │ ham  │
        └──────┘
        """
        return wrap_expr(self._pyexpr.cat_get_categories())
