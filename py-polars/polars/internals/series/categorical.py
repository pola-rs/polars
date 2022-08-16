from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import polars.internals as pli
from polars.internals.series.utils import call_expr

if TYPE_CHECKING:
    from polars.internals.type_aliases import CategoricalOrdering
    from polars.polars import PySeries

    if sys.version_info >= (3, 8):
        from typing import Final
    else:
        from typing_extensions import Final


class CatNameSpace:
    """Namespace for categorical related series."""

    _accessor: Final = "cat"

    def __init__(self, series: pli.Series):
        self._s: PySeries = series._s

    @call_expr
    def set_ordering(self, ordering: CategoricalOrdering) -> pli.Series:
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
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ b    ┆ 3    │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ k    ┆ 2    │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ z    ┆ 1    │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ z    ┆ 3    │
        └──────┴──────┘

        """
        ...
