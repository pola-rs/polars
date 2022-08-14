from __future__ import annotations

from typing import TYPE_CHECKING

import polars.internals as pli

if TYPE_CHECKING:
    from polars.internals.type_aliases import CategoricalOrdering


class CatNameSpace:
    """Namespace for categorical related series."""

    def __init__(self, s: pli.Series):
        self._s = s

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
        return pli.select(pli.lit(self._s).cat.set_ordering(ordering)).to_series()
