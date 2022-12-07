from __future__ import annotations

from typing import TYPE_CHECKING

import polars.internals as pli
from polars.internals.series.utils import expr_dispatch

if TYPE_CHECKING:
    from polars.internals.type_aliases import CategoricalOrdering
    from polars.polars import PySeries


@expr_dispatch
class CatNameSpace:
    """Namespace for categorical related series."""

    _accessor = "cat"

    def __init__(self, series: pli.Series):
        self._s: PySeries = series._s

    @property
    def ordered(self) -> bool:
        """Return if sorting uses the categories or the lexical order of the string values."""  # noqa: E501
        # see https://github.com/pola-rs/polars/pull/5705#issuecomment-1339131964
        # for a beautiful image and the reason for this branch.
        if self._s is not None:
            return self._s.cat_is_ordered()
        else:
            return None  # type: ignore[return-value]

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
