from __future__ import annotations

from typing import TYPE_CHECKING

from polars.series.utils import expr_dispatch
from polars.utils._wrap import wrap_s

if TYPE_CHECKING:
    from polars import Series
    from polars.polars import PySeries
    from polars.type_aliases import CategoricalOrdering


@expr_dispatch
class CatNameSpace:
    """Namespace for categorical related series."""

    _accessor = "cat"

    def __init__(self, series: Series):
        self._s: PySeries = series._s

    def set_ordering(self, ordering: CategoricalOrdering) -> Series:
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

    def get_categories(self) -> Series:
        """
        Get the categories stored in this data type.

        Examples
        --------
        >>> s = pl.Series(["foo", "bar", "foo", "foo", "ham"], dtype=pl.Categorical)
        >>> s.cat.get_categories()
        shape: (3,)
        Series: '' [str]
        [
            "foo"
            "bar"
            "ham"
        ]

        """

    def is_local(self) -> bool:
        """
        Return whether or not the column is a local categorical.

        Examples
        --------
        Categoricals constructed without a string cache are considered local.

        >>> s = pl.Series(["a", "b", "a"], dtype=pl.Categorical)
        >>> s.cat.is_local()
        True

        Categoricals constructed with a string cache are considered global.

        >>> with pl.StringCache():
        ...     s = pl.Series(["a", "b", "a"], dtype=pl.Categorical)
        ...
        >>> s.cat.is_local()
        False

        """
        return self._s.cat_is_local()

    def to_local(self) -> Series:
        """
        Convert a categorical column to its local representation.

        This may change the underlying physical representation of the column.

        See the documentation of :func:`StringCache` for more information on the
        difference between local and global categoricals.

        Examples
        --------
        Compare the global and local representations of a categorical.

        >>> with pl.StringCache():
        ...     _ = pl.Series("x", ["a", "b", "a"], dtype=pl.Categorical)
        ...     s = pl.Series("y", ["c", "b", "d"], dtype=pl.Categorical)
        ...
        >>> s.to_physical()
        shape: (3,)
        Series: 'y' [u32]
        [
                2
                1
                3
        ]
        >>> s.cat.to_local().to_physical()
        shape: (3,)
        Series: 'y' [u32]
        [
                0
                1
                2
        ]

        """
        return wrap_s(self._s.cat_to_local())

    def uses_lexical_ordering(self) -> bool:
        """
        Return whether or not the series uses lexical ordering.

        This can be set using :func:`set_ordering`.

        Warnings
        --------
        This API is experimental and may change without it being considered a breaking
        change.

        See Also
        --------
        set_ordering

        Examples
        --------
        >>> s = pl.Series(["b", "a", "b"]).cast(pl.Categorical)
        >>> s.cat.uses_lexical_ordering()
        False
        >>> s = s.cat.set_ordering("lexical")
        >>> s.cat.uses_lexical_ordering()
        True

        """
        return self._s.cat_uses_lexical_ordering()
