from __future__ import annotations

from typing import TYPE_CHECKING

from polars._utils.deprecation import deprecate_function
from polars._utils.unstable import unstable
from polars._utils.wrap import wrap_s
from polars.series.utils import expr_dispatch

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

    @deprecate_function(
        "Set the ordering directly on the datatype `pl.Categorical('lexical')`"
        " or `pl.Categorical('physical')` or `cast()` to the intended data type."
        " This method will be removed in the next breaking change",
        version="0.19.19",
    )
    def set_ordering(self, ordering: CategoricalOrdering) -> Series:
        """
        Determine how this categorical series should be sorted.

        .. deprecated:: 0.19.19
            Set the ordering directly on the datatype `pl.Categorical('lexical')`
            or `pl.Categorical('physical')` or `cast()` to the intended data type.
            This method will be removed in the next breaking change

        Parameters
        ----------
        ordering : {'physical', 'lexical'}
            Ordering type:

            - 'physical' -> Use the physical representation of the categories to
              determine the order (default).
            - 'lexical' -> Use the string values to determine the ordering.
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

    @unstable()
    def uses_lexical_ordering(self) -> bool:
        """
        Return whether or not the series uses lexical ordering.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        Examples
        --------
        >>> s = pl.Series(["b", "a", "b"]).cast(pl.Categorical)
        >>> s.cat.uses_lexical_ordering()
        False
        >>> s = s.cast(pl.Categorical("lexical"))
        >>> s.cat.uses_lexical_ordering()
        True
        """
        return self._s.cat_uses_lexical_ordering()
