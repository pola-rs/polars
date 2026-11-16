from __future__ import annotations

from typing import TYPE_CHECKING

from polars.expr.union import _NO_DEFAULT
from polars.series.utils import expr_dispatch

if TYPE_CHECKING:
    from collections.abc import Mapping

    from polars import Series
    from polars._plr import PySeries
    from polars._typing import IntoExpr


@expr_dispatch
class UnionNameSpace:
    """Series.union namespace."""

    _accessor = "union"

    def __init__(self, series: Series) -> None:
        self._s: PySeries = series._s

    def tag(self) -> Series:
        """
        Get the variant of every row.

        Returns an :class:`Enum` of the variant names, in variant order. Null rows
        give null.

        .. engine-support:: in-memory, streaming

        Examples
        --------
        >>> dtype = pl.Union({"i": pl.Int64, "s": pl.String})
        >>> s = pl.select(pl.union_value(dtype, i=pl.Series([1, None]))).to_series()
        >>> s.union.tag()
        shape: (2,)
        Series: '' [enum]
        [
                "i"
                null
        ]
        """

    def field(self, name: str) -> Series:
        """
        Get the values of one variant.

        Rows of other variants and null rows give null.

        .. engine-support:: in-memory, streaming

        Parameters
        ----------
        name
            The name of the variant.

        Examples
        --------
        >>> dtype = pl.Union({"i": pl.Int64, "s": pl.String})
        >>> s = pl.select(pl.union_value(dtype, i=pl.Series([1, None]))).to_series()
        >>> s.union.field("i")
        shape: (2,)
        Series: '' [i64]
        [
                1
                null
        ]
        """

    def match(
        self,
        arms: Mapping[str, IntoExpr] | None = None,
        /,
        *,
        default: IntoExpr = _NO_DEFAULT,
        **named_arms: IntoExpr,
    ) -> Series:
        """
        Evaluate an expression that depends on the variant of each row.

        Inside the arm of a variant, :func:`element` is the value of that variant.

        .. engine-support:: in-memory, streaming

        Parameters
        ----------
        arms
            A mapping from variant names to arms, for names that are not valid
            keyword arguments.
        default
            The value for variants without an arm. Required unless every variant
            has an arm; `None` gives null.
        **named_arms
            The arm of each variant, by variant name.

        Notes
        -----
        The arms and the default must have the same data type, except that null
        and literals such as `0` take the data type of the other arms. They must
        be elementwise. A null row gives null.

        Examples
        --------
        >>> dtype = pl.Union({"i": pl.Int64, "s": pl.String})
        >>> s = pl.select(pl.union_value(dtype, i=pl.Series([1, None]))).to_series()
        >>> s.union.match(i=pl.element() * 10, default=0)
        shape: (2,)
        Series: '' [i64]
        [
                10
                null
        ]
        """
