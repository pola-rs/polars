from __future__ import annotations

from typing import TYPE_CHECKING

from polars.series.utils import expr_dispatch

if TYPE_CHECKING:
    from polars import Series
    from polars.polars import PySeries
    from polars.type_aliases import IntoExpr, TransferEncoding


@expr_dispatch
class BinaryNameSpace:
    """Series.bin namespace."""

    _accessor = "bin"

    def __init__(self, series: Series):
        self._s: PySeries = series._s

    def contains(self, literal: IntoExpr) -> Series:
        """
        Check if binaries in Series contain a binary substring.

        Parameters
        ----------
        literal
            The binary substring to look for

        Returns
        -------
        Series
            Series of data type :class:`Boolean`.

        """

    def ends_with(self, suffix: IntoExpr) -> Series:
        """
        Check if string values end with a binary substring.

        Parameters
        ----------
        suffix
            Suffix substring.

        """

    def starts_with(self, prefix: IntoExpr) -> Series:
        """
        Check if values start with a binary substring.

        Parameters
        ----------
        prefix
            Prefix substring.

        """

    def decode(self, encoding: TransferEncoding, *, strict: bool = True) -> Series:
        """
        Decode a value using the provided encoding.

        Parameters
        ----------
        encoding : {'hex', 'base64'}
            The encoding to use.
        strict
            Raise an error if the underlying value cannot be decoded,
            otherwise mask out with a null value.

        """

    def encode(self, encoding: TransferEncoding) -> Series:
        """
        Encode a value using the provided encoding.

        Parameters
        ----------
        encoding : {'hex', 'base64'}
            The encoding to use.

        Returns
        -------
        Series
            Series of data type :class:`Boolean`.

        """
