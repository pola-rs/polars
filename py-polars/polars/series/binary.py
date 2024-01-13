from __future__ import annotations

from typing import TYPE_CHECKING

from polars.series.utils import expr_dispatch

if TYPE_CHECKING:
    from polars import Series
    from polars.polars import PySeries
    from polars.type_aliases import IntoExpr, TransferEncoding


@expr_dispatch
class BinaryNameSpace:
    """A namespace for :class:`Binary` `Series`."""

    _accessor = "bin"

    def __init__(self, series: Series):
        self._s: PySeries = series._s

    def contains(self, literal: IntoExpr) -> Series:
        """
        Check if each binary element contains a binary substring.

        Parameters
        ----------
        literal
            The binary substring to search for.

        Returns
        -------
        Expr
            A :class:`Boolean` `Series`.

        """

    def ends_with(self, suffix: IntoExpr) -> Series:
        """
        Check if each binary element ends with a binary substring.

        Parameters
        ----------
        suffix
            The binary substring to search for.

        Returns
        -------
        Expr
            A :class:`Boolean` `Series`.

        """

    def starts_with(self, prefix: IntoExpr) -> Series:
        """
        Check if each binary element starts with a binary substring.

        Parameters
        ----------
        prefix
            The binary substring to search for.

        Returns
        -------
        Expr
            A :class:`Boolean` `Series`.

        """

    def decode(self, encoding: TransferEncoding, *, strict: bool = True) -> Series:
        """
        Decode binary values using the provided encoding.

        Parameters
        ----------
        encoding : {'hex', 'base64'}
            The encoding to use.
        strict
            Whether to raise an error if the underlying value cannot be decoded,
            instead of replacing it with `null`.

        """

    def encode(self, encoding: TransferEncoding) -> Series:
        """
        Encode binary values using the provided encoding.

        Parameters
        ----------
        encoding : {'hex', 'base64'}
            The encoding to use.

        Returns
        -------
        Expr
            A :class:`String` `Series` with values encoded using the specified encoding.

        """
