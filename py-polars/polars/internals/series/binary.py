from __future__ import annotations

from typing import TYPE_CHECKING

import polars.internals as pli
from polars.internals.series.utils import expr_dispatch

if TYPE_CHECKING:
    from polars.internals.type_aliases import TransferEncoding
    from polars.polars import PySeries


@expr_dispatch
class BinaryNameSpace:
    """Series.bin namespace."""

    _accessor = "bin"

    def __init__(self, series: pli.Series):
        self._s: PySeries = series._s

    def contains(self, lit: bytes) -> pli.Series:
        """
        Check if binaries in Series contain a binary substring.

        Parameters
        ----------
        lit
            The binary substring to look for

        Returns
        -------
        Boolean mask

        """

    def ends_with(self, sub: bytes) -> pli.Series:
        """
        Check if string values end with a binary substring.

        Parameters
        ----------
        sub
            Suffix substring.

        """

    def starts_with(self, sub: bytes) -> pli.Series:
        """
        Check if values start with a binary substring.

        Parameters
        ----------
        sub
            Prefix substring.

        """

    def decode(self, encoding: TransferEncoding, *, strict: bool = True) -> pli.Series:
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

    def encode(self, encoding: TransferEncoding) -> pli.Series:
        """
        Encode a value using the provided encoding.

        Parameters
        ----------
        encoding : {'hex', 'base64'}
            The encoding to use.

        Returns
        -------
        Binary array with values encoded using provided encoding

        """
