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
        r"""
        Check if binaries in Series contain a binary substring.

        Parameters
        ----------
        literal
            The binary substring to look for

        Returns
        -------
        Series
            Series of data type :class:`Boolean`.

        Examples
        --------
        >>> s = pl.Series("colors", [b"\x00\x00\x00", b"\xff\xff\x00", b"\x00\x00\xff"])
        >>> s.bin.contains(b"\xff")
        shape: (3,)
        Series: 'colors' [bool]
        [
            false
            true
            true
        ]
        """

    def ends_with(self, suffix: IntoExpr) -> Series:
        r"""
        Check if string values end with a binary substring.

        Parameters
        ----------
        suffix
            Suffix substring.

        Examples
        --------
        >>> s = pl.Series("colors", [b"\x00\x00\x00", b"\xff\xff\x00", b"\x00\x00\xff"])
        >>> s.bin.ends_with(b"\x00")
        shape: (3,)
        Series: 'colors' [bool]
        [
            true
            true
            false
        ]
        """

    def starts_with(self, prefix: IntoExpr) -> Series:
        r"""
        Check if values start with a binary substring.

        Parameters
        ----------
        prefix
            Prefix substring.

        Examples
        --------
        >>> s = pl.Series("colors", [b"\x00\x00\x00", b"\xff\xff\x00", b"\x00\x00\xff"])
        >>> s.bin.starts_with(b"\x00")
        shape: (3,)
        Series: 'colors' [bool]
        [
            true
            false
            true
        ]
        """

    def decode(self, encoding: TransferEncoding, *, strict: bool = True) -> Series:
        r"""
        Decode a value using the provided encoding.

        Parameters
        ----------
        encoding : {'hex', 'base64'}
            The encoding to use.
        strict
            Raise an error if the underlying value cannot be decoded,
            otherwise mask out with a null value.

        Examples
        --------
        >>> s = pl.Series("colors", [b"000000", b"ffff00", b"0000ff"])
        >>> s.bin.decode("hex")
        shape: (3,)
        Series: 'colors' [binary]
        [
            b"\x00\x00\x00"
            b"\xff\xff\x00"
            b"\x00\x00\xff"
        ]
        >>> s = pl.Series("colors", [b"AAAA", b"//8A", b"AAD/"])
        >>> s.bin.decode("base64")
        shape: (3,)
        Series: 'colors' [binary]
        [
            b"\x00\x00\x00"
            b"\xff\xff\x00"
            b"\x00\x00\xff"
        ]
        """

    def encode(self, encoding: TransferEncoding) -> Series:
        r"""
        Encode a value using the provided encoding.

        Parameters
        ----------
        encoding : {'hex', 'base64'}
            The encoding to use.

        Returns
        -------
        Series
            Series of data type :class:`Boolean`.

        Examples
        --------
        >>> s = pl.Series("colors", [b"\x00\x00\x00", b"\xff\xff\x00", b"\x00\x00\xff"])
        >>> s.bin.encode("hex")
        shape: (3,)
        Series: 'colors' [str]
        [
            "000000"
            "ffff00"
            "0000ff"
        ]
        >>> s.bin.encode("base64")
        shape: (3,)
        Series: 'colors' [str]
        [
            "AAAA"
            "//8A"
            "AAD/"
        ]
        """
