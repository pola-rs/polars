from __future__ import annotations

from typing import TYPE_CHECKING

import polars.internals as pli
from polars.datatypes import (
    DataType,
    is_polars_dtype,
)

if TYPE_CHECKING:
    from polars.internals.type_aliases import TransferEncoding

class ExprBinaryNameSpace:
    """Namespace for binary related expressions."""

    _accessor = "binary"

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
        return pli.wrap_expr(self._pyexpr.binary_contains(pattern, literal))

    def ends_with(self, sub: bytes) -> pli.Series:
        """
        Check if string values end with a binary substring.

        Parameters
        ----------
        sub
            Suffix substring.

        """
        return pli.wrap_expr(self._pyexpr.binary_ends_with(sub))

    def starts_with(self, sub: bytes) -> pli.Series:
        """
        Check if values start with a binary substring.

        Parameters
        ----------
        sub
            Prefix substring.

        """
        return pli.wrap_expr(self._pyexpr.binary_starts_with(sub))

    def decode(self, encoding: TransferEncoding, strict: bool = False) -> pli.Series:
        """
        Decode a value using the provided encoding.

        Parameters
        ----------
        encoding : {'hex', 'base64'}
            The encoding to use.
        strict
            How to handle invalid inputs:

            - ``True``: An error will be thrown if unable to decode a value.
            - ``False``: Unhandled values will be replaced with `None`.
        """
        if encoding == "hex":
            return pli.wrap_expr(self._pyexpr.binary_hex_decode(strict))
        elif encoding == "base64":
            return pli.wrap_expr(self._pyexpr.binary_base64_decode(strict))
        else:
            raise ValueError(
                f"encoding must be one of {{'hex', 'base64'}}, got {encoding}"
            )

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
        if encoding == "hex":
            return pli.wrap_expr(self._pyexpr.binary_hex_encode())
        elif encoding == "base64":
            return pli.wrap_expr(self._pyexpr.binary_base64_encode())
        else:
            raise ValueError(
                f"encoding must be one of {{'hex', 'base64'}}, got {encoding}"
            )