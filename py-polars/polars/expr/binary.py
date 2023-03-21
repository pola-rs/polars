from __future__ import annotations

from typing import TYPE_CHECKING

from polars.utils._wrap import wrap_expr

if TYPE_CHECKING:
    from polars.expr import Expr
    from polars.type_aliases import TransferEncoding


class ExprBinaryNameSpace:
    """Namespace for bin related expressions."""

    _accessor = "bin"

    def __init__(self, expr: Expr):
        self._pyexpr = expr._pyexpr

    def contains(self, lit: bytes) -> Expr:
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
        return wrap_expr(self._pyexpr.binary_contains(lit))

    def ends_with(self, sub: bytes) -> Expr:
        """
        Check if string values end with a binary substring.

        Parameters
        ----------
        sub
            Suffix substring.

        """
        return wrap_expr(self._pyexpr.binary_ends_with(sub))

    def starts_with(self, sub: bytes) -> Expr:
        """
        Check if values start with a binary substring.

        Parameters
        ----------
        sub
            Prefix substring.

        """
        return wrap_expr(self._pyexpr.binary_starts_with(sub))

    def decode(self, encoding: TransferEncoding, *, strict: bool = True) -> Expr:
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
        if encoding == "hex":
            return wrap_expr(self._pyexpr.binary_hex_decode(strict))
        elif encoding == "base64":
            return wrap_expr(self._pyexpr.binary_base64_decode(strict))
        else:
            raise ValueError(
                f"encoding must be one of {{'hex', 'base64'}}, got {encoding}"
            )

    def encode(self, encoding: TransferEncoding) -> Expr:
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
            return wrap_expr(self._pyexpr.binary_hex_encode())
        elif encoding == "base64":
            return wrap_expr(self._pyexpr.binary_base64_encode())
        else:
            raise ValueError(
                f"encoding must be one of {{'hex', 'base64'}}, got {encoding}"
            )
