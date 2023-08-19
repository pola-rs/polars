from __future__ import annotations

from typing import TYPE_CHECKING

from polars.utils._wrap import wrap_expr

if TYPE_CHECKING:
    from polars import Expr
    from polars.type_aliases import TransferEncoding


class ExprBinaryNameSpace:
    """Namespace for bin related expressions."""

    _accessor = "bin"

    def __init__(self, expr: Expr):
        self._pyexpr = expr._pyexpr

    def contains(self, literal: bytes) -> Expr:
        r"""
        Check if binaries in Series contain a binary substring.

        Parameters
        ----------
        literal
            The binary substring to look for

        Returns
        -------
        Expr
            Expression of data type :class:`Boolean`.

        See Also
        --------
        starts_with : Check if the binary substring exists at the start
        ends_with : Check if the binary substring exists at the end

        Examples
        --------
        >>> colors = pl.DataFrame(
        ...     {
        ...         "name": ["black", "yellow", "blue"],
        ...         "code": [b"\x00\x00\x00", b"\xff\xff\x00", b"\x00\x00\xff"],
        ...     }
        ... )
        >>> colors.select(
        ...     "name",
        ...     pl.col("code").bin.encode("hex").alias("code_encoded_hex"),
        ...     pl.col("code").bin.contains(b"\xff").alias("contains_ff"),
        ...     pl.col("code").bin.starts_with(b"\xff").alias("starts_with_ff"),
        ...     pl.col("code").bin.ends_with(b"\xff").alias("ends_with_ff"),
        ... )
        shape: (3, 5)
        ┌────────┬──────────────────┬─────────────┬────────────────┬──────────────┐
        │ name   ┆ code_encoded_hex ┆ contains_ff ┆ starts_with_ff ┆ ends_with_ff │
        │ ---    ┆ ---              ┆ ---         ┆ ---            ┆ ---          │
        │ str    ┆ str              ┆ bool        ┆ bool           ┆ bool         │
        ╞════════╪══════════════════╪═════════════╪════════════════╪══════════════╡
        │ black  ┆ 000000           ┆ false       ┆ false          ┆ false        │
        │ yellow ┆ ffff00           ┆ true        ┆ true           ┆ false        │
        │ blue   ┆ 0000ff           ┆ true        ┆ false          ┆ true         │
        └────────┴──────────────────┴─────────────┴────────────────┴──────────────┘
        """
        return wrap_expr(self._pyexpr.bin_contains(literal))

    def ends_with(self, suffix: bytes) -> Expr:
        r"""
        Check if string values end with a binary substring.

        Parameters
        ----------
        suffix
            Suffix substring.

        Returns
        -------
        Expr
            Expression of data type :class:`Boolean`.

        See Also
        --------
        starts_with : Check if the binary substring exists at the start
        contains : Check if the binary substring exists anywhere

        Examples
        --------
        >>> colors = pl.DataFrame(
        ...     {
        ...         "name": ["black", "yellow", "blue"],
        ...         "code": [b"\x00\x00\x00", b"\xff\xff\x00", b"\x00\x00\xff"],
        ...     }
        ... )
        >>> colors.select(
        ...     "name",
        ...     pl.col("code").bin.encode("hex").alias("code_encoded_hex"),
        ...     pl.col("code").bin.contains(b"\xff").alias("contains_ff"),
        ...     pl.col("code").bin.starts_with(b"\xff").alias("starts_with_ff"),
        ...     pl.col("code").bin.ends_with(b"\xff").alias("ends_with_ff"),
        ... )
        shape: (3, 5)
        ┌────────┬──────────────────┬─────────────┬────────────────┬──────────────┐
        │ name   ┆ code_encoded_hex ┆ contains_ff ┆ starts_with_ff ┆ ends_with_ff │
        │ ---    ┆ ---              ┆ ---         ┆ ---            ┆ ---          │
        │ str    ┆ str              ┆ bool        ┆ bool           ┆ bool         │
        ╞════════╪══════════════════╪═════════════╪════════════════╪══════════════╡
        │ black  ┆ 000000           ┆ false       ┆ false          ┆ false        │
        │ yellow ┆ ffff00           ┆ true        ┆ true           ┆ false        │
        │ blue   ┆ 0000ff           ┆ true        ┆ false          ┆ true         │
        └────────┴──────────────────┴─────────────┴────────────────┴──────────────┘
        """
        return wrap_expr(self._pyexpr.bin_ends_with(suffix))

    def starts_with(self, prefix: bytes) -> Expr:
        r"""
        Check if values start with a binary substring.

        Parameters
        ----------
        prefix
            Prefix substring.

        Returns
        -------
        Expr
            Expression of data type :class:`Boolean`.

        See Also
        --------
        ends_with : Check if the binary substring exists at the end
        contains : Check if the binary substring exists anywhere

        Examples
        --------
        >>> colors = pl.DataFrame(
        ...     {
        ...         "name": ["black", "yellow", "blue"],
        ...         "code": [b"\x00\x00\x00", b"\xff\xff\x00", b"\x00\x00\xff"],
        ...     }
        ... )
        >>> colors.select(
        ...     "name",
        ...     pl.col("code").bin.encode("hex").alias("code_encoded_hex"),
        ...     pl.col("code").bin.contains(b"\xff").alias("contains_ff"),
        ...     pl.col("code").bin.starts_with(b"\xff").alias("starts_with_ff"),
        ...     pl.col("code").bin.ends_with(b"\xff").alias("ends_with_ff"),
        ... )
        shape: (3, 5)
        ┌────────┬──────────────────┬─────────────┬────────────────┬──────────────┐
        │ name   ┆ code_encoded_hex ┆ contains_ff ┆ starts_with_ff ┆ ends_with_ff │
        │ ---    ┆ ---              ┆ ---         ┆ ---            ┆ ---          │
        │ str    ┆ str              ┆ bool        ┆ bool           ┆ bool         │
        ╞════════╪══════════════════╪═════════════╪════════════════╪══════════════╡
        │ black  ┆ 000000           ┆ false       ┆ false          ┆ false        │
        │ yellow ┆ ffff00           ┆ true        ┆ true           ┆ false        │
        │ blue   ┆ 0000ff           ┆ true        ┆ false          ┆ true         │
        └────────┴──────────────────┴─────────────┴────────────────┴──────────────┘
        """
        return wrap_expr(self._pyexpr.bin_starts_with(prefix))

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
            return wrap_expr(self._pyexpr.bin_hex_decode(strict))
        elif encoding == "base64":
            return wrap_expr(self._pyexpr.bin_base64_decode(strict))
        else:
            raise ValueError(
                f"encoding must be one of {{'hex', 'base64'}}, got {encoding!r}"
            )

    def encode(self, encoding: TransferEncoding) -> Expr:
        r"""
        Encode a value using the provided encoding.

        Parameters
        ----------
        encoding : {'hex', 'base64'}
            The encoding to use.

        Returns
        -------
        Expr
            Expression of data type :class:`Utf8` with values encoded using provided
            encoding.

        Examples
        --------
        >>> colors = pl.DataFrame(
        ...     {
        ...         "name": ["black", "yellow", "blue"],
        ...         "code": [b"\x00\x00\x00", b"\xff\xff\x00", b"\x00\x00\xff"],
        ...     }
        ... )
        >>> colors.with_columns(
        ...     pl.col("code").bin.encode("hex").alias("code_encoded_hex"),
        ... )
        shape: (3, 3)
        ┌────────┬───────────────┬──────────────────┐
        │ name   ┆ code          ┆ code_encoded_hex │
        │ ---    ┆ ---           ┆ ---              │
        │ str    ┆ binary        ┆ str              │
        ╞════════╪═══════════════╪══════════════════╡
        │ black  ┆ [binary data] ┆ 000000           │
        │ yellow ┆ [binary data] ┆ ffff00           │
        │ blue   ┆ [binary data] ┆ 0000ff           │
        └────────┴───────────────┴──────────────────┘

        """
        if encoding == "hex":
            return wrap_expr(self._pyexpr.bin_hex_encode())
        elif encoding == "base64":
            return wrap_expr(self._pyexpr.bin_base64_encode())
        else:
            raise ValueError(
                f"encoding must be one of {{'hex', 'base64'}}, got {encoding!r}"
            )
