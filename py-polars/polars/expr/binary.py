from __future__ import annotations

from typing import TYPE_CHECKING

from polars.utils._parse_expr_input import parse_as_expression
from polars.utils._wrap import wrap_expr

if TYPE_CHECKING:
    from polars import Expr
    from polars.type_aliases import IntoExpr, TransferEncoding


class ExprBinaryNameSpace:
    """Namespace for bin related expressions."""

    _accessor = "bin"

    def __init__(self, expr: Expr):
        self._pyexpr = expr._pyexpr

    def contains(self, literal: IntoExpr) -> Expr:
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
        ...         "lit": [b"\x00", b"\xff\x00", b"\xff\xff"],
        ...     }
        ... )
        >>> colors.select(
        ...     "name",
        ...     pl.col("code").bin.contains(b"\xff").alias("contains_with_lit"),
        ...     pl.col("code").bin.contains(pl.col("lit")).alias("contains_with_expr"),
        ... )
        shape: (3, 3)
        ┌────────┬───────────────────┬────────────────────┐
        │ name   ┆ contains_with_lit ┆ contains_with_expr │
        │ ---    ┆ ---               ┆ ---                │
        │ str    ┆ bool              ┆ bool               │
        ╞════════╪═══════════════════╪════════════════════╡
        │ black  ┆ false             ┆ true               │
        │ yellow ┆ true              ┆ true               │
        │ blue   ┆ true              ┆ false              │
        └────────┴───────────────────┴────────────────────┘
        """
        literal = parse_as_expression(literal, str_as_lit=True)
        return wrap_expr(self._pyexpr.bin_contains(literal))

    def ends_with(self, suffix: IntoExpr) -> Expr:
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
        ...         "suffix": [b"\x00", b"\xff\x00", b"\x00\x00"],
        ...     }
        ... )
        >>> colors.select(
        ...     "name",
        ...     pl.col("code").bin.ends_with(b"\xff").alias("ends_with_lit"),
        ...     pl.col("code").bin.ends_with(pl.col("suffix")).alias("ends_with_expr"),
        ... )
        shape: (3, 3)
        ┌────────┬───────────────┬────────────────┐
        │ name   ┆ ends_with_lit ┆ ends_with_expr │
        │ ---    ┆ ---           ┆ ---            │
        │ str    ┆ bool          ┆ bool           │
        ╞════════╪═══════════════╪════════════════╡
        │ black  ┆ false         ┆ true           │
        │ yellow ┆ false         ┆ true           │
        │ blue   ┆ true          ┆ false          │
        └────────┴───────────────┴────────────────┘
        """
        suffix = parse_as_expression(suffix, str_as_lit=True)
        return wrap_expr(self._pyexpr.bin_ends_with(suffix))

    def starts_with(self, prefix: IntoExpr) -> Expr:
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
        ...         "prefix": [b"\x00", b"\xff\x00", b"\x00\x00"],
        ...     }
        ... )
        >>> colors.select(
        ...     "name",
        ...     pl.col("code").bin.starts_with(b"\xff").alias("starts_with_lit"),
        ...     pl.col("code")
        ...     .bin.starts_with(pl.col("prefix"))
        ...     .alias("starts_with_expr"),
        ... )
        shape: (3, 3)
        ┌────────┬─────────────────┬──────────────────┐
        │ name   ┆ starts_with_lit ┆ starts_with_expr │
        │ ---    ┆ ---             ┆ ---              │
        │ str    ┆ bool            ┆ bool             │
        ╞════════╪═════════════════╪══════════════════╡
        │ black  ┆ false           ┆ true             │
        │ yellow ┆ true            ┆ false            │
        │ blue   ┆ false           ┆ true             │
        └────────┴─────────────────┴──────────────────┘
        """
        prefix = parse_as_expression(prefix, str_as_lit=True)
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
                f"`encoding` must be one of {{'hex', 'base64'}}, got {encoding!r}"
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
                f"`encoding` must be one of {{'hex', 'base64'}}, got {encoding!r}"
            )
