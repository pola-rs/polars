from __future__ import annotations

import polars._reexport as pl


class DataTypeExprIntNameSpace:
    """Namespace for integers datatype expressions."""

    _accessor = "int"

    def __init__(self, expr: pl.DataTypeExpr) -> None:
        self._pydatatype_expr = expr._pydatatype_expr

    def to_unsigned(self) -> pl.DataTypeExpr:
        """
        Get the unsigned integer version of the same bitsize.

        Examples
        --------
        >>> int32 = pl.Int32.to_dtype_expr()
        >>> int32.int.to_unsigned().collect_dtype({})
        UInt32
        """
        return pl.DataTypeExpr._from_pydatatype_expr(
            self._pydatatype_expr.int_to_unsigned()
        )

    def to_signed(self) -> pl.DataTypeExpr:
        """
        Get the signed integer version of the same bitsize.

        Examples
        --------
        >>> uint32 = pl.UInt32.to_dtype_expr()
        >>> uint32.int.to_signed().collect_dtype({})
        Int32
        """
        return pl.DataTypeExpr._from_pydatatype_expr(
            self._pydatatype_expr.int_to_signed()
        )

    def is_unsigned(self) -> pl.Expr:
        """
        Get whether the given integer is unsigned.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     [
        ...         pl.Series("a", [1, 2, 3], pl.Int32()),
        ...         pl.Series("b", [1, 2, 3], pl.UInt64()),
        ...     ]
        ... )
        >>> df.select(
        ...     is_a_unsigned=pl.dtype_of("a").int.is_unsigned(),
        ...     is_b_unsigned=pl.dtype_of("b").int.is_unsigned(),
        ... )
        shape: (1, 2)
        ┌───────────────┬───────────────┐
        │ is_a_unsigned ┆ is_b_unsigned │
        │ ---           ┆ ---           │
        │ bool          ┆ bool          │
        ╞═══════════════╪═══════════════╡
        │ false         ┆ true          │
        └───────────────┴───────────────┘
        """
        return pl.Expr._from_pyexpr(self._pydatatype_expr.int_is_unsigned())

    def is_signed(self) -> pl.Expr:
        """
        Get whether the given integer is signed.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     [
        ...         pl.Series("a", [1, 2, 3], pl.Int32()),
        ...         pl.Series("b", [1, 2, 3], pl.UInt64()),
        ...     ]
        ... )
        >>> df.select(
        ...     is_a_signed=pl.dtype_of("a").int.is_signed(),
        ...     is_b_signed=pl.dtype_of("b").int.is_signed(),
        ... )
        shape: (1, 2)
        ┌─────────────┬─────────────┐
        │ is_a_signed ┆ is_b_signed │
        │ ---         ┆ ---         │
        │ bool        ┆ bool        │
        ╞═════════════╪═════════════╡
        │ true        ┆ false       │
        └─────────────┴─────────────┘
        """
        return pl.Expr._from_pyexpr(self._pydatatype_expr.int_is_signed())
