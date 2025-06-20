from __future__ import annotations

import polars._reexport as pl


class DataTypeExprIntNameSpace:
    """Namespace for integers datatype expressions."""

    _accessor = "int"

    def __init__(self, expr: pl.DataTypeExpr) -> None:
        self._pydatatype_expr = expr._pydatatype_expr

    def to_unsigned(self) -> pl.DataTypeExpr:
        """Get the unsigned integer version of the same bitsize."""
        return pl.DataTypeExpr._from_pydatatype_expr(
            self._pydatatype_expr.int_to_unsigned()
        )

    def to_signed(self) -> pl.DataTypeExpr:
        """Get the signed integer version of the same bitsize."""
        return pl.DataTypeExpr._from_pydatatype_expr(self._pydatatype_expr.int_to_signed())

    def is_unsigned(self) -> pl.Expr:
        """Get whether the given integer is unsigned."""
        return pl.Expr._from_pyexpr(self._pydatatype_expr.int_is_unsigned())

    def is_signed(self) -> pl.Expr:
        """Get whether the given integer is signed."""
        return pl.Expr._from_pyexpr(self._pydatatype_expr.int_is_signed())
