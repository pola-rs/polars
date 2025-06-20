from __future__ import annotations

import polars._reexport as pl


class DataTypeExprEnumNameSpace:
    """Namespace for enum datatype expressions."""

    _accessor = "enum"

    def __init__(self, expr: pl.DataTypeExpr) -> None:
        self._pydatatype_expr = expr._pydatatype_expr

    def num_categories(self) -> pl.Expr:
        """Get the number of enum categories."""
        return pl.Expr._from_pyexpr(self._pydatatype_expr.enum_num_categories())

    def categories(self) -> pl.Expr:
        """Get the enum categories as a list."""
        return pl.Expr._from_pyexpr(self._pydatatype_expr.enum_categories())

    def get_category(self, index: int, *, raise_on_oob: bool = True) -> pl.Expr:
        """Get the enum category at a specific index."""
        return pl.Expr._from_pyexpr(
            self._pydatatype_expr.enum_get_category(index, raise_on_oob)
        )

    def index_of_category(
        self, category: str, *, raise_on_missing: bool = True
    ) -> pl.Expr:
        """Get the index of a specific enum category."""
        return pl.Expr._from_pyexpr(
            self._pydatatype_expr.enum_index_of_category(category, raise_on_missing)
        )
