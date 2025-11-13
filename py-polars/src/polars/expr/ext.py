from __future__ import annotations

from typing import TYPE_CHECKING

from polars._utils.wrap import wrap_expr
from polars.datatypes import parse_into_datatype_expr

if TYPE_CHECKING:
    import polars._reexport as pl
    from polars import Expr
    from polars._typing import (
        PolarsDataType,
    )


class ExprExtensionNameSpace:
    """Namespace for extension type related expressions."""

    _accessor = "ext"

    def __init__(self, expr: Expr) -> None:
        self._pyexpr = expr._pyexpr

    def to(
        self,
        dtype: PolarsDataType | pl.DataTypeExpr,
    ) -> Expr:
        """
        Convert to an extension `dtype`.

        The input must be of the storage type of the extension dtype.
        """
        dtype = parse_into_datatype_expr(dtype)._pydatatype_expr
        return wrap_expr(self._pyexpr.ext_to(dtype))

    def storage(self) -> Expr:
        """
        Get the storage values of an extension data type.

        If the input does not have an extension data type, it is returned as-is.
        """
        return wrap_expr(self._pyexpr.ext_storage())
