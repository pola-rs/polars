from __future__ import annotations

from typing import TYPE_CHECKING

from polars._utils.unstable import unstable
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

    @unstable()
    def to(
        self,
        dtype: PolarsDataType | pl.DataTypeExpr,
    ) -> Expr:
        """
        Convert to an extension `dtype`.

        The input must be of the storage type of the extension dtype.

        .. warning::
            This functionality is currently considered **unstable**. It may be
            changed at any point without it being considered a breaking change.
        """
        py_dtype = parse_into_datatype_expr(dtype)._pydatatype_expr
        return wrap_expr(self._pyexpr.ext_to(py_dtype))

    @unstable()
    def storage(self) -> Expr:
        """
        Get the storage values of an extension data type.

        If the input does not have an extension data type, it is returned as-is.

        .. warning::
            This functionality is currently considered **unstable**. It may be
            changed at any point without it being considered a breaking change.
        """
        return wrap_expr(self._pyexpr.ext_storage())
