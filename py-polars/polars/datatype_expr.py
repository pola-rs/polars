from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Self

import polars._reexport as pl

if TYPE_CHECKING:
    import contextlib
    with contextlib.suppress(ImportError):  # Module not available when building docs
        from polars.polars import PyDataTypeExpr
    from polars._typing import SchemaDict


class DataTypeExpr:
    """
    A lazily instantiated [`DataType`] that can be used in [`Expr`]s.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.

    A expression made to represent a [`DataType`] that can be used to reference the
    datatype of an expression of column in a lazy context.
    """

    _pydatatype_expr: PyDataTypeExpr

    @staticmethod
    def _from_pydatatype_expr(cls, pydatatype_expr: PyDataTypeExpr) -> Self:
        cls._pydatatype_expr = pydatatype_expr
        return cls

    @classmethod
    def collect_dtype(
        cls, context: SchemaDict | pl.Schema | pl.DataFrame | pl.LazyFrame
    ) -> pl.DataType:
        """Materialize the [`DataTypeExpr`] in a specific context."""
        schema: pl.Schema
        if isinstance(context, pl.Schema):
            schema = context
        elif isinstance(context, Mapping):
            schema = pl.Schema(context)
        elif isinstance(context, pl.DataFrame):
            schema = context.schema
        elif isinstance(context, pl.LazyFrame):
            schema = context.collect_schema()
        else:
            msg = f"DataTypeExpr.collect_dtype did not expect {context!r}"
            raise TypeError(msg)

        cls._pydatatype_expr.collect_dtype(schema)
