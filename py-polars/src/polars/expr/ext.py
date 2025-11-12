from __future__ import annotations

# import warnings
# from collections.abc import Mapping
from typing import TYPE_CHECKING

# import polars._reexport as pl
# from polars import functions as F
# from polars._utils.deprecation import deprecate_nonkeyword_arguments, deprecated
# from polars._utils.parse import parse_into_expression
# from polars._utils.unstable import unstable
# from polars._utils.various import (
#     find_stacklevel,
#     issue_warning,
#     no_default,
#     qualified_type_name,
# )
# from polars._utils.wrap import wrap_expr
# from polars.datatypes import Date, Datetime, Int64, Time, parse_into_datatype_expr
# from polars.exceptions import ChronoFormatWarning

if TYPE_CHECKING:
    # import sys

    from polars import Expr
    # from polars._typing import (
        # Ambiguous,
        # IntoExpr,
        # IntoExprColumn,
        # PolarsDataType,
        # PolarsIntegerType,
        # PolarsTemporalType,
        # TimeUnit,
        # TransferEncoding,
        # UnicodeForm,
    # )
    # from polars._utils.various import NoDefault

    # if sys.version_info >= (3, 13):
    #     from warnings import deprecated
    # else:
    #     from typing_extensions import deprecated  # noqa: TC004


class ExprExtensionNameSpace:
    """Namespace for extension type related expressions."""

    _accessor = "ext"

    def __init__(self, expr: Expr) -> None:
        self._pyexpr = expr._pyexpr
