from __future__ import annotations

from typing import TYPE_CHECKING

# import polars._reexport as pl
# import polars.functions as F
# from polars._utils.deprecation import deprecate_nonkeyword_arguments, deprecated
# from polars._utils.unstable import unstable
# from polars._utils.various import no_default
from polars._utils.wrap import wrap_s
# from polars.datatypes import Int64
# from polars.datatypes.classes import Datetime
# from polars.datatypes.constants import N_INFER_DEFAULT
from polars import datatypes as dt
from polars.series.utils import expr_dispatch

if TYPE_CHECKING:
#     import sys
#     from collections.abc import Mapping

    from polars import Expr, Series
    from polars._plr import PySeries
    from polars._typing import (
#         Ambiguous,
#         IntoExpr,
#         IntoExprColumn,
        PolarsDataType,
#         PolarsIntegerType,
#         PolarsTemporalType,
#         TimeUnit,
#         TransferEncoding,
#         UnicodeForm,
    )
#     from polars._utils.various import NoDefault

#     if sys.version_info >= (3, 13):
#         from warnings import deprecated
#     else:
#         from typing_extensions import deprecated  # noqa: TC004


@expr_dispatch
class ExtensionNameSpace:
    """Series.ext namespace."""

    _accessor = "ext"

    def __init__(self, series: Series) -> None:
        self._s: PySeries = series._s

    def from_storage(self, dtype: PolarsDataType) -> Series:
        """Create an Extension Series from its storage Series."""
        assert isinstance(dtype, dt.BaseExtension)
        return wrap_s(self._s.ext_from_storage(dtype))
