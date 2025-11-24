from __future__ import annotations

from typing import TYPE_CHECKING

from polars import datatypes as dt
from polars._utils.unstable import unstable
from polars._utils.wrap import wrap_s
from polars.series.utils import expr_dispatch

if TYPE_CHECKING:
    from polars import Series
    from polars._plr import PySeries
    from polars._typing import (
        PolarsDataType,
    )


@expr_dispatch
class ExtensionNameSpace:
    """Series.ext namespace."""

    _accessor = "ext"

    def __init__(self, series: Series) -> None:
        self._s: PySeries = series._s

    @unstable()
    def to(self, dtype: PolarsDataType) -> Series:
        """
        Create a Series with an extension `dtype`.

        The input series must have the storage type of the extension dtype.

        .. warning::
            This functionality is currently considered **unstable**. It may be
            changed at any point without it being considered a breaking change.
        """
        assert isinstance(dtype, dt.BaseExtension)
        return wrap_s(self._s.ext_to(dtype))

    @unstable()
    def storage(self) -> Series:
        """
        Get the storage values of a Series with an extension data type.

        If the input series does not have an extension data type, it is returned as-is.

        .. warning::
            This functionality is currently considered **unstable**. It may be
            changed at any point without it being considered a breaking change.
        """
        return wrap_s(self._s.ext_storage())
