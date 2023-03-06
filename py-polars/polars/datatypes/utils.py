from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import get_idx_type as _get_idx_type

if TYPE_CHECKING:
    from polars.datatypes import DataTypeClass


def get_idx_type() -> DataTypeClass:
    """
    Get the datatype used for Polars indexing.

    Returns
    -------
    UInt32 in regular Polars, UInt64 in bigidx Polars.

    """
    return _get_idx_type()
