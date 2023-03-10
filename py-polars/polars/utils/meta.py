"""Various public utility functions."""
from __future__ import annotations

import contextlib
import warnings
from typing import TYPE_CHECKING

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import get_index_type as _get_index_type
    from polars.polars import threadpool_size as _threadpool_size

if TYPE_CHECKING:
    from polars.datatypes import DataTypeClass


def get_index_type() -> DataTypeClass:
    """
    Get the datatype used for Polars indexing.

    Returns
    -------
    UInt32 in regular Polars, UInt64 in bigidx Polars.

    """
    return _get_index_type()


def get_idx_type() -> DataTypeClass:
    """Get the datatype used for Polars indexing."""
    warnings.warn(
        "`get_idx_type` has been renamed; this"
        " redirect is temporary, please use `get_index_type` instead",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return get_index_type()


def threadpool_size() -> int:
    """Get the number of threads in the Polars thread pool."""
    return _threadpool_size()
