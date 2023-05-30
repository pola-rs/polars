"""Various public utility functions."""
from __future__ import annotations

import contextlib
import warnings
from typing import TYPE_CHECKING

from polars.utils.various import find_stacklevel

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
        stacklevel=find_stacklevel(),
    )
    return get_index_type()


def threadpool_size() -> int:
    """
    Get the number of threads in the Polars thread pool.

    Notes
    -----
    The threadpool size can be overridden by setting the ``POLARS_MAX_THREADS``
    environment variable before process start. (The thread pool is not behind a
    lock, so it cannot be modified once set). A reasonable use-case for this might
    be temporarily setting max threads to a low value before importing polars in a
    pyspark UDF or similar context. Otherwise, it is strongly recommended not to
    override this value as it will be set automatically by the engine.

    """
    return _threadpool_size()
