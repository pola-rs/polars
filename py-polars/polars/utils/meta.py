"""Various public utility functions."""
from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

with contextlib.suppress(ImportError):  # Module not available when building docs
    import polars.polars as plr

if TYPE_CHECKING:
    from polars.datatypes import DataType


def get_index_type() -> DataType:
    """
    Get the datatype used for Polars indexing.

    Returns
    -------
    DataType
        :class:`UInt32` in regular Polars, :class:`UInt64` in bigidx Polars.

    Examples
    --------
    >>> pl.get_index_type()
    UInt32
    """
    return plr.get_index_type()


def threadpool_size() -> int:
    """
    Get the number of threads in the Polars thread pool.

    Notes
    -----
    The threadpool size can be overridden by setting the `POLARS_MAX_THREADS`
    environment variable before process start. (The thread pool is not behind a
    lock, so it cannot be modified once set). A reasonable use-case for this might
    be temporarily setting max threads to a low value before importing polars in a
    pyspark UDF or similar context. Otherwise, it is strongly recommended not to
    override this value as it will be set automatically by the engine.

    Examples
    --------
    >>> pl.threadpool_size()  # doctest: +SKIP
    24
    """
    return plr.threadpool_size()
