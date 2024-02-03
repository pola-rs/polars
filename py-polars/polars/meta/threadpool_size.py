from __future__ import annotations

import contextlib

with contextlib.suppress(ImportError):  # Module not available when building docs
    import polars.polars as plr


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
    16
    """
    return plr.threadpool_size()
