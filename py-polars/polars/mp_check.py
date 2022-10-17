import multiprocessing as mp


def mp_check() -> None:
    """
    Check if python multiprocessing start method is compatible with Polars.

    Polars will only work with python multiprocessing if:
      - `mp.set_start_method("spawn")`
      -  or `mp.set_start_method("forkserver")`
    is ran before importing polars.

    If python multiprocessing method is undefined at import
    time of polars, it will be set to "spawn" by default.

    Python multiprocessing start method "fork" (default value
    on Linux) can not be used as forking does not copy threads
    (Polars spawns multiple threads for its rayon threadpool),
    resulting in locks that can't be released, resulting in a
    hanging python interpreter. This also happens with other
    Python modules that use native code.

    Examples
    --------
    Set forkserver instead of spawn as multiprocessing method
    before importing polars.

    >>> import multiprocessing as mp  # doctest: +SKIP
    >>> if not mp.get_start_method(allow_none=True):  # doctest: +SKIP
    ...     mp.set_start_method("forkserver")  # doctest: +SKIP
    ...
    >>> import polars as pl  # doctest: +SKIP

    """
    mp_method = mp.get_start_method(allow_none=True)
    if not mp_method:
        # Set default multiprocessing start method:
        #   - "spawn": Already the default value on Windows/MacOS.
        #   - "fork": Still the default on Linux.
        mp.set_start_method("spawn")
    elif mp_method == "fork":
        raise ImportError(
            "Polars only works with python multiprocessing method set to: "
            '`mp.set_start_method("spawn")` or `mp.set_start_method("forkserver")`'
            "before importing polars. "
            "See: https://docs.python.org/3/library/multiprocessing.html"
            "#contexts-and-start-methods"
        )


mp_check()
