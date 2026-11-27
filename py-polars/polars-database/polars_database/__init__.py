"""
Native database scan plugin for Polars (`polars.scan_database`).

This package is a dependency of Polars and is not meant to be used directly. Polars
loads it through the versioned FFI contracts it exports (`_polars_io_plugin_ids`,
`_capsule(id)`).
"""

import atexit

from polars_database._lib import (
    __build_commit__,
    __version__,
    _at_exit,
    _capsule,
    _polars_io_plugin_ids,
    adbc_db_kwargs,
    default_max_connections,
    dialect_from_scheme,
    driver_statuses,
    find_adbc_driver,
    scan_resolver,
)

_at_exit_gave_up = False


def _run_at_exit() -> None:
    """
    `_at_exit`, registered twice (see `_bridge._register_at_exit_before_duckdb`): the
    second call only waits for queries started in between, unless the first gave up.
    """
    global _at_exit_gave_up
    if not _at_exit_gave_up:
        _at_exit_gave_up = not _at_exit()


# Queries can outlive the call that started them (a `collect_batches()` iterator dropped
# just before exit): their sessions are closed before Python finalizes, or left alone.
atexit.register(_run_at_exit)

__all__ = [
    "__build_commit__",
    "__version__",
    "_capsule",
    "_polars_io_plugin_ids",
    "adbc_db_kwargs",
    "default_max_connections",
    "dialect_from_scheme",
    "driver_statuses",
    "find_adbc_driver",
    "scan_resolver",
]
