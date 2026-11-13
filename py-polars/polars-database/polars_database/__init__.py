"""
Native database scan plugin for Polars (`polars.scan_database`).

This package is a dependency of Polars and is not meant to be used directly. Polars
loads it through the versioned FFI contracts it exports (`_polars_io_plugin_ids`,
`_capsule(id)`).
"""

import atexit

from polars_database._lib import (
    __version__,
    _at_exit,
    _capsule,
    _polars_io_plugin_ids,
    dialect_from_scheme,
    driver_statuses,
    find_adbc_driver,
    scan_resolver,
)

# Queries can outlive the call that started them (a `collect_batches()` iterator dropped
# just before exit): their sessions are closed before Python finalizes, or left alone.
atexit.register(_at_exit)

__all__ = [
    "__version__",
    "_capsule",
    "_polars_io_plugin_ids",
    "dialect_from_scheme",
    "driver_statuses",
    "find_adbc_driver",
    "scan_resolver",
]
