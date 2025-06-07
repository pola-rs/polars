from __future__ import annotations

import contextlib
from pathlib import Path
from typing import IO, TYPE_CHECKING

from polars._utils.wrap import wrap_ldf

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import PyLazyFrame

if TYPE_CHECKING:
    from polars import LazyFrame

def scan_fwf(
    source: (
        str
        | Path
        | IO[str]
        | IO[bytes]
        | bytes
        | list[str]
        | list[Path]
        | list[IO[str]]
        | list[IO[bytes]]
        | list[bytes]
    )
) -> LazyFrame:
    """
    Lazily read from a CSV file or multiple files via glob patterns.

    This allows the query optimizer to push down predicates and
    projections to the scan level, thereby potentially reducing
    memory overhead.

    Parameters
    ----------
    source
        Path(s) to a file or directory
        When needing to authenticate for scanning cloud locations, see the
        `storage_options` parameter.

    Returns
    -------
    LazyFrame
    """
    return _scan_fwf_impl(
        source,
    )


def _scan_fwf_impl(
    source: str
    | IO[str]
    | IO[bytes]
    | bytes
    | list[str]
    | list[Path]
    | list[IO[str]]
    | list[IO[bytes]]
    | list[bytes],
) -> LazyFrame:
    pylf = PyLazyFrame.new_from_fwf(
        source,
    )
    return wrap_ldf(pylf)
