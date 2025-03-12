from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from polars._utils.unstable import issue_unstable_warning

if TYPE_CHECKING:
    from pathlib import Path

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import PyPartitioning


class PartitionMaxSize:
    """
    Partitioning scheme to write files with a maximum size.

    This partitioning scheme generates files that have a given maximum size. If
    the size reaches the maximum size, it is closed and a new file is opened.

    .. warning::
        This functionality is currently considered **unstable**. It may be
        changed at any point without it being considered a breaking change.

    Parameters
    ----------
    path
        The path to the output files. The format string `{part}` is replaced to the
        zero-based index of the file.
    max_size : int
        The maximum size in rows of each of the generated files.
    """

    _p: PyPartitioning

    def __init__(self, path: Path | str, *, max_size: int) -> None:
        issue_unstable_warning("Partitioning strategies are considered unstable.")
        self._p = PyPartitioning.new_max_size(path, max_size)

    @property
    def _path(self) -> str:
        return self._p.path
