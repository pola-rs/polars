import contextlib

from polars._utils.unstable import issue_unstable_warning

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import PyPartitioning


class MaxSizePartitioning:
    """
    Partitioning scheme to write files with a maximum size.

    This partitioning scheme generates files that have a given maximum size. If
    the size reaches the maximum size, it is closes and a new file is opened.

    The `path` can be given a `{part}` to specify the output files.

    .. warning::
        This functionality is currently considered **unstable**. It may be
        changed at any point without it being considered a breaking change.
    """

    _p: PyPartitioning

    def __init__(self, path: str, max_size: int) -> None:
        msg = "Partitioning strategies are considered unstable."
        issue_unstable_warning(msg)
        self._p = PyPartitioning.new_max_size(path, max_size)
