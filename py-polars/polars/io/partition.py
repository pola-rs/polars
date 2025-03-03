import contextlib
from typing import Literal, TypeAlias, Union

from polars._utils.unstable import issue_unstable_warning

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import PyPartitioning


class MaxSizePartitioning:
    """
    File Partitioning scheme that generates files up to a maximum size and then switches to the next file.

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


IOPartitioning: TypeAlias = Union[MaxSizePartitioning]
