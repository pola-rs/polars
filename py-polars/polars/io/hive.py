from __future__ import annotations

from pathlib import Path
from typing import Sequence

from polars._utils.various import (
    is_str_sequence,
    normalize_filepath,
)


class PartitionedWriteOptions:
    """
    Configuration for writing a partitioned dataset.

    This is passed to `write_*` functions that support writing partitioned datasets.
    """

    def __init__(
        self,
        path: str | Path,
        partition_by: str | Sequence[str],
        *,
        chunk_size_bytes: int = 4_294_967_296,
    ):
        if not isinstance(path, (str, Path)):
            msg = f"`path` should be of type str or Path, got {type(path).__name__!r}"
            raise TypeError(msg)

        path = normalize_filepath(path, check_not_directory=False)

        if isinstance(partition_by, str):
            partition_by = [partition_by]

        if not is_str_sequence(partition_by):
            msg = f"`partition_by` should be of type str or Collection[str], got {type(path).__name__!r}"
            raise TypeError(msg)

        from polars.polars import PartitionedWriteOptions

        self._inner = PartitionedWriteOptions(path, partition_by, chunk_size_bytes)
