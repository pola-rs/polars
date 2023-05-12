from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars._reexport as pl
from polars.datatypes import N_INFER_DEFAULT
from polars.utils.various import normalise_filepath

if TYPE_CHECKING:
    from io import IOBase

    from polars import DataFrame, LazyFrame


def read_ndjson(source: str | Path | IOBase) -> DataFrame:
    """
    Read into a DataFrame from a newline delimited JSON file.

    Parameters
    ----------
    source
        Path to a file or a file-like object.

    """
    return pl.DataFrame._read_ndjson(source)


def scan_ndjson(
    source: str | Path,
    *,
    infer_schema_length: int | None = N_INFER_DEFAULT,
    batch_size: int | None = 1024,
    n_rows: int | None = None,
    low_memory: bool = False,
    rechunk: bool = True,
    row_count_name: str | None = None,
    row_count_offset: int = 0,
) -> LazyFrame:
    """
    Lazily read from a newline delimited JSON file or multiple files via glob patterns.

    This allows the query optimizer to push down predicates and projections to the scan
    level, thereby potentially reducing memory overhead.

    Parameters
    ----------
    source
        Path to a file.
    infer_schema_length
        Infer the schema from the first ``infer_schema_length`` rows.
    batch_size
        Number of rows to read in each batch.
    n_rows
        Stop reading from JSON file after reading ``n_rows``.
    low_memory
        Reduce memory pressure at the expense of performance.
    rechunk
        Reallocate to contiguous memory when all chunks/ files are parsed.
    row_count_name
        If not None, this will insert a row count column with give name into the
        DataFrame
    row_count_offset
        Offset to start the row_count column (only use if the name is set)

    """
    if isinstance(source, (str, Path)):
        source = normalise_filepath(source)

    return pl.LazyFrame._scan_ndjson(
        source,
        infer_schema_length=infer_schema_length,
        batch_size=batch_size,
        n_rows=n_rows,
        low_memory=low_memory,
        rechunk=rechunk,
        row_count_name=row_count_name,
        row_count_offset=row_count_offset,
    )
