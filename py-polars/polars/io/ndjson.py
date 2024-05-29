from __future__ import annotations

import contextlib
from io import BytesIO, StringIO
from pathlib import Path
from typing import TYPE_CHECKING

from polars._utils.deprecation import deprecate_renamed_parameter
from polars._utils.various import normalize_filepath
from polars._utils.wrap import wrap_df, wrap_ldf
from polars.datatypes import N_INFER_DEFAULT
from polars.io._utils import parse_row_index_args

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import PyDataFrame, PyLazyFrame

if TYPE_CHECKING:
    from io import IOBase

    from polars import DataFrame, LazyFrame
    from polars.type_aliases import SchemaDefinition


def read_ndjson(
    source: str | Path | IOBase | bytes,
    *,
    schema: SchemaDefinition | None = None,
    schema_overrides: SchemaDefinition | None = None,
    ignore_errors: bool = False,
) -> DataFrame:
    r"""
    Read into a DataFrame from a newline delimited JSON file.

    Parameters
    ----------
    source
        Path to a file or a file-like object (by "file-like object" we refer to objects
        that have a `read()` method, such as a file handler like the builtin `open`
        function, or a `BytesIO` instance).
    schema : Sequence of str, (str,DataType) pairs, or a {str:DataType,} dict
        The DataFrame schema may be declared in several ways:

        * As a dict of {name:type} pairs; if type is None, it will be auto-inferred.
        * As a list of column names; in this case types are automatically inferred.
        * As a list of (name,type) pairs; this is equivalent to the dictionary form.

        If you supply a list of column names that does not match the names in the
        underlying data, the names given here will overwrite them. The number
        of names given in the schema should match the underlying data dimensions.
    schema_overrides : dict, default None
        Support type specification or override of one or more columns; note that
        any dtypes inferred from the schema param will be overridden.
    ignore_errors
        Return `Null` if parsing fails because of schema mismatches.

    Examples
    --------
    >>> from io import StringIO
    >>> json_str = '{"foo":1,"bar":6}\n{"foo":2,"bar":7}\n{"foo":3,"bar":8}\n'
    >>> pl.read_ndjson(StringIO(json_str))
    shape: (3, 2)
    ┌─────┬─────┐
    │ foo ┆ bar │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 6   │
    │ 2   ┆ 7   │
    │ 3   ┆ 8   │
    └─────┴─────┘
    """
    if isinstance(source, StringIO):
        source = BytesIO(source.getvalue().encode())
    elif isinstance(source, (str, Path)):
        source = normalize_filepath(source)

    pydf = PyDataFrame.read_ndjson(
        source,
        ignore_errors=ignore_errors,
        schema=schema,
        schema_overrides=schema_overrides,
    )
    return wrap_df(pydf)


@deprecate_renamed_parameter("row_count_name", "row_index_name", version="0.20.4")
@deprecate_renamed_parameter("row_count_offset", "row_index_offset", version="0.20.4")
def scan_ndjson(
    source: str | Path | list[str] | list[Path],
    *,
    schema: SchemaDefinition | None = None,
    infer_schema_length: int | None = N_INFER_DEFAULT,
    batch_size: int | None = 1024,
    n_rows: int | None = None,
    low_memory: bool = False,
    rechunk: bool = False,
    row_index_name: str | None = None,
    row_index_offset: int = 0,
    ignore_errors: bool = False,
) -> LazyFrame:
    """
    Lazily read from a newline delimited JSON file or multiple files via glob patterns.

    This allows the query optimizer to push down predicates and projections to the scan
    level, thereby potentially reducing memory overhead.

    Parameters
    ----------
    source
        Path to a file.
    schema : Sequence of str, (str,DataType) pairs, or a {str:DataType,} dict
        The DataFrame schema may be declared in several ways:

        * As a dict of {name:type} pairs; if type is None, it will be auto-inferred.
        * As a list of column names; in this case types are automatically inferred.
        * As a list of (name,type) pairs; this is equivalent to the dictionary form.

        If you supply a list of column names that does not match the names in the
        underlying data, the names given here will overwrite them. The number
        of names given in the schema should match the underlying data dimensions.
    infer_schema_length
        The maximum number of rows to scan for schema inference.
        If set to `None`, the full data may be scanned *(this is slow)*.
    batch_size
        Number of rows to read in each batch.
    n_rows
        Stop reading from JSON file after reading `n_rows`.
    low_memory
        Reduce memory pressure at the expense of performance.
    rechunk
        Reallocate to contiguous memory when all chunks/ files are parsed.
    row_index_name
        If not None, this will insert a row index column with give name into the
        DataFrame
    row_index_offset
        Offset to start the row index column (only use if the name is set)
    ignore_errors
        Return `Null` if parsing fails because of schema mismatches.
    """
    if isinstance(source, (str, Path)):
        source = normalize_filepath(source)
        sources = []
    else:
        sources = [normalize_filepath(source) for source in source]
        source = None  # type: ignore[assignment]

    pylf = PyLazyFrame.new_from_ndjson(
        source,
        sources,
        infer_schema_length,
        schema,
        batch_size,
        n_rows,
        low_memory,
        rechunk,
        parse_row_index_args(row_index_name, row_index_offset),
        ignore_errors,
    )
    return wrap_ldf(pylf)
