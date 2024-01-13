from __future__ import annotations

from typing import TYPE_CHECKING

import polars._reexport as pl
from polars.datatypes import N_INFER_DEFAULT

if TYPE_CHECKING:
    from io import IOBase
    from pathlib import Path

    from polars import DataFrame, LazyFrame
    from polars.type_aliases import SchemaDefinition


def read_ndjson(
    source: str | Path | IOBase | bytes,
    *,
    schema: SchemaDefinition | None = None,
    schema_overrides: SchemaDefinition | None = None,
    ignore_errors: bool = False,
) -> DataFrame:
    """
    Read into a `DataFrame` from a newline-delimited JSON file.

    Parameters
    ----------
    source
        A path to a file or a file-like object. By file-like object, we refer to objects
        that have a `read()` method, such as a file handler (e.g. from the builtin `open
        <https://docs.python.org/3/library/functions.html#open>`_ function) or `BytesIO
        <https://docs.python.org/3/library/io.html#io.BytesIO>`_.
    schema : Sequence of `str`, `(str, DataType)` pairs, or a `{str: DataType}` dict.
        The schema of the `DataFrame`. It may be declared in several ways:

        * As a dict of `{name: dtype}` pairs; if type is `None`, it will be
          auto-inferred.
        * As a list of column names; in this case types are automatically inferred.
        * As a list of `(name, type)` pairs; this is equivalent to the dictionary form.

        If you supply a list of column names that does not match the names in the
        underlying data, the names given here will overwrite them. The number
        of names given in the schema should match the underlying data dimensions.
    schema_overrides : dict, default None
        A dict of `{name: dtype}` pairs to override the dtypes of specific columns,
        instead of automatically inferring them or using the dtypes specified in
        the schema.
    ignore_errors
        Whether to return `null` if parsing fails because of schema mismatches,
        instead of raising an error.

    """
    return pl.DataFrame._read_ndjson(
        source,
        schema=schema,
        schema_overrides=schema_overrides,
        ignore_errors=ignore_errors,
    )


def scan_ndjson(
    source: str | Path | list[str] | list[Path],
    *,
    infer_schema_length: int | None = N_INFER_DEFAULT,
    batch_size: int | None = 1024,
    n_rows: int | None = None,
    low_memory: bool = False,
    rechunk: bool = True,
    row_count_name: str | None = None,
    row_count_offset: int = 0,
    schema: SchemaDefinition | None = None,
) -> LazyFrame:
    """
    Lazily read from a newline-delimited JSON file, or multiple files via glob patterns.

    This allows the query optimizer to push down predicates and projections to the scan
    level, thereby potentially reducing memory overhead.

    Parameters
    ----------
    source
        A path to a newline-delimited JSON file, or a glob pattern matching multiple
        files.
    infer_schema_length
        The number of rows to read when inferring the `schema`. If dtypes are inferred
        wrongly (e.g. as :class:`Int64` instead of :class:`Float64`), try to increase
        `infer_schema_length` or specify `schema`.
    batch_size
        The number of rows at a time to read into an intermediate buffer during JSON
        file reading. Modify this to change performance.
    n_rows
        The number of rows to read from the JSON file.
    low_memory
        Whether to reduce memory usage at the expense of speed.
    rechunk
        Whether to ensure each column of the result is stored contiguously in
        memory; see :func:`DataFrame.rechunk` for details.
    row_count_name
        If not `None`, add a row count column with this name as the first column.
    row_count_offset
        An integer offset to start the row count at; only used when `row_count_name`
        is not `None`.
    schema : Sequence of `str`, `(str, DataType)` pairs, or a `{str: DataType}` dict.
        The schema of the `DataFrame`. It may be declared in several ways:

        * As a dict of `{name: dtype}` pairs; if type is `None`, it will be
          auto-inferred.
        * As a list of column names; in this case types are automatically inferred.
        * As a list of `(name, type)` pairs; this is equivalent to the dictionary form.

        If you supply a list of column names that does not match the names in the
        underlying data, the names given here will overwrite them. The number
        of names given in the schema should match the underlying data dimensions.

    """
    return pl.LazyFrame._scan_ndjson(
        source,
        infer_schema_length=infer_schema_length,
        schema=schema,
        batch_size=batch_size,
        n_rows=n_rows,
        low_memory=low_memory,
        rechunk=rechunk,
        row_count_name=row_count_name,
        row_count_offset=row_count_offset,
    )
