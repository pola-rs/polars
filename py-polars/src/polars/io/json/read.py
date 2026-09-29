from __future__ import annotations

import contextlib
from glob import glob, has_magic
from io import BytesIO, StringIO
from pathlib import Path
from typing import TYPE_CHECKING

from polars import functions as F
from polars._utils.various import normalize_filepath
from polars._utils.wrap import wrap_df
from polars.datatypes import N_INFER_DEFAULT, String
from polars.exceptions import DuplicateError

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars._plr import PyDataFrame

if TYPE_CHECKING:
    from io import IOBase

    from polars import DataFrame
    from polars._typing import SchemaDefinition


def read_json(
    source: str | Path | IOBase | bytes,
    *,
    schema: SchemaDefinition | None = None,
    schema_overrides: SchemaDefinition | None = None,
    infer_schema_length: int | None = N_INFER_DEFAULT,
    include_file_paths: str | None = None,
) -> DataFrame:
    """
    Read into a DataFrame from a JSON file.

    Parameters
    ----------
    source
        A file path, glob pattern, file-like object, or bytes containing JSON data.
        File-like objects include file handles returned by `open` and `BytesIO`
        instances. Their stream position may not be updated after reading.
        Glob matches are read in sorted path order. An existing file path is read
        literally, even if its name contains glob characters.
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
    infer_schema_length
        The maximum number of rows per file to scan for schema inference.
        If set to `None`, the full data may be scanned *(this is slow)*.
    include_file_paths
        Include the path of the source file as a column with this name. For in-memory
        data and file-like objects, the column contains "in-mem". The name must not
        conflict with a column in the data.

    Notes
    -----
    When reading multiple files, schemas are inferred separately for each file.
    Columns are matched by name, missing columns are filled with nulls, and differing
    data types are cast to their common supertype.

    See Also
    --------
    read_ndjson

    Notes
    -----
    JSON objects can be read as :class:`Map` with String, Categorical or Enum keys.
    Specify Map through `schema` or `schema_overrides`; it is never inferred.
    Duplicate keys retain their first position and last value.

    Examples
    --------
    >>> from io import StringIO
    >>> json_str = '[{"foo":1,"bar":6},{"foo":2,"bar":7},{"foo":3,"bar":8}]'
    >>> pl.read_json(StringIO(json_str))
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

    With the schema defined.

    >>> pl.read_json(StringIO(json_str), schema={"foo": pl.Int64, "bar": pl.Float64})
    shape: (3, 2)
    ┌─────┬─────┐
    │ foo ┆ bar │
    │ --- ┆ --- │
    │ i64 ┆ f64 │
    ╞═════╪═════╡
    │ 1   ┆ 6.0 │
    │ 2   ┆ 7.0 │
    │ 3   ┆ 8.0 │
    └─────┴─────┘

    Read multiple files and include their paths in the result:

    >>> pl.read_json("data/*.json", include_file_paths="source")  # doctest: +SKIP
    """
    if isinstance(source, StringIO):
        source = BytesIO(source.getvalue().encode())
    elif isinstance(source, (str, Path)):
        source = normalize_filepath(source)
        if has_magic(source) and not Path(source).exists():
            paths = sorted(glob(source, recursive=True))  # noqa: PTH207
            if not paths:
                msg = f"no JSON files found at path {source!r}"
                raise FileNotFoundError(msg)
            frames = [
                read_json(
                    path,
                    schema=schema,
                    schema_overrides=schema_overrides,
                    infer_schema_length=infer_schema_length,
                    include_file_paths=include_file_paths,
                )
                for path in paths
            ]
            result = F.concat(frames, how="diagonal_relaxed")
            if include_file_paths is not None:
                columns = [c for c in result.columns if c != include_file_paths]
                result = result[[*columns, include_file_paths]]
            return result

    pydf = PyDataFrame.read_json(
        source,
        infer_schema_length=infer_schema_length,
        schema=schema,
        schema_overrides=schema_overrides,
    )
    result = wrap_df(pydf)
    if include_file_paths is not None:
        if include_file_paths in result.columns:
            msg = (
                f"column name for file paths {include_file_paths!r} "
                "conflicts with column name from file"
            )
            raise DuplicateError(msg)
        path = source if isinstance(source, str) else "in-mem"
        result = result.with_columns(
            F.lit(path, dtype=String).alias(include_file_paths)
        )
    return result
