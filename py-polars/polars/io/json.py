from __future__ import annotations

from typing import TYPE_CHECKING

import polars._reexport as pl
from polars.datatypes import N_INFER_DEFAULT

if TYPE_CHECKING:
    from io import IOBase
    from pathlib import Path

    from polars import DataFrame
    from polars.type_aliases import SchemaDefinition


def read_json(
    source: str | Path | IOBase | bytes,
    *,
    infer_schema_length: int | None = N_INFER_DEFAULT,
    schema: SchemaDefinition | None = None,
    schema_overrides: SchemaDefinition | None = None,
) -> DataFrame:
    """
    Read into a `DataFrame` from a JSON file.

    Parameters
    ----------
    source
        A path to a file or a file-like object. By file-like object, we refer to objects
        that have a `read()` method, such as a file handler (e.g. from the builtin `open
        <https://docs.python.org/3/library/functions.html#open>`_ function) or `BytesIO
        <https://docs.python.org/3/library/io.html#io.BytesIO>`_.
    infer_schema_length
        The number of rows to read when inferring the `schema`. If dtypes are inferred
        wrongly (e.g. as :class:`Int64` instead of :class:`Float64`), try to increase
        `infer_schema_length` or specify `schema`.
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

    See Also
    --------
    read_ndjson

    """
    return pl.DataFrame._read_json(
        source,
        infer_schema_length=infer_schema_length,
        schema=schema,
        schema_overrides=schema_overrides,
    )
