from __future__ import annotations

from typing import TYPE_CHECKING

import polars._reexport as pl

if TYPE_CHECKING:
    from io import IOBase
    from pathlib import Path

    from polars import DataFrame
    from polars.type_aliases import SchemaDefinition


def read_json(
    source: str | Path | IOBase | bytes,
    *,
    schema: SchemaDefinition | None = None,
) -> DataFrame:
    """
    Read into a DataFrame from a JSON file.

    Parameters
    ----------
    source
        Path to a file or a file-like object.
    schema : Sequence of str, (str,DataType) pairs, or a {str:DataType,} dict
        The DataFrame schema may be declared in several ways:

        * As a dict of {name:type} pairs; if type is None, it will be auto-inferred.
        * As a list of column names; in this case types are automatically inferred.
        * As a list of (name,type) pairs; this is equivalent to the dictionary form.

        If you supply a list of column names that does not match the names in the
        underlying data, the names given here will overwrite them. The number
        of names given in the schema should match the underlying data dimensions.

    See Also
    --------
    read_ndjson

    """
    return pl.DataFrame._read_json(source, schema=schema)
