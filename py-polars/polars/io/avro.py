from __future__ import annotations

from typing import TYPE_CHECKING, BinaryIO

import polars._reexport as pl

if TYPE_CHECKING:
    from io import BytesIO
    from pathlib import Path

    from polars import DataFrame


def read_avro(
    source: str | Path | BytesIO | BinaryIO,
    *,
    columns: list[int] | list[str] | None = None,
    n_rows: int | None = None,
) -> DataFrame:
    """
    Read into a `DataFrame` from an Apache Avro file.

    Parameters
    ----------
    source
        A path to a file or a file-like object. By file-like object, we refer to objects
        that have a `read()` method, such as a file handler (e.g. from the builtin `open
        <https://docs.python.org/3/library/functions.html#open>`_ function) or `BytesIO
        <https://docs.python.org/3/library/io.html#io.BytesIO>`_.
    columns
        A list of column indices (starting at zero) or column names to read.
    n_rows
        The number of rows to read from the Apache Avro file.

    Returns
    -------
    DataFrame

    """
    return pl.DataFrame._read_avro(source, n_rows=n_rows, columns=columns)
