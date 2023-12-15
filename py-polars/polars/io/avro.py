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
    Read into a DataFrame from Apache Avro format.

    Parameters
    ----------
    source
        Path to a file or a file-like object (by file-like object, we refer to objects
        that have a `read()` method, such as a file handler (e.g. via builtin `open`
        function) or `BytesIO`).
    columns
        Columns to select. Accepts a list of column indices (starting at zero) or a list
        of column names.
    n_rows
        Stop reading from Apache Avro file after reading `n_rows`.

    Returns
    -------
    DataFrame

    """
    return pl.DataFrame._read_avro(source, n_rows=n_rows, columns=columns)
