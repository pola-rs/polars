from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

from polars.internals import DataFrame
from polars.utils.decorators import deprecate_nonkeyword_arguments
from polars.utils.various import normalise_filepath

if TYPE_CHECKING:
    from io import BytesIO


@deprecate_nonkeyword_arguments()
def read_avro(
    file: str | Path | BytesIO | BinaryIO,
    columns: list[int] | list[str] | None = None,
    n_rows: int | None = None,
) -> DataFrame:
    """
    Read into a DataFrame from Apache Avro format.

    Parameters
    ----------
    file
        Path to a file or a file-like object.
    columns
        Columns to select. Accepts a list of column indices (starting at zero) or a list
        of column names.
    n_rows
        Stop reading from Apache Avro file after reading ``n_rows``.

    Returns
    -------
    DataFrame

    """
    if isinstance(file, (str, Path)):
        file = normalise_filepath(file)

    return DataFrame._read_avro(file, n_rows=n_rows, columns=columns)
