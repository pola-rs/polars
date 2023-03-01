from __future__ import annotations

from typing import TYPE_CHECKING

from polars.internals import DataFrame

if TYPE_CHECKING:
    from io import IOBase
    from pathlib import Path


def read_json(file: str | Path | IOBase) -> DataFrame:
    """
    Read into a DataFrame from a JSON file.

    Parameters
    ----------
    file
        Path to a file or a file-like object.

    See Also
    --------
    read_ndjson

    """
    return DataFrame._read_json(file)
