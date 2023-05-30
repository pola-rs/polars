from __future__ import annotations

from typing import TYPE_CHECKING

import polars._reexport as pl

if TYPE_CHECKING:
    from io import IOBase
    from pathlib import Path

    from polars import DataFrame


def read_json(source: str | Path | IOBase) -> DataFrame:
    """
    Read into a DataFrame from a JSON file.

    Parameters
    ----------
    source
        Path to a file or a file-like object.

    See Also
    --------
    read_ndjson

    """
    return pl.DataFrame._read_json(source)
