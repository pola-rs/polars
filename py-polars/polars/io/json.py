from __future__ import annotations

from typing import TYPE_CHECKING

from polars.internals import DataFrame
from polars.utils.decorators import deprecated_alias

if TYPE_CHECKING:
    from io import IOBase
    from pathlib import Path


@deprecated_alias(file="source")
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
    return DataFrame._read_json(source)
