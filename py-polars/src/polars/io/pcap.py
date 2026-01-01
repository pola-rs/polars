from __future__ import annotations

import contextlib
from pathlib import Path
from typing import IO, TYPE_CHECKING

from polars._utils.various import normalize_filepath
from polars._utils.wrap import wrap_df

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars._plr import PyDataFrame

if TYPE_CHECKING:
    from polars import DataFrame


def read_pcap(
    source: str | Path | IO[bytes] | bytes,
    *,
    n_rows: int | None = None,
) -> DataFrame:
    """
    Read into a DataFrame from PCAP format.

    Parameters
    ----------
    source
        Path to a file or a file-like object (by "file-like object" we refer to objects
        that have a `read()` method, such as a file handler like the builtin `open`
        function, or a `BytesIO` instance).
    n_rows
        Stop reading from PCAP file after reading `n_rows`.

    Returns
    -------
    DataFrame
    """
    if isinstance(source, (str, Path)):
        source = normalize_filepath(source)

    pydf = PyDataFrame.read_pcap(source, n_rows, True)
    return wrap_df(pydf)
