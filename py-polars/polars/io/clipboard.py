from __future__ import annotations

import contextlib
from typing import Any, TYPE_CHECKING

from polars.io.csv.functions import read_csv

from io import StringIO

with contextlib.suppress(ImportError):
    from polars.polars import read_clipboard_string as _read_clipboard_string

if TYPE_CHECKING:
    from polars import DataFrame


def read_clipboard(separator: str = "\t", **kwargs: Any) -> DataFrame:
    csv_string: str = _read_clipboard_string()
    io_string = StringIO(csv_string)
    return read_csv(source=io_string, separator=separator, **kwargs)
