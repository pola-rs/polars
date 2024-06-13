from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import polars._reexport as pl
import polars.io.ipc
from polars.io._utils import prepare_file_arg

if TYPE_CHECKING:
    from polars import DataFrame, LazyFrame


def _scan_ipc_fsspec(
    source: str,
    storage_options: dict[str, object] | None = None,
) -> LazyFrame:
    func = partial(_scan_ipc_impl, source, storage_options=storage_options)

    with prepare_file_arg(source, storage_options=storage_options) as data:
        schema = polars.io.ipc.read_ipc_schema(data)

    return pl.LazyFrame._scan_python_function(schema, func)


def _scan_ipc_impl(  # noqa: D417
    source: str,
    columns: list[str] | None,
    predicate: str | None,
    n_rows: int | None,
    **kwargs: Any,
) -> DataFrame:
    """
    Take the projected columns and materialize an arrow table.

    Parameters
    ----------
    source
        Source URI
    columns
        Columns that are projected
    """
    from polars import read_ipc

    return read_ipc(source, columns=columns, n_rows=n_rows, **kwargs)
