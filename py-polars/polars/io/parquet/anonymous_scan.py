from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import polars.io.parquet
from polars import internals as pli
from polars.dependencies import pickle
from polars.io._utils import _prepare_file_arg

if TYPE_CHECKING:
    from polars.dataframe import DataFrame
    from polars.lazyframe import LazyFrame


def _scan_parquet_fsspec(
    source: str,
    storage_options: dict[str, object] | None = None,
) -> LazyFrame:
    func = partial(_scan_parquet_impl, source, storage_options=storage_options)
    func_serialized = pickle.dumps(func)

    storage_options = storage_options or {}
    with _prepare_file_arg(source, **storage_options) as data:
        schema = polars.io.parquet.read_parquet_schema(data)

    return pli.LazyFrame._scan_python_function(schema, func_serialized)


def _scan_parquet_impl(  # noqa: D417
    uri: str, columns: list[str] | None, **kwargs: Any
) -> DataFrame:
    """
    Take the projected columns and materialize an arrow table.

    Parameters
    ----------
    uri
        Source URI
    columns
        Columns that are projected

    """
    import polars as pl

    return pl.read_parquet(uri, columns=columns, **kwargs)
