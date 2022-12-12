from __future__ import annotations

import pickle
from functools import partial
from typing import Any, cast

import polars as pl
from polars import internals as pli
from polars.dependencies import pyarrow as pa


def _deser_and_exec(
    buf: bytes, with_columns: list[str] | None, *args: Any
) -> pli.DataFrame:
    """
    Deserialize and execute the given function for the projected columns.

    Called from polars-lazy. Polars-lazy provides the bytes of the pickled function and
    the projected columns.

    Parameters
    ----------
    buf
        Pickled function
    with_columns
        Columns that are projected

    """
    func = pickle.loads(buf)
    return func(with_columns, *args)


def _scan_ds_impl(
    ds: pa.dataset.dataset, with_columns: list[str] | None, predicate: str | None
) -> pli.DataFrame:
    """
    Take the projected columns and materialize an arrow table.

    Parameters
    ----------
    ds
        pyarrow dataset
    with_columns
        Columns that are projected
    predicate
        pyarrow expression that can be evaluated with eval

    Returns
    -------
    DataFrame

    """
    _filter = None
    if predicate:
        _filter = eval(predicate)
    return cast(
        pli.DataFrame, pl.from_arrow(ds.to_table(columns=with_columns, filter=_filter))
    )


def _scan_ds(
    ds: pa.dataset.dataset, allow_pyarrow_filter: bool = True
) -> pli.LazyFrame:
    """
    Pickle the partially applied function `_scan_ds_impl`.

    The bytes are then sent to the polars logical plan. It can be deserialized once
    executed and ran.

    Parameters
    ----------
    ds
        pyarrow dataset
    allow_pyarrow_filter
        Allow predicates to be pushed down to pyarrow. This can lead to different
        results if comparisons are done with null values as pyarrow handles this
        different than polars does.

    """
    func = partial(_scan_ds_impl, ds)
    func_serialized = pickle.dumps(func)
    return pli.LazyFrame._scan_python_function(
        ds.schema, func_serialized, allow_pyarrow_filter
    )


def _scan_ipc_impl(
    uri: str, with_columns: list[str] | None, *args: Any
) -> pli.DataFrame:
    """
    Take the projected columns and materialize an arrow table.

    Parameters
    ----------
    uri
        Source URI
    with_columns
        Columns that are projected

    """
    import polars as pl

    return pl.read_ipc(uri, with_columns)


def _scan_ipc_fsspec(
    file: str,
    storage_options: dict[str, object] | None = None,
) -> pli.LazyFrame:
    func = partial(_scan_ipc_impl, file)
    func_serialized = pickle.dumps(func)

    storage_options = storage_options or {}
    with pli._prepare_file_arg(file, **storage_options) as data:
        schema = pli.read_ipc_schema(data)

    return pli.LazyFrame._scan_python_function(schema, func_serialized)


def _scan_parquet_impl(
    uri: str, with_columns: list[str] | None, *args: Any
) -> pli.DataFrame:
    """
    Take the projected columns and materialize an arrow table.

    Parameters
    ----------
    uri
        Source URI
    with_columns
        Columns that are projected

    """
    import polars as pl

    return pl.read_parquet(uri, with_columns)


def _scan_parquet_fsspec(
    file: str,
    storage_options: dict[str, object] | None = None,
) -> pli.LazyFrame:
    func = partial(_scan_parquet_impl, file)
    func_serialized = pickle.dumps(func)

    storage_options = storage_options or {}
    with pli._prepare_file_arg(file, **storage_options) as data:
        schema = pli.read_parquet_schema(data)

    return pli.LazyFrame._scan_python_function(schema, func_serialized)
