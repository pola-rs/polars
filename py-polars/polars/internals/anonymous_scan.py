import pickle
from functools import partial
from typing import Dict, List, Optional

import polars as pl
from polars import internals as pli

try:
    import pyarrow as pa

    _PYARROW_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYARROW_AVAILABLE = False


def _deser_and_exec(buf: bytes, with_columns: Optional[List[str]]) -> "pli.DataFrame":
    """
    Called from polars-lazy. Polars-lazy provides the bytes of the pickled function and the
    projected columns.

    Parameters
    ----------
    buf
        Pickled function
    with_columns
        Columns that are projected
    """
    func = pickle.loads(buf)
    return func(with_columns)


def _scan_ds_impl(
    ds: "pa.dataset.dataset", with_columns: Optional[List[str]]
) -> "pli.DataFrame":
    """
    Takes the projected columns and materializes an arrow table.

    Parameters
    ----------
    ds
    with_columns

    Returns
    -------

    """
    if not _PYARROW_AVAILABLE:
        raise ImportError(  # pragma: no cover
            "'pyarrow' is required for scanning from pyarrow datasets."
        )
    return pl.from_arrow(ds.to_table(columns=with_columns))  # type: ignore


def _scan_ds(ds: "pa.dataset.dataset") -> "pli.LazyFrame":
    """
    This pickles the partially applied function `_scan_ds_impl`. That bytes are then send to in the polars
    logical plan. It can be deserialized once executed and ran.

    Parameters
    ----------
    ds
        pyarrow dataset
    """
    func = partial(_scan_ds_impl, ds)
    func_serialized = pickle.dumps(func)
    return pli.LazyFrame._scan_python_function(ds.schema, func_serialized)


def _scan_ipc_impl(uri: "str", with_columns: Optional[List[str]]) -> "pli.DataFrame":
    """
    Takes the projected columns and materializes an arrow table.

    Parameters
    ----------
    uri
    with_columns
    """
    import polars as pl

    return pl.read_ipc(uri, with_columns)


def _scan_ipc_fsspec(
    file: str,
    storage_options: Optional[Dict] = None,
) -> "pli.LazyFrame":
    func = partial(_scan_ipc_impl, file)
    func_serialized = pickle.dumps(func)

    storage_options = storage_options or {}
    with pli._prepare_file_arg(file, **storage_options) as data:
        schema = pli.read_ipc_schema(data)

    return pli.LazyFrame._scan_python_function(schema, func_serialized)


def _scan_parquet_impl(
    uri: "str", with_columns: Optional[List[str]]
) -> "pli.DataFrame":
    """
    Takes the projected columns and materializes an arrow table.

    Parameters
    ----------
    uri
    with_columns
    """
    import polars as pl

    return pl.read_parquet(uri, with_columns)


def _scan_parquet_fsspec(
    file: str,
    storage_options: Optional[Dict] = None,
) -> "pli.LazyFrame":
    func = partial(_scan_parquet_impl, file)
    func_serialized = pickle.dumps(func)

    storage_options = storage_options or {}
    with pli._prepare_file_arg(file, **storage_options) as data:
        schema = pli.read_parquet_schema(data)

    return pli.LazyFrame._scan_python_function(schema, func_serialized)
