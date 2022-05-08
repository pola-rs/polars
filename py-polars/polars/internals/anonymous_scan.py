import pickle
from functools import partial
from typing import List, Optional

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
