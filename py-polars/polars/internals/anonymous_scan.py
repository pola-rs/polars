from polars import internals as pli
from typing import List, Callable, Dict
import pickle
from functools import partial
from polars.datatypes import Schema

try:
    import pyarrow as pa
    import pyarrow.fs as pa_fs
    import pyarrow.dataset as pa_ds

    _PYARROW_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYARROW_AVAILABLE = False


def _deser_and_exec(buf: bytes, with_columns: List[str]) -> "pli.DataFrame":
    func = pickle.loads(buf)
    return func(with_columns)


def _scan_ds_impl(ds: "pa.dataset.dataset", with_columns: List[str]) -> "pli.DataFrame":
    if not _PYARROW_AVAILABLE:
        raise ImportError(  # pragma: no cover
            "'pyarrow' is required for scanning from pyarrow datasets."
        )
    pli.from_arrow(ds.to_table(columns=with_columns))


def _scan_ds(ds: "pa.dataset.dataset") -> "pli.LazyFrame":
    func = partial(_scan_ds_impl, ds)
    func_serialized = pickle.dumps(func)
    return pli.LazyFrame._scan_python_function(ds.schema, func_serialized)


