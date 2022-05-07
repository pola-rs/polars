from polars import internals as pli
from typing import List, Callable, Dict
import pickle
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


def register_callback(fn: Callable[[List[str]], "pli.DataFrame"], schema: Dict[str, "pli.DataType"]) -> "pli.LazyFrame":
    buf = pickle.dumps(fn)


def _scan_parquet_s3_impl(with_columns: List[str]) -> "pli.DataFrame":
    if not _PYARROW_AVAILABLE:
        raise ImportError(  # pragma: no cover
            "'pyarrow' is required for scanning from s3."
        )
    s3 = pa_fs.S3FileSystem()
    ds = pa_ds.dataset(
        format="parquet",
        filesystem=s3,
    )
    pli.from_arrow(ds.to_table(columns=with_columns))


def _scan_parquet_s3(source: str) -> "pli.LazyFrame":


    buf = pickle.dumps(_scan_parquet_s3_impl)


