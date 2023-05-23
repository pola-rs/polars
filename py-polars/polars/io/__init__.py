"""Functions for reading data."""

from polars.io.avro import read_avro
from polars.io.csv import read_csv, read_csv_batched, scan_csv
from polars.io.database import read_database
from polars.io.delta import read_delta, scan_delta
from polars.io.excel import read_excel
from polars.io.ipc import read_ipc, read_ipc_schema, scan_ipc
from polars.io.json import read_json
from polars.io.ndjson import read_ndjson, scan_ndjson
from polars.io.parquet import read_parquet, read_parquet_schema, scan_parquet
from polars.io.pyarrow_dataset import scan_ds, scan_pyarrow_dataset

__all__ = [
    "read_avro",
    "read_csv",
    "read_csv_batched",
    "read_database",
    "read_delta",
    "read_excel",
    "read_ipc",
    "read_ipc_schema",
    "read_json",
    "read_ndjson",
    "read_parquet",
    "read_parquet_schema",
    "scan_csv",
    "scan_delta",
    "scan_ds",
    "scan_ipc",
    "scan_ndjson",
    "scan_parquet",
    "scan_pyarrow_dataset",
]
