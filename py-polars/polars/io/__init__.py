"""Functions for reading data."""

from polars.io.avro import read_avro
from polars.io.csv import read_csv, read_csv_batched, scan_csv
from polars.io.delta import read_delta, scan_delta
from polars.io.excel import read_excel
from polars.io.ipc import read_ipc, scan_ipc
from polars.io.json import read_json
from polars.io.ndjson import read_ndjson, scan_ndjson
from polars.io.parquet import read_parquet, scan_parquet
from polars.io.pyarrow_dataset import scan_ds
from polars.io.sql import read_sql

__all__ = [
    "read_avro",
    "read_csv",
    "read_csv_batched",
    "read_delta",
    "read_excel",
    "read_ipc",
    "read_json",
    "read_ndjson",
    "read_parquet",
    "read_sql",
    "scan_csv",
    "scan_delta",
    "scan_ds",
    "scan_ipc",
    "scan_ndjson",
    "scan_parquet",
]
