"""
Script to generate data for running the TPC-H benchmark.

Data generation logic was adapted from the TPC-H benchmark tools:
https://www.tpc.org/tpch/
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import polars as pl

logging.basicConfig(level=logging.INFO)

CURRENT_DIR = Path(__file__).parent
DBGEN_DIR = CURRENT_DIR / "dbgen"

__all__ = ["load_tpch_table"]


def load_tpch_table(table_name: str, scale_factor: float = 0.01) -> pl.DataFrame:
    """
    Load a TPC-H table from disk.

    If the file does not exist, it is generated along with all other tables.
    """
    folder = CURRENT_DIR / f"sf-{scale_factor:g}"
    file_path = folder / f"{table_name}.parquet"

    if not file_path.exists():
        _generate_tpch_data(scale_factor)

    return pl.read_parquet(file_path)


def _generate_tpch_data(scale_factor: float = 0.01) -> None:
    """Generate all TPC-H datasets with the given scale factor."""
    # TODO: Can we make this work under Windows?
    if sys.platform == "win32":
        msg = "cannot generate TPC-H data under Windows"
        raise RuntimeError(msg)

    subprocess.run(["./dbgen", "-f", "-v", "-s", str(scale_factor)], cwd=DBGEN_DIR)

    _process_data(scale_factor)


def _process_data(scale_factor: float = 0.01) -> None:
    """Process the data into Parquet files with the correct schema."""
    dest = CURRENT_DIR / f"sf-{scale_factor:g}"
    dest.mkdir(exist_ok=True)

    for table_name, columns in TABLE_COLUMN_NAMES.items():
        logging.info(f"Processing table: {table_name}")

        table_path = DBGEN_DIR / f"{table_name}.tbl"
        lf = pl.scan_csv(
            table_path,
            has_header=False,
            separator="|",
            try_parse_dates=True,
            new_columns=columns,
        )

        # Drop empty last column because CSV ends with a separator
        lf = lf.select(columns)

        lf.sink_parquet(dest / f"{table_name}.parquet")
        table_path.unlink()


TABLE_COLUMN_NAMES = {
    "customer": [
        "c_custkey",
        "c_name",
        "c_address",
        "c_nationkey",
        "c_phone",
        "c_acctbal",
        "c_mktsegment",
        "c_comment",
    ],
    "lineitem": [
        "l_orderkey",
        "l_partkey",
        "l_suppkey",
        "l_linenumber",
        "l_quantity",
        "l_extendedprice",
        "l_discount",
        "l_tax",
        "l_returnflag",
        "l_linestatus",
        "l_shipdate",
        "l_commitdate",
        "l_receiptdate",
        "l_shipinstruct",
        "l_shipmode",
        "comments",
    ],
    "nation": [
        "n_nationkey",
        "n_name",
        "n_regionkey",
        "n_comment",
    ],
    "orders": [
        "o_orderkey",
        "o_custkey",
        "o_orderstatus",
        "o_totalprice",
        "o_orderdate",
        "o_orderpriority",
        "o_clerk",
        "o_shippriority",
        "o_comment",
    ],
    "part": [
        "p_partkey",
        "p_name",
        "p_mfgr",
        "p_brand",
        "p_type",
        "p_size",
        "p_container",
        "p_retailprice",
        "p_comment",
    ],
    "partsupp": [
        "ps_partkey",
        "ps_suppkey",
        "ps_availqty",
        "ps_supplycost",
        "ps_comment",
    ],
    "region": [
        "r_regionkey",
        "r_name",
        "r_comment",
    ],
    "supplier": [
        "s_suppkey",
        "s_name",
        "s_address",
        "s_nationkey",
        "s_phone",
        "s_acctbal",
        "s_comment",
    ],
}
