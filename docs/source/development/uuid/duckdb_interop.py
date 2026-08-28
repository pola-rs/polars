"""Verify bidirectional UUID Parquet compatibility between Polars and DuckDB."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from uuid import UUID

import duckdb
import polars as pl

VALUES = [
    UUID("a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11"),
    None,
    UUID("019482e4-1441-7aad-8127-eec99573b0a0"),
]

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    polars_path = root / "polars.parquet"
    duckdb_path = root / "duckdb.parquet"

    source = pl.DataFrame({"id": VALUES})
    source.write_parquet(polars_path)

    connection = duckdb.connect()
    duckdb_type = connection.execute(
        "SELECT typeof(id) FROM read_parquet(?) LIMIT 1", [str(polars_path)]
    ).fetchone()[0]
    duckdb_values = connection.execute(
        "SELECT CAST(id AS VARCHAR) FROM read_parquet(?) ORDER BY id NULLS LAST",
        [str(polars_path)],
    ).fetchall()

    connection.execute(
        "COPY (SELECT id::UUID AS id FROM VALUES "
        "('a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11'), (NULL), "
        "('019482e4-1441-7aad-8127-eec99573b0a0') t(id)) "
        "TO ? (FORMAT PARQUET)",
        [str(duckdb_path)],
    )
    from_duckdb = pl.read_parquet(duckdb_path)

    assert duckdb_type == "UUID"
    assert duckdb_values == [
        ("019482e4-1441-7aad-8127-eec99573b0a0",),
        ("a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",),
        (None,),
    ]
    assert from_duckdb.schema == {"id": pl.UUID}
    assert sorted(value for value in from_duckdb["id"].to_list() if value) == sorted(
        value for value in VALUES if value
    )
    assert from_duckdb["id"].null_count() == 1

print(
    json.dumps(
        {
            "duckdb_version": duckdb.__version__,
            "duckdb_reads_polars_type": duckdb_type,
            "polars_reads_duckdb_type": str(from_duckdb.schema["id"]),
            "rows_verified_each_direction": len(VALUES),
        },
        indent=2,
        sort_keys=True,
    )
)
