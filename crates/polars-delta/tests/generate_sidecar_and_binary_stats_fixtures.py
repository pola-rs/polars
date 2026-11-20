"""Regenerate the V2 sidecar and binary-statistics checkpoint fixtures (requires pyarrow)."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

data = Path(__file__).parent / "data"
map_type = pa.map_(pa.string(), pa.string())

# A sidecar holds file actions only.
add_type = pa.struct(
    [
        ("path", pa.string()),
        ("size", pa.int64()),
        ("partitionValues", map_type),
        ("stats", pa.string()),
    ]
)
stats = '{"numRecords":5,"minValues":{"x":1},"maxValues":{"x":5},"nullCount":{"x":0}}'
rows = [
    {"add": {"path": "a.parquet", "size": 10, "partitionValues": [], "stats": stats}},
    {"add": {"path": "b.parquet", "size": 10, "partitionValues": [], "stats": stats}},
]
pq.write_table(
    pa.Table.from_pylist(rows, schema=pa.schema([("add", add_type)])),
    data / "v2-sidecar.sidecar.parquet",
    compression="zstd",
)

# A checkpoint whose typed statistics hold a column of an unusable type (binary).
columns = [
    {"name": "t", "type": "long", "nullable": True, "metadata": {}},
    {"name": "b", "type": "binary", "nullable": True, "metadata": {}},
]
metadata = {
    "id": "binary-stats",
    "format": {"provider": "parquet", "options": []},
    "schemaString": json.dumps({"type": "struct", "fields": columns}),
    "partitionColumns": [],
    "configuration": [],
}
values = pa.struct([("t", pa.int64()), ("b", pa.binary())])
counts = pa.struct([("t", pa.int64()), ("b", pa.int64())])
schema = pa.schema(
    [
        (
            "add",
            pa.struct(
                [
                    ("path", pa.string()),
                    ("size", pa.int64()),
                    ("partitionValues", map_type),
                    (
                        "stats_parsed",
                        pa.struct(
                            [
                                ("numRecords", pa.int64()),
                                ("minValues", values),
                                ("maxValues", values),
                                ("nullCount", counts),
                            ]
                        ),
                    ),
                ]
            ),
        ),
        (
            "metaData",
            pa.struct(
                [
                    ("id", pa.string()),
                    (
                        "format",
                        pa.struct([("provider", pa.string()), ("options", map_type)]),
                    ),
                    ("schemaString", pa.string()),
                    ("partitionColumns", pa.list_(pa.string())),
                    ("configuration", map_type),
                ]
            ),
        ),
        (
            "protocol",
            pa.struct(
                [("minReaderVersion", pa.int32()), ("minWriterVersion", pa.int32())]
            ),
        ),
    ]
)
rows = [
    {
        "add": {
            "path": "a.parquet",
            "size": 10,
            "partitionValues": [],
            "stats_parsed": {
                "numRecords": 5,
                "minValues": {"t": 1, "b": b"\x00"},
                "maxValues": {"t": 5, "b": b"\xff"},
                "nullCount": {"t": 0, "b": 0},
            },
        }
    },
    {"metaData": metadata},
    {"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}},
]
pq.write_table(
    pa.Table.from_pylist(rows, schema=schema),
    data / "binary-stats.checkpoint.parquet",
    compression="zstd",
)
