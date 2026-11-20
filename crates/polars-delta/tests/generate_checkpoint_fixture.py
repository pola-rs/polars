"""Regenerate the small wide/binary-partition regression checkpoint (requires pyarrow)."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

width = 128
names = [f"c{i}" for i in range(width)]
fields = [
    {"name": name, "type": "long", "nullable": True, "metadata": {}} for name in names
]
fields.append({"name": "p", "type": "binary", "nullable": True, "metadata": {}})
metadata = {
    "id": "wide-checkpoint-regression",
    "format": {"provider": "parquet", "options": []},
    "schemaString": json.dumps({"type": "struct", "fields": fields}),
    "partitionColumns": ["p"],
    "configuration": [],
}
map_type = pa.map_(pa.string(), pa.string())
values = pa.struct([(name, pa.int64()) for name in names])
schema = pa.schema(
    [
        (
            "add",
            pa.struct(
                [
                    ("path", pa.string()),
                    ("size", pa.int64()),
                    ("partitionValues", map_type),
                    ("partitionValues_parsed", pa.struct([("p", pa.binary())])),
                    (
                        "stats_parsed",
                        pa.struct(
                            [
                                ("numRecords", pa.int64()),
                                ("tightBounds", pa.bool_()),
                                ("minValues", values),
                                ("maxValues", values),
                                ("nullCount", values),
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
            "partitionValues": [("p", "hi")],
            "partitionValues_parsed": {"p": b"hi"},
            "stats_parsed": {
                "numRecords": 5,
                "tightBounds": True,
                "minValues": dict.fromkeys(names, 1),
                "maxValues": dict.fromkeys(names, 5),
                "nullCount": dict.fromkeys(names, 0),
            },
        }
    },
    {"metaData": metadata},
    {"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}},
]
pq.write_table(
    pa.Table.from_pylist(rows, schema=schema),
    Path(__file__).parent / "data/wide-binary.checkpoint.parquet",
    compression="zstd",
)
