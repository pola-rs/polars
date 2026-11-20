"""Regenerate the V2 sidecar, binary-statistics, widened-date, partial-statistics and
binary-text checkpoint fixtures (requires pyarrow)."""

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


def checkpoint(
    name,
    columns,
    add_type,
    rows,
    partitions=(),
    configuration=(),
    with_meta=True,
    row_group_size=None,
    meta_first=False,
):
    """One checkpoint file with `rows` of add actions plus the table metadata and protocol."""
    meta = {
        "id": name,
        "format": {"provider": "parquet", "options": []},
        "schemaString": json.dumps({"type": "struct", "fields": columns}),
        "partitionColumns": list(partitions),
        "configuration": list(configuration),
    }
    schema = pa.schema(
        [
            ("add", add_type),
            (
                "metaData",
                pa.struct(
                    [
                        ("id", pa.string()),
                        (
                            "format",
                            pa.struct(
                                [("provider", pa.string()), ("options", map_type)]
                            ),
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
    rows = [{"add": r} for r in rows]
    if with_meta:
        small = [
            {"metaData": meta},
            {"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}},
        ]
        rows = small + rows if meta_first else rows + small
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        data / f"{name}.checkpoint.parquet",
        compression="zstd",
        row_group_size=row_group_size,
    )


def parsed_type(value_type):
    values = pa.struct([("x", value_type)])
    return pa.struct(
        [
            ("numRecords", pa.int64()),
            ("minValues", values),
            ("maxValues", values),
            ("nullCount", pa.struct([("x", pa.int64())])),
        ]
    )


# A `date` column widened to `timestamp_ntz`: the checkpoint's typed bounds are still dates.
widened = {
    "name": "x",
    "type": "timestamp_ntz",
    "nullable": True,
    "metadata": {
        "delta.typeChanges": [{"fromType": "date", "toType": "timestamp_ntz"}]
    },
}
checkpoint(
    "date-widened",
    [widened],
    pa.struct(
        [
            ("path", pa.string()),
            ("size", pa.int64()),
            ("partitionValues", map_type),
            ("stats_parsed", parsed_type(pa.date32())),
        ]
    ),
    [
        {
            "path": "a.parquet",
            "size": 10,
            "partitionValues": [],
            "stats_parsed": {
                "numRecords": 5,
                "minValues": {"x": 18262},  # 2020-01-01
                "maxValues": {"x": 18292},  # 2020-01-31
                "nullCount": {"x": 0},
            },
        }
    ],
)

# Typed statistics for some files only; the others have the `stats` text.
long_column = [{"name": "x", "type": "long", "nullable": True, "metadata": {}}]
partial_type = pa.struct(
    [
        ("path", pa.string()),
        ("size", pa.int64()),
        ("partitionValues", map_type),
        ("stats", pa.string()),
        ("stats_parsed", parsed_type(pa.int64())),
    ]
)


def text(n, low, high):
    return (
        f'{{"numRecords":{n},"minValues":{{"x":{low}}},'
        f'"maxValues":{{"x":{high}}},"nullCount":{{"x":0}}}}'
    )


checkpoint(
    "partial-parsed",
    long_column,
    partial_type,
    [
        {
            "path": "a.parquet",
            "size": 10,
            "partitionValues": [],
            "stats": text(5, 1, 5),
            "stats_parsed": None,
        },
        {
            "path": "b.parquet",
            "size": 10,
            "partitionValues": [],
            "stats": text(7, 10, 20),
            "stats_parsed": {
                "numRecords": 7,
                "minValues": {"x": 10},
                "maxValues": {"x": 20},
                "nullCount": {"x": 0},
            },
        },
    ],
)

# Partition values stored as plain byte arrays (no string annotation).
binary_map = pa.map_(pa.string(), pa.binary())
for name, value in (
    ("binary-partition", b"north"),
    ("binary-partition-invalid", b"\xff"),
):
    checkpoint(
        name,
        [{"name": "p", "type": "string", "nullable": True, "metadata": {}}],
        pa.struct(
            [
                ("path", pa.string()),
                ("size", pa.int64()),
                ("partitionValues", binary_map),
            ]
        ),
        [
            {
                "path": "a.parquet",
                "size": 10,
                "partitionValues": [("p", value)],
            }
        ],
        partitions=["p"],
    )

# A two-part checkpoint: the table metadata is in the second part only.
checkpoint(
    "multipart-1",
    long_column,
    partial_type,
    [
        {
            "path": "a.parquet",
            "size": 10,
            "partitionValues": [],
            "stats": None,
            "stats_parsed": {
                "numRecords": 5,
                "minValues": {"x": 1},
                "maxValues": {"x": 5},
                "nullCount": {"x": 0},
            },
        }
    ],
    with_meta=False,
)
checkpoint(
    "multipart-2",
    long_column,
    partial_type,
    [
        {
            "path": "b.parquet",
            "size": 10,
            "partitionValues": [],
            "stats": None,
            "stats_parsed": {
                "numRecords": 7,
                "minValues": {"x": 10},
                "maxValues": {"x": 20},
                "nullCount": {"x": 0},
            },
        }
    ],
)


# Multi-row-group checkpoints (row_group_size=2 unless noted).
def parsed(path, low, high, n=5):
    return {
        "path": f"{path}.parquet",
        "size": 10,
        "partitionValues": [],
        "stats": None,
        "stats_parsed": {
            "numRecords": n,
            "minValues": {"x": low},
            "maxValues": {"x": high},
            "nullCount": {"x": 0},
        },
    }


def textual(path, low, high, n=5):
    return {
        "path": f"{path}.parquet",
        "size": 10,
        "partitionValues": [],
        "stats": text(n, low, high),
        "stats_parsed": None,
    }


# Adds spread over row groups, the metadata and protocol rows in the last: [a b | c d | e f | meta protocol].
checkpoint(
    "rg-spread",
    long_column,
    partial_type,
    [parsed(p, i * 10, i * 10 + 5) for i, p in enumerate("abcdef")],
    row_group_size=2,
)
# One row group holds only metadata and protocol, the others only adds: [meta protocol | a b | c d].
checkpoint(
    "rg-meta-only",
    long_column,
    partial_type,
    [parsed(p, i * 10, i * 10 + 5) for i, p in enumerate("abcd")],
    row_group_size=2,
    meta_first=True,
)
# Only the middle row group needs the `stats` text: [a b | c(text) d | e meta | protocol].
checkpoint(
    "rg-stats-text",
    long_column,
    partial_type,
    [
        parsed("a", 0, 5),
        parsed("b", 10, 15),
        textual("c", 20, 25),
        parsed("d", 30, 35),
        parsed("e", 40, 45),
    ],
    row_group_size=2,
)
# Two parts of several row groups each; the metadata is in the second part's first row group.
checkpoint(
    "rg-multipart-1",
    long_column,
    partial_type,
    [parsed(p, i * 10, i * 10 + 5) for i, p in enumerate("abcde")],
    with_meta=False,
    row_group_size=2,
)
checkpoint(
    "rg-multipart-2",
    long_column,
    partial_type,
    [
        textual("f", 50, 55),
        parsed("g", 60, 65),
        parsed("h", 70, 75),
        textual("i", 80, 85),
    ],
    row_group_size=2,
    meta_first=True,
)
