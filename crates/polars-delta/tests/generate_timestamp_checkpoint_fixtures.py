"""Regenerate the timestamp-unit regression checkpoints (requires pyarrow).

Each checkpoint holds two files whose `stats_parsed` bounds for the timestamp
column `t` are written as Parquet TIMESTAMP(unit) or INT96, as different
writers do. File `a.parquet` spans 2024-01-01..2024-01-02 UTC, file
`b.parquet` spans 1500-01-01..1500-01-02 (outside nanosecond range).
"""

import datetime as dt
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

UTC = dt.timezone.utc
RANGES = {
    "a.parquet": (
        dt.datetime(2024, 1, 1, tzinfo=UTC),
        dt.datetime(2024, 1, 2, tzinfo=UTC),
    ),
    "b.parquet": (
        dt.datetime(1500, 1, 1, tzinfo=UTC),
        dt.datetime(1500, 1, 2, tzinfo=UTC),
    ),
}
map_type = pa.map_(pa.string(), pa.string())
metadata = {
    "id": "timestamp-checkpoint",
    "format": {"provider": "parquet", "options": []},
    "schemaString": json.dumps(
        {
            "type": "struct",
            "fields": [
                {"name": "t", "type": "timestamp", "nullable": True, "metadata": {}}
            ],
        }
    ),
    "partitionColumns": [],
    "configuration": [],
}


def write(name: str, unit: str | None) -> None:
    # unit None means INT96.
    ts = pa.timestamp(unit or "us", tz="UTC")
    values = pa.struct([("t", ts)])
    count = pa.struct([("t", pa.int64())])
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
                                    ("tightBounds", pa.bool_()),
                                    ("minValues", values),
                                    ("maxValues", values),
                                    ("nullCount", count),
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
                    [
                        ("minReaderVersion", pa.int32()),
                        ("minWriterVersion", pa.int32()),
                    ]
                ),
            ),
        ]
    )
    rows = [
        {
            "add": {
                "path": path,
                "size": 10,
                "partitionValues": [],
                "stats_parsed": {
                    "numRecords": 5,
                    "tightBounds": True,
                    "minValues": {"t": lo},
                    "maxValues": {"t": hi},
                    "nullCount": {"t": 0},
                },
            }
        }
        for path, (lo, hi) in RANGES.items()
    ]
    rows.append({"metaData": metadata})
    rows.append({"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}})
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        Path(__file__).parent / f"data/timestamp-{name}.checkpoint.parquet",
        compression="zstd",
        use_deprecated_int96_timestamps=unit is None,
        coerce_timestamps=None,
    )


for name, unit in [("ms", "ms"), ("us", "us"), ("ns", "ns"), ("int96", None)]:
    if unit == "ns":
        # pyarrow cannot hold year 1500 in ns; skip file b's range there.
        RANGES_SAVED = dict(RANGES)
        RANGES.pop("b.parquet")
        write(name, unit)
        RANGES.update(RANGES_SAVED)
    else:
        write(name, unit)
