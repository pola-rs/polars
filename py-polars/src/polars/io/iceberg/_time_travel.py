"""Resolve a SQL time travel clause to the snapshot it reads."""

from __future__ import annotations

import gzip
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyiceberg.table
    from pyiceberg.table.metadata import TableMetadata

# ("snapshot", id), ("ref", name) or ("timestamp", microseconds since the epoch)
TableVersion = tuple[str, int | str]

# A catalog may send only the snapshots of branches and tags (REST `snapshots=refs`).
_NOT_SENT = (
    "or the catalog did not send it: attach the catalog with the property "
    "`snapshot-loading-mode` = `all`"
)


def resolve_version(table: pyiceberg.table.Table, version: TableVersion) -> int:
    """The id of the snapshot that `version` reads on `table`."""
    kind, value = version
    if kind == "snapshot":
        assert isinstance(value, int)
        snapshot_id = value
    elif kind == "ref":
        ref = table.metadata.refs.get(str(value))
        if ref is None:
            msg = f"iceberg branch or tag not found: {value!r}"
            raise ValueError(msg)
        return ref.snapshot_id
    else:
        assert kind == "timestamp"
        assert isinstance(value, int)
        snapshot_id = _snapshot_as_of(table, value)
    if table.snapshot_by_id(snapshot_id) is None:
        msg = f"iceberg snapshot ID not found: {snapshot_id}"
        # a snapshot the table had: gone, or not sent
        if any(e.snapshot_id == snapshot_id for e in table.metadata.snapshot_log):
            msg += f" (it expired, {_NOT_SENT})"
        raise ValueError(msg)
    return snapshot_id


def _snapshot_as_of(table: pyiceberg.table.Table, micros: int) -> int:
    """
    The snapshot current on the main branch at `micros`.

    Each commit writes a metadata file. The one current at `micros` is the newest file
    committed at or before then, from the metadata log; it names the current snapshot.
    Times before the metadata log are refused.
    """
    metadata = table.metadata
    if not metadata.snapshots:
        msg = "iceberg table has no snapshots"
        raise ValueError(msg)
    files = [(e.metadata_file, e.timestamp_ms) for e in metadata.metadata_log]
    files.append((table.metadata_location, metadata.last_updated_ms))
    found = next(
        (i for i in reversed(range(len(files))) if files[i][1] * 1000 <= micros), None
    )
    unavailable = f"iceberg table history at {_at(micros)} is not available"
    if found is None:
        msg = (
            f"{unavailable}: its oldest recorded metadata file is from "
            f"{_at(files[0][1] * 1000)} (see the table property "
            "`write.metadata.previous-versions-max`)"
        )
        raise ValueError(msg)
    if any(stamp * 1000 > micros for _, stamp in files[:found]):
        msg = f"{unavailable}: the commit times in its metadata log are out of order"
        raise ValueError(msg)

    location, stamp = files[found]
    if found + 1 == len(files):
        snapshot_id = metadata.current_snapshot_id
    else:
        then = _read_metadata(table, location, unavailable)
        if then.table_uuid != metadata.table_uuid or then.last_updated_ms != stamp:
            msg = (
                f"{unavailable}: metadata file {location} does not match its log entry"
            )
            raise ValueError(msg)
        snapshot_id = then.current_snapshot_id
    if snapshot_id is None:
        msg = f"iceberg table has no snapshot at or before {_at(micros)}"
        raise ValueError(msg)
    if table.snapshot_by_id(snapshot_id) is None:
        msg = (
            f"{unavailable}: snapshot {snapshot_id}, current then, has expired, "
            f"{_NOT_SENT}"
        )
        raise ValueError(msg)
    return snapshot_id


def _read_metadata(
    table: pyiceberg.table.Table, location: str, unavailable: str
) -> TableMetadata:
    """A logged metadata file, gzipped or not, whatever its name."""
    from pyiceberg.table.metadata import TableMetadataUtil

    try:
        with table.io.new_input(location).open() as stream:
            data = stream.read()
        if data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)
        return TableMetadataUtil.parse_raw(data.decode())
    except Exception as e:
        msg = f"{unavailable}: metadata file {location} could not be read ({type(e).__name__})"
        raise ValueError(msg) from e


def _at(micros: int) -> datetime | str:
    try:
        return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=micros)
    except OverflowError:
        return f"{micros} microseconds since the epoch"
