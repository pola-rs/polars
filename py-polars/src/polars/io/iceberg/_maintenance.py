"""Iceberg table maintenance, run by SQL `CALL <catalog>.system.<procedure>(...)`."""

from __future__ import annotations

import contextlib
import functools
import math
import struct
import uuid
from collections import Counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Collection

    import pyiceberg.table
    from pyiceberg.manifest import ManifestEntry, ManifestFile
    from pyiceberg.table import Transaction
    from pyiceberg.table.metadata import TableMetadata
    from pyiceberg.typedef import Record

# Iceberg's default of `commit.manifest.target-size-bytes`.
_MANIFEST_TARGET_SIZE_BYTES = 8 * 1024 * 1024


@functools.cache
def _pyiceberg_rounds_doubles() -> bool:
    """Whether PyIceberg reads Avro doubles as 32-bit floats, as 0.12.0 does."""
    from pyiceberg.avro.decoder import new_decoder

    return bool(new_decoder(struct.pack("<d", 0.1)).read_double() != 0.1)


def _double_partition_fields(
    metadata: TableMetadata, spec_ids: Collection[int] | None = None
) -> list[str]:
    """The partition fields of the specs (default: all) whose source was ever DOUBLE."""
    from pyiceberg.transforms import IdentityTransform
    from pyiceberg.types import DoubleType

    # identity is the only transform with DOUBLE values; the source may be dropped since
    return [
        field.name
        for spec in metadata.partition_specs
        if spec_ids is None or spec.spec_id in spec_ids
        for field in spec.fields
        if isinstance(field.transform, IdentityTransform)
        and any(
            isinstance(_source_type(schema, field.source_id), DoubleType)
            for schema in metadata.schemas
        )
    ]


def _source_type(schema: Any, source_id: int) -> Any:
    try:
        return schema.find_field(source_id).field_type
    except ValueError:
        return None


def _rounded_doubles_error(
    metadata: TableMetadata,
    name: str,
    operation: str,
    spec_ids: Collection[int] | None = None,
) -> str | None:
    """Why `operation`, rewriting manifests, would store rounded values, if it would.

    PyIceberg 0.12.0 reads DOUBLE partition values as 32-bit floats, and a rewritten
    manifest stores what was read (#30832).
    """
    if not _pyiceberg_rounds_doubles():
        return None
    if fields := _double_partition_fields(metadata, spec_ids):
        return (
            f"{operation} of Iceberg table '{name}' is not supported with this "
            f"PyIceberg version: it reads the DOUBLE partition values of {fields} as "
            "32-bit floats, and the commit would store them rounded. A PyIceberg release "
            "with apache/iceberg-python#3808 fixes this. Manifests already written this "
            "way keep their rounded values: upgrading does not repair them."
        )
    return None


class _NothingToRewrite(Exception):
    """A retry found nothing left to rewrite: a concurrent commit did it."""


def rewrite_manifests(
    table: pyiceberg.table.Table, *, spec_id: int | None = None
) -> tuple[int, int]:
    """
    Rewrite the data manifests of a partition spec of `table` in one `replace` commit.

    Live entries keep their snapshot ids and sequence numbers. They are sorted by
    partition and split into manifests of about `commit.manifest.target-size-bytes`.
    Delete manifests and other specs' manifests stay as they are. Nothing is committed
    if the spec (default: the current one) has no data manifest, or one that fits the
    target size. Returns the numbers of rewritten and added manifests.
    """
    from pyiceberg.exceptions import CommitFailedException, CommitStateUnknownException
    from pyiceberg.table import Table

    from polars.io.iceberg._sink import (
        _SINK_UUID_PROPERTY,
        _AlreadyCommitted,
        _sink_transaction,
        _SinkCatalog,
    )

    name = ".".join(table.name())
    if table.format_version > 2:
        msg = (
            f"rewrite_manifests of an Iceberg v{table.format_version} table is not "
            f"supported yet: table '{name}'"
        )
        raise NotImplementedError(msg)
    specs = table.specs()
    if spec_id is None:
        spec_id = table.metadata.default_spec_id
    elif spec_id not in specs:
        msg = f"spec_id {spec_id} is not a partition spec of table '{name}': {sorted(specs)}"
        raise ValueError(msg)
    schema = table.schema()
    for field in specs[spec_id].fields:
        # PyIceberg writes manifests with the current schema
        if _source_type(schema, field.source_id) is None:
            msg = (
                f"rewrite_manifests of spec {spec_id} of table '{name}' is not supported: "
                f"the source column of its partition field '{field.name}' was dropped"
            )
            raise NotImplementedError(msg)
    # only this spec's manifests are read and written again
    if rounded := _rounded_doubles_error(
        table.metadata, name, "rewrite_manifests", [spec_id]
    ):
        raise NotImplementedError(rounded)

    snapshot = table.current_snapshot()
    target = _target_size(table.metadata)
    matching = (
        _matching_manifests(snapshot.manifests(table.io), spec_id)
        if snapshot is not None
        else []
    )
    if not _needs_rewrite(matching, target):
        return 0, 0

    # as a sink's commit: a failed commit is checked for having landed
    commit_uuid = str(uuid.uuid4())
    catalog = _SinkCatalog(table.catalog, commit_uuid, verbose=False)
    wrapped = Table(
        table.name(),
        table.metadata,
        table.metadata_location,
        table.io,
        catalog,  # type: ignore[arg-type]
        table.config,
    )
    producers: list[Any] = []

    def counts() -> tuple[int, int]:
        # of the last planned attempt, the one that landed
        return producers[0]._replaced, producers[0]._created

    def clean_up() -> None:
        for producer in producers:
            producer._clean_all_uncommitted()

    try:
        with _sink_transaction(wrapped, commit_uuid, None) as tx:
            producers.append(
                _producer(tx, name, spec_id, target, {_SINK_UUID_PROPERTY: commit_uuid})
            )
            producers[0].commit()
    except _AlreadyCommitted:
        return counts()
    except CommitStateUnknownException:
        raise
    except Exception as error:
        # not the last request's own refusal: did the rewrite commit?
        if catalog._snapshot_ids and error is not catalog._raised:
            if _landed(table, commit_uuid):
                # failed after this rewrite committed
                return counts()
            if not isinstance(catalog._raised, CommitFailedException):
                msg = "a failed Iceberg commit may still land"
                raise CommitStateUnknownException(msg) from error
        # nothing of this rewrite landed: no request, or every one refused. PyIceberg
        # removes the manifests of some refusals only; removing twice is a no-op.
        clean_up()
        if isinstance(error, _NothingToRewrite):
            return 0, 0
        raise
    return counts()


def _landed(table: pyiceberg.table.Table, commit_uuid: str) -> bool:
    """Whether the commit `commit_uuid` is in `table`'s catalog."""
    from pyiceberg.exceptions import CommitStateUnknownException

    from polars.io.iceberg._sink import _already_committed

    try:
        current = table.catalog.load_table(table.name())
    except Exception as load_error:
        msg = "a failed Iceberg commit may still land"
        raise CommitStateUnknownException(msg) from load_error
    return _already_committed(current.metadata, commit_uuid)


def _target_size(metadata: TableMetadata) -> int:
    value = metadata.properties.get("commit.manifest.target-size-bytes")
    if not value:
        return _MANIFEST_TARGET_SIZE_BYTES
    if int(value) <= 0:
        msg = f"commit.manifest.target-size-bytes must be positive, found {value}"
        raise ValueError(msg)
    return int(value)


def _matching_manifests(
    manifests: list[ManifestFile], spec_id: int
) -> list[ManifestFile]:
    from pyiceberg.manifest import ManifestContent

    return [
        m
        for m in manifests
        if m.content == ManifestContent.DATA and m.partition_spec_id == spec_id
    ]


def _needs_rewrite(manifests: list[ManifestFile], target: int) -> bool:
    """Whether `manifests` need a rewrite: not none, nor one that fits the target."""
    return bool(manifests) and not (
        len(manifests) == 1 and _manifest_count(manifests, target) == 1
    )


def _manifest_count(manifests: list[ManifestFile], target: int) -> int:
    """The number of manifests to write for `manifests`, as Iceberg Java."""
    size = sum(m.manifest_length for m in manifests)
    return max(1, math.ceil(size / target))


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _partition_values(partition: Record) -> list[Any]:
    return [partition[i] for i in range(len(partition))]


def _partition_key(partition: Record) -> tuple[Any, ...]:
    # NaN after numbers, nulls last; a field's other values have one type
    return tuple(
        (v is None, _is_nan(v), None if v is None or _is_nan(v) else v)
        for v in _partition_values(partition)
    )


def _entry_key(entry: ManifestEntry) -> tuple[Any, ...]:
    return (
        entry.data_file.file_path,
        # NaN never equals itself
        tuple(
            "NaN" if _is_nan(v) else v
            for v in _partition_values(entry.data_file.partition)
        ),
        entry.snapshot_id,
        # v1 manifests store no sequence numbers: read as 0, or unset once rewritten
        entry.sequence_number or 0,
        entry.file_sequence_number or 0,
    )


def _check_same_entries(
    written: list[ManifestEntry], entries: list[ManifestEntry]
) -> None:
    if Counter(map(_entry_key, written)) != Counter(map(_entry_key, entries)):
        msg = "rewrite_manifests: the new manifests do not match the old ones"
        raise ValueError(msg)


def _producer(
    tx: Transaction,
    name: str,
    spec_id: int,
    target: int,
    snapshot_properties: dict[str, str],
) -> Any:
    from pyiceberg.exceptions import ValidationException
    from pyiceberg.manifest import ManifestEntry, ManifestEntryStatus
    from pyiceberg.table.snapshots import Operation, Summary, update_snapshot_summaries
    from pyiceberg.table.update.snapshot import _SnapshotProducer
    from pyiceberg.typedef import EMPTY_DICT

    # private in PyIceberg 0.12, as its own producers
    class _RewriteManifests(_SnapshotProducer):  # type: ignore[misc,type-arg]
        _replaced: int = 0
        _created: int = 0
        _kept: int = 0
        _entries: int = 0

        def _parent(self) -> Any:
            if self._parent_snapshot_id is None:
                return None
            return self._transaction.table_metadata.snapshot_by_id(
                self._parent_snapshot_id
            )

        def _validate_concurrency(self) -> None:
            # no data changes: nothing a concurrent commit could conflict with
            pass

        def _deleted_entries(self) -> list[ManifestEntry]:
            return []

        def _existing_manifests(self) -> list[ManifestFile]:
            # planned on each attempt, from the table as that attempt sees it
            metadata = self._transaction.table_metadata
            if msg := _rounded_doubles_error(
                metadata, name, "rewrite_manifests", [spec_id]
            ):
                raise ValidationException(msg)
            parent = self._parent()
            manifests = parent.manifests(self._io) if parent is not None else []
            rewritten = _matching_manifests(manifests, spec_id)
            if not _needs_rewrite(rewritten, target):
                raise _NothingToRewrite
            paths = {m.manifest_path for m in rewritten}
            kept = [m for m in manifests if m.manifest_path not in paths]
            entries = [e for m in rewritten for e in self.fetch_manifest_entry(m, True)]
            entries.sort(key=lambda e: _partition_key(e.data_file.partition))
            # no empty manifests, also when no entry is live any more
            n = min(_manifest_count(rewritten, target), len(entries))
            added: list[ManifestFile] = []
            try:
                for i in range(n):
                    chunk = entries[i * len(entries) // n : (i + 1) * len(entries) // n]
                    with self.new_manifest_writer(self.spec(spec_id)) as writer:
                        for e in chunk:
                            writer.add_entry(
                                ManifestEntry.from_args(
                                    status=ManifestEntryStatus.EXISTING,
                                    snapshot_id=e.snapshot_id,
                                    sequence_number=e.sequence_number,
                                    file_sequence_number=e.file_sequence_number,
                                    data_file=e.data_file,
                                )
                            )
                    added.append(writer.to_manifest_file())
                # the new manifests must hold exactly the live entries they replace
                _check_same_entries(
                    [e for m in added for e in self.fetch_manifest_entry(m, True)],
                    entries,
                )
            except BaseException:
                # this attempt's manifests, before any commit request uses them
                for path in self._written_manifests:
                    with contextlib.suppress(Exception):
                        self._io.delete(path)
                raise
            self._replaced, self._created = len(rewritten), len(added)
            self._kept, self._entries = len(kept), len(entries)
            return added + kept

        def _summary(self, snapshot_properties: dict[str, str] = EMPTY_DICT) -> Summary:
            # PyIceberg 0.12 sums up only appends, overwrites and deletes: the totals of
            # an overwrite without file changes, labelled as a replace
            parent = self._parent()
            totals = update_snapshot_summaries(
                summary=Summary(operation=Operation.OVERWRITE, **snapshot_properties),
                previous_summary=parent.summary if parent is not None else None,
            )
            return Summary(
                operation=Operation.REPLACE,
                **totals.additional_properties,
                **{
                    "manifests-created": str(self._created),
                    "manifests-kept": str(self._kept),
                    "manifests-replaced": str(self._replaced),
                    "entries-processed": str(self._entries),
                },
            )

    return _RewriteManifests(
        operation=Operation.REPLACE,
        transaction=tx,
        io=tx._table.io,
        snapshot_properties=snapshot_properties,
    )
