"""Iceberg table maintenance, run by SQL `CALL <catalog>.system.<procedure>(...)`."""

from __future__ import annotations

import contextlib
import functools
import math
import re
import struct
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Collection

    import pyiceberg.table
    from pyiceberg.expressions import BooleanExpression
    from pyiceberg.manifest import ManifestEntry, ManifestFile
    from pyiceberg.table import FileScanTask, Transaction
    from pyiceberg.table.metadata import TableMetadata
    from pyiceberg.typedef import Record

    from polars._typing import StorageOptionsDict
    from polars.io.iceberg._sink import _IcebergSinkedFile

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
    return _values_key(_partition_values(partition))


def _values_key(values: list[Any]) -> tuple[Any, ...]:
    # NaN after numbers, nulls last; a field's other values have one type
    return tuple(
        (v is None, _is_nan(v), None if v is None or _is_nan(v) else v) for v in values
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


# Iceberg's default of `write.target-file-size-bytes`.
_DATA_TARGET_SIZE_BYTES = 512 * 1024 * 1024

# The options of `rewrite_data_files` (binpack), as Iceberg's.
_REWRITE_DATA_FILES_OPTIONS = (
    "delete-file-threshold",
    "delete-ratio-threshold",
    "max-file-group-input-files",
    "max-file-group-size-bytes",
    "max-file-size-bytes",
    "min-file-size-bytes",
    "min-input-files",
    "rewrite-all",
    "target-file-size-bytes",
)


@dataclass(frozen=True)
class _BinPack:
    """Iceberg's file selection and sizing of `rewrite_data_files` (binpack)."""

    target: int
    min_size: int
    max_size: int
    min_input_files: int
    rewrite_all: bool
    max_group_size: int
    max_group_files: int
    delete_file_threshold: int
    delete_ratio_threshold: float

    @staticmethod
    def of(properties: dict[str, str], options: dict[str, str]) -> _BinPack:
        if unknown := sorted(set(options) - set(_REWRITE_DATA_FILES_OPTIONS)):
            msg = (
                f"option '{unknown[0]}' of rewrite_data_files is not supported; "
                f"supported: {', '.join(_REWRITE_DATA_FILES_OPTIONS)}"
            )
            raise NotImplementedError(msg)
        target = _int_option(options, "target-file-size-bytes", None)
        if target is None:
            target = int(
                properties.get("write.target-file-size-bytes", _DATA_TARGET_SIZE_BYTES)
            )
        binpack = _BinPack(
            target=target,
            min_size=_int_option(options, "min-file-size-bytes", int(target * 0.75)),
            max_size=_int_option(options, "max-file-size-bytes", int(target * 1.8)),
            min_input_files=_int_option(options, "min-input-files", 5),
            rewrite_all=_bool_option(options, "rewrite-all"),
            max_group_size=_int_option(
                options, "max-file-group-size-bytes", 100 * 1024**3
            ),
            max_group_files=_int_option(
                options, "max-file-group-input-files", 2**63 - 1
            ),
            delete_file_threshold=_int_option(
                options, "delete-file-threshold", 2**31 - 1
            ),
            delete_ratio_threshold=_float_option(
                options, "delete-ratio-threshold", 0.3
            ),
        )
        binpack._check()
        return binpack

    def _check(self) -> None:
        # as Iceberg Java
        target, min_size, max_size = self.target, self.min_size, self.max_size
        for ok, rule in (
            (target > 0, f"target-file-size-bytes ({target}) must be > 0"),
            (min_size >= 0, f"min-file-size-bytes ({min_size}) must be >= 0"),
            (
                target > min_size,
                f"target-file-size-bytes ({target}) must be > min-file-size-bytes "
                f"({min_size})",
            ),
            (
                target < max_size,
                f"target-file-size-bytes ({target}) must be < max-file-size-bytes "
                f"({max_size})",
            ),
            (
                self.min_input_files > 0,
                f"min-input-files ({self.min_input_files}) must be > 0",
            ),
            (
                self.max_group_size > 0,
                f"max-file-group-size-bytes ({self.max_group_size}) must be > 0",
            ),
            (
                self.max_group_files > 0,
                f"max-file-group-input-files ({self.max_group_files}) must be > 0",
            ),
            (
                self.delete_file_threshold >= 0,
                f"delete-file-threshold ({self.delete_file_threshold}) must be >= 0",
            ),
            (
                0 < self.delete_ratio_threshold <= 1,
                f"delete-ratio-threshold ({self.delete_ratio_threshold}) must be > 0 "
                "and <= 1",
            ),
        ):
            if not ok:
                msg = f"rewrite_data_files: {rule}"
                raise ValueError(msg)

    def is_candidate(self, task: FileScanTask) -> bool:
        size = task.file.file_size_in_bytes
        return (
            size < self.min_size or size > self.max_size or self.has_delete_issues(task)
        )

    def has_delete_issues(self, task: FileScanTask) -> bool:
        from polars.io.iceberg._dataset import _referenced_data_file

        deletes = task.delete_files
        if len(deletes) >= self.delete_file_threshold:
            return True
        rows = task.file.record_count
        if not deletes or rows <= 0:
            return False
        # as Iceberg Java: the rows of delete files of this data file only
        deleted = sum(d.record_count for d in deletes if _referenced_data_file(d))
        return min(deleted, rows) / rows >= self.delete_ratio_threshold

    def is_rewritten(self, group: list[FileScanTask]) -> bool:
        size = sum(t.file.file_size_in_bytes for t in group)
        n = len(group)
        return (
            (n > 1 and n >= self.min_input_files)
            or (n > 1 and size > self.target)
            or size > self.max_size
            or any(self.has_delete_issues(t) for t in group)
        )

    def split_size(self, size: int) -> int:
        """The input bytes per written file, as Iceberg Java's `inputSplitSize`.

        Files of that size, and a smaller remainder.
        """
        write_max = int(self.target + (self.max_size - self.target) * 0.5)
        split = size // self._output_files(size, write_max) + 5 * 1024
        return self.target if split < self.target else min(split, write_max)

    def _output_files(self, size: int, write_max: int) -> int:
        if size < self.target:
            return 1
        without_remainder = size // self.target
        if size % self.target > self.min_size:
            return without_remainder + 1
        if size // without_remainder < min(1.1 * self.target, write_max):
            return without_remainder
        return without_remainder + 1


def _int_option(options: dict[str, str], key: str, default: Any) -> Any:
    value = options.get(key)
    if value is None:
        return default
    if not re.fullmatch(r"[+-]?[0-9]+", value):
        msg = f"rewrite_data_files option '{key}' must be an integer, found {value!r}"
        raise ValueError(msg)
    return int(value)


def _float_option(options: dict[str, str], key: str, default: float) -> float:
    value = options.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        msg = f"rewrite_data_files option '{key}' must be a number, found {value!r}"
        raise ValueError(msg) from None


def _bool_option(options: dict[str, str], key: str) -> bool:
    value = options.get(key)
    if value is None:
        return False
    if value.lower() not in ("true", "false"):
        msg = f"rewrite_data_files option '{key}' must be 'true' or 'false', found {value!r}"
        raise ValueError(msg)
    return value.lower() == "true"


def _plan_rewrite(
    metadata: TableMetadata, tasks: list[FileScanTask], binpack: _BinPack
) -> list[list[FileScanTask]]:
    """The groups of data files to rewrite together, by Iceberg's binpack rules."""
    spec = metadata.spec()
    # as Iceberg Java: other specs' files go with the current spec's null partition
    other_specs = _values_key([None] * len(spec.fields))
    partitions: dict[tuple[Any, ...], list[FileScanTask]] = {}
    for task in tasks:
        key = (
            _partition_key(task.file.partition)
            if task.file.spec_id == spec.spec_id
            else other_specs
        )
        partitions.setdefault(key, []).append(task)
    groups = []
    for files in partitions.values():
        if not binpack.rewrite_all:
            files = [t for t in files if binpack.is_candidate(t)]
        packed = _pack(files, binpack.max_group_size, binpack.max_group_files)
        groups += [g for g in packed if binpack.rewrite_all or binpack.is_rewritten(g)]
    return groups


def _pack(
    tasks: list[FileScanTask], max_size: int, max_files: int
) -> list[list[FileScanTask]]:
    """`tasks` in order, in groups of at most `max_size` bytes and `max_files` files.

    A file larger than `max_size` is a group of its own, as Iceberg Java's packing.
    """
    groups: list[list[FileScanTask]] = []
    size = 0
    for task in tasks:
        n = task.file.file_size_in_bytes
        if not groups or size + n > max_size or len(groups[-1]) >= max_files:
            groups.append([])
            size = 0
        groups[-1].append(task)
        size += n
    return groups


def _unrewritable(
    table: pyiceberg.table.Table, selected: list[FileScanTask]
) -> str | None:
    """Why the files of `selected` cannot be rewritten, if they cannot."""
    from pyiceberg.manifest import DataFileContent, FileFormat

    from polars.io.iceberg._dataset import _referenced_data_file

    schema = table.schema()
    specs = table.specs()
    for spec_id in sorted({t.file.spec_id for t in selected}):
        for field in specs[spec_id].fields:
            # PyIceberg writes manifests with the current schema
            if _source_type(schema, field.source_id) is None:
                return (
                    f"the source column of partition field '{field.name}' of spec "
                    f"{spec_id} was dropped"
                )
    for task in selected:
        if task.file.file_format != FileFormat.PARQUET:
            return f"data file '{task.file.file_path}' is not Parquet"
        # PyIceberg cannot match NaN partition values to remove the file
        if any(map(_is_nan, _partition_values(task.file.partition))):
            return f"data file '{task.file.file_path}' has a NaN partition value"
        for delete in task.delete_files:
            # what the native reader reads
            if (
                delete.content != DataFileContent.POSITION_DELETES
                or delete.file_format != FileFormat.PARQUET
                or _referenced_data_file(delete) is None
            ):
                return (
                    f"delete file '{delete.file_path}' is not a Parquet position "
                    "delete file of one data file"
                )
    return None


def _has_equality_deletes(table: pyiceberg.table.Table) -> bool:
    from pyiceberg.manifest import DataFileContent, ManifestContent

    snapshot = table.current_snapshot()
    if snapshot is None:
        return False
    return any(
        entry.data_file.content == DataFileContent.EQUALITY_DELETES
        for manifest in snapshot.manifests(table.io)
        if manifest.content == ManifestContent.DELETES
        for entry in manifest.fetch_manifest_entry(table.io, discard_deleted=True)
    )


def rewrite_data_files(
    table: pyiceberg.table.Table,
    *,
    options: dict[str, str],
    row_filter: BooleanExpression | None,
    storage_options: StorageOptionsDict | None,
) -> tuple[int, int, int]:
    """
    Compact the data files of `table` in one `replace` commit, as Iceberg's binpack.

    Data files (of those that may match `row_filter`) are selected and grouped by
    partition by Iceberg's rules and `options`. Each group's live rows are written
    again, under the current partition spec and sort order, into files of about the
    target size. Delete files stay. Concurrent appends do not conflict. Nothing is
    committed if no group needs a rewrite. Returns the numbers of rewritten and added
    data files and the rewritten bytes.
    """
    from pyiceberg.exceptions import CommitFailedException, CommitStateUnknownException
    from pyiceberg.expressions import AlwaysFalse
    from pyiceberg.table import Table

    from polars.io.iceberg._credentials import refresh_file_io
    from polars.io.iceberg._delete import _candidate_tasks
    from polars.io.iceberg._sink import (
        _SINK_UUID_PROPERTY,
        IcebergSinkState,
        _AlreadyCommitted,
        _assert_replace_base,
        _data_files_for_commit,
        _remove_files,
        _replace_transaction,
        _ReplaceFiles,
        _SinkCatalog,
    )

    name = ".".join(table.name())
    if table.format_version > 2:
        msg = (
            f"rewrite_data_files of an Iceberg v{table.format_version} table is not "
            f"supported yet: table '{name}'"
        )
        raise NotImplementedError(msg)
    binpack = _BinPack.of(table.metadata.properties, options)
    if table.current_snapshot() is None:
        return 0, 0, 0
    if _has_equality_deletes(table):
        msg = (
            f"rewrite_data_files of Iceberg table '{name}' is not supported: it has "
            "equality delete files"
        )
        raise NotImplementedError(msg)
    tasks = (
        list(table.scan().plan_files())
        if row_filter is None
        else _candidate_tasks(table, row_filter)
    )
    groups = _plan_rewrite(table.metadata, tasks, binpack)
    if not groups:
        return 0, 0, 0
    selected = [t for group in groups for t in group]
    if reason := _unrewritable(table, selected):
        msg = f"rewrite_data_files of Iceberg table '{name}' is not supported: {reason}"
        raise NotImplementedError(msg)
    # the commit rewrites the manifests of the selected files
    if rounded := _rounded_doubles_error(
        table.metadata,
        name,
        "rewrite_data_files",
        {t.file.spec_id for t in selected} | {table.metadata.default_spec_id},
    ):
        raise NotImplementedError(rounded)

    removed = [t.file for t in selected]
    counts = (len(removed), 0, sum(f.file_size_in_bytes for f in removed))
    replace = _ReplaceFiles(
        data_files=removed,
        conflict_filter=AlwaysFalse(),
        schema_id=table.metadata.current_schema_id,
        spec_id=table.metadata.default_spec_id,
    )
    # one directory and uuid for the files of every group and the commit
    state = IcebergSinkState.new(table, storage_options=storage_options)
    sink_uuid = state.sink_uuid_str
    catalog = _SinkCatalog(table.catalog, sink_uuid, verbose=False)
    # the commit's base: the snapshot planned on, also if `table` is refreshed
    wrapped = Table(
        table.name(),
        table.metadata,
        table.metadata_location,
        table.io,
        catalog,  # type: ignore[arg-type]
        table.config,
    )
    written: list[tuple[list[_IcebergSinkedFile], int | None]] = []
    producers: list[Any] = []
    landed = False
    try:
        # one by one: a failed group keeps the earlier ones' files for the cleanup
        written.extend(
            _write_group(table, group, binpack, storage_options, sink_uuid)
            for group in groups
        )
        # vended credentials may have expired while the groups were written
        refresh_file_io(table)
        wrapped.io = table.io
        with _replace_transaction(wrapped, sink_uuid, replace) as tx:
            _assert_replace_base(tx, replace)
            added = [
                data_file
                for files, sort_order_id in written
                for data_file in _data_files_for_commit(tx, files, sort_order_id)
            ]
            counts = (counts[0], len(added), counts[2])
            producer = tx.update_snapshot(
                snapshot_properties={_SINK_UUID_PROPERTY: sink_uuid}
            ).overwrite()
            producers.append(producer)
            _as_compaction(producer)
            for data_file in removed:
                producer.delete_data_file(data_file)
            for data_file in added:
                producer.append_data_file(data_file)
            producer.commit()
        landed = True
    except _AlreadyCommitted:
        landed = True
    except CommitStateUnknownException as error:
        location, _ = state.written_files_dir()
        error.add_note(
            f"the compaction may have landed; data files kept under '{location}'"
        )
        raise
    except Exception as error:
        if catalog._snapshot_ids and error is not catalog._raised:
            # failed after a commit request, not by it
            landed = _landed(table, sink_uuid)
            if not landed and not isinstance(catalog._raised, CommitFailedException):
                msg = "a failed Iceberg commit may still land"
                raise CommitStateUnknownException(msg) from error
        if not landed:
            # PyIceberg removes the manifests of some refusals only; twice is a no-op
            for producer in producers:
                producer._clean_all_uncommitted()
            # by path: the object-storage layout shares directories with other writers
            _remove_files(
                state.storage_file_io(),
                [file[0] for files, _ in written for file in files],
            )
            raise
    return counts


def _write_group(
    table: pyiceberg.table.Table,
    group: list[FileScanTask],
    binpack: _BinPack,
    storage_options: StorageOptionsDict | None,
    sink_uuid: str,
) -> tuple[list[_IcebergSinkedFile], int | None]:
    """Write the live rows of the files of `group` in files of about the target size.

    Uncommitted. Returns the written files and their sort order.
    """
    from polars.io.iceberg._sink import IcebergSinkState, _HandBack, _remove_files
    from polars.io.iceberg.functions import _scan_iceberg_files

    size = sum(t.file.file_size_in_bytes for t in group)
    rows = sum(t.file.record_count for t in group)
    # as Iceberg Java, by the input's bytes: output that compresses better than the
    # input (eg: another writer's) is smaller, and compacted again by a next CALL
    split_rows = rows * binpack.split_size(size) // max(size, 1)
    # the sink's limit
    max_rows = min(max(1, split_rows), 2**32 - 1)
    hand_back = _HandBack(max_rows_per_file=max_rows)
    state = IcebergSinkState.new(
        table, storage_options=storage_options, replace_files=hand_back
    )
    state.sink_uuid_str = sink_uuid
    try:
        state.attach_sink(_scan_iceberg_files(table, group, storage_options)).collect()
    except Exception as error:
        # its files are in its own directory, which the caller does not know
        if (remaining := state.remove_written_files()) is not None:
            error.add_note(f"data files may remain under '{remaining}'")
        raise
    assert hand_back.files is not None
    # a group whose rows were all deleted leaves no file
    if empty := [path for path, rows, *_ in hand_back.files if rows == 0]:
        _remove_files(state.storage_file_io(), empty)
    return [f for f in hand_back.files if f[1] > 0], hand_back.sort_order_id


def _as_compaction(producer: Any) -> None:
    """Have the overwrite `producer` commit a compaction, as Iceberg Java.

    The snapshot is a `replace`. A retry is refused if a rewritten file is gone or has
    new deletes, not for concurrent appends.
    """
    from pyiceberg.table.snapshots import Operation, Summary

    # private in PyIceberg 0.12, as its own producers' validation
    from pyiceberg.table.update.validate import (
        _validate_data_files_exist,
        _validate_no_new_deletes_for_data_files,
    )
    from pyiceberg.typedef import EMPTY_DICT

    if not hasattr(producer, "_commit_window"):
        msg = "this PyIceberg version does not support compaction commits"
        raise NotImplementedError(msg)

    summary = producer._summary

    def _summary(snapshot_properties: dict[str, str] = EMPTY_DICT) -> Summary:
        # PyIceberg 0.12 sums up only appends, overwrites and deletes: the totals of an
        # overwrite, labelled as a replace
        producer._operation = Operation.OVERWRITE
        totals = summary(snapshot_properties)
        return Summary(operation=Operation.REPLACE, **totals.additional_properties)

    def _validate_concurrency() -> None:
        window = producer._commit_window
        if window is None or window.is_empty() or window.head is None:
            return
        table = producer._transaction._table
        files = producer._deleted_data_files
        _validate_data_files_exist(table, window.head, files, window.base)
        _validate_no_new_deletes_for_data_files(
            table, window.head, None, files, window.base
        )

    # private in PyIceberg 0.12
    producer._summary = _summary
    producer._validate_concurrency = _validate_concurrency


# The table property of the maximum age of non-main refs, as Iceberg Java.
_MAX_REF_AGE_MS = "history.expire.max-ref-age-ms"


@dataclass(frozen=True)
class ExpiredFiles:
    """The files an `expire_snapshots` deleted, as the procedure's result columns."""

    deleted_data_files_count: int = 0
    deleted_position_delete_files_count: int = 0
    deleted_equality_delete_files_count: int = 0
    deleted_manifest_files_count: int = 0
    deleted_manifest_lists_count: int = 0
    deleted_statistics_files_count: int = 0


@dataclass(frozen=True)
class _Expiry:
    """The refs and snapshots an `expire_snapshots` removes."""

    refs: list[str]
    snapshot_ids: list[int]
    # Snapshots whose partition statistics go with them.
    partition_statistics: list[int]


def expire_snapshots(
    table: pyiceberg.table.Table,
    *,
    older_than_ms: int | None,
    retain_last: int | None,
    snapshot_ids: list[int],
    name: str,
) -> ExpiredFiles:
    """
    Expire snapshots of `table` as Iceberg's `expire_snapshots`, in one commit.

    Each branch keeps its ancestors newer than its maximum snapshot age (default:
    `older_than`) or among its newest minimum number (default: `retain_last`). Refs
    older than their maximum ref age go, never `main`. Snapshots no ref reaches are
    kept while newer than `older_than`. `snapshot_ids` are expired too. Then the files
    only expired snapshots referenced are deleted, and counted.
    """
    from pyiceberg.exceptions import CommitFailedException, CommitStateUnknownException
    from pyiceberg.table import TableProperties
    from pyiceberg.utils.properties import property_as_int

    from polars.exceptions import InvalidOperationError

    properties = table.metadata.properties
    _refuse_without_gc(table.metadata, name)
    _refuse_refs_loading(table, name)
    for snapshot_id in snapshot_ids:
        if table.metadata.snapshot_by_id(snapshot_id) is None:
            msg = f"snapshot {snapshot_id} of table '{name}' not found"
            raise InvalidOperationError(msg)

    retries = max(
        0,
        property_as_int(
            properties,
            TableProperties.COMMIT_NUM_RETRIES,
            TableProperties.COMMIT_NUM_RETRIES_DEFAULT,
        )
        or 0,
    )
    before = table.metadata
    for attempt in range(retries + 1):
        before = table.metadata
        # Another writer may have disabled it meanwhile.
        _refuse_without_gc(before, name)
        expiry = _expiry(
            before,
            _now_ms(),
            older_than_ms,
            retain_last,
            # Explicit ids another writer expired meanwhile are gone already.
            [i for i in snapshot_ids if before.snapshot_by_id(i) is not None],
            name,
        )
        if not expiry.refs and not expiry.snapshot_ids:
            return ExpiredFiles()
        try:
            _commit_expiry(table, before, expiry)
            break
        except CommitFailedException:
            # A ref moved: plan again on the table as it is now.
            if attempt == retries:
                raise
            table.refresh()
        except CommitStateUnknownException:
            # Landed only if the planned snapshots and refs are gone now.
            current_table = table.catalog.load_table(table.name())
            current = current_table.metadata
            if current.table_uuid != before.table_uuid or _expiry_overtaken(
                current, expiry, all_gone=False
            ):
                raise
            table.metadata = current
            table.metadata_location = current_table.metadata_location
            break
        except Exception:
            # Failed in another way: a catalog that applies the updates here, or that
            # rejects removals another expiry made meanwhile, or a commit that landed
            # and whose response could not be read.
            current_table = table.catalog.load_table(table.name())
            current = current_table.metadata
            if current.table_uuid != before.table_uuid:
                raise
            if not _expiry_overtaken(current, expiry, all_gone=False):
                # Every planned removal is done: clean up the files.
                table.metadata = current
                table.metadata_location = current_table.metadata_location
                break
            if attempt == retries or not _expiry_overtaken(current, expiry):
                raise
            table.refresh()
    return _delete_expired_files(table, before, expiry.snapshot_ids, name)


def _refuse_without_gc(
    metadata: TableMetadata,
    name: str,
    refused: str = "snapshots of table '{}' cannot be expired",
) -> None:
    from pyiceberg.utils.properties import property_as_bool

    from polars.exceptions import InvalidOperationError

    if not property_as_bool(metadata.properties, "gc.enabled", True):
        msg = (
            f"{refused.format(name)}: its property gc.enabled is false (its files may "
            "belong to other tables)"
        )
        raise InvalidOperationError(msg)


def _refuse_refs_loading(
    table: pyiceberg.table.Table,
    name: str,
    refused: str = "snapshots of table '{}' cannot be expired",
) -> None:
    from polars.exceptions import InvalidOperationError

    mode = table.catalog.properties.get("snapshot-loading-mode", "all")
    if str(mode).lower() == "refs":
        msg = (
            f"{refused.format(name)}: the catalog sends only the snapshots of branches "
            "and tags; attach it with the property `snapshot-loading-mode` = `all`"
        )
        raise InvalidOperationError(msg)


def _now_ms() -> int:
    import time

    return int(time.time() * 1000)


def _expiry_overtaken(
    current: TableMetadata, expiry: _Expiry, *, all_gone: bool = True
) -> bool:
    """
    Whether planned removals are already gone from `current`.

    With `all_gone`: some are (another writer removed them). Without: not all are (the
    commit did not land).
    """
    partition_statistics = {s.snapshot_id for s in current.partition_statistics}
    present = (
        [current.snapshot_by_id(i) is not None for i in expiry.snapshot_ids]
        + [r in current.refs for r in expiry.refs]
        + [i in partition_statistics for i in expiry.partition_statistics]
    )
    return not all(present) if all_gone else any(present)


def _expiry(
    metadata: TableMetadata,
    now_ms: int,
    older_than_ms: int | None,
    retain_last: int | None,
    snapshot_ids: list[int],
    name: str,
) -> _Expiry:
    """The refs and snapshots to remove, by Iceberg Java's rules (`RemoveSnapshots`)."""
    from pyiceberg.table import TableProperties
    from pyiceberg.table.refs import MAIN_BRANCH, SnapshotRefType
    from pyiceberg.table.snapshots import ancestors_of
    from pyiceberg.utils.properties import property_as_int

    from polars.exceptions import InvalidOperationError

    properties = metadata.properties
    if older_than_ms is None:
        max_age = property_as_int(
            properties,
            TableProperties.MAX_SNAPSHOT_AGE_MS,
            TableProperties.MAX_SNAPSHOT_AGE_MS_DEFAULT,
        )
        older_than_ms = now_ms - max_age  # type: ignore[operator]
    if retain_last is None:
        retain_last = property_as_int(
            properties,
            TableProperties.MIN_SNAPSHOTS_TO_KEEP,
            TableProperties.MIN_SNAPSHOTS_TO_KEEP_DEFAULT,
        )
    max_ref_age = property_as_int(properties, _MAX_REF_AGE_MS, None)

    kept_refs = {}
    removed_refs = []
    for ref_name, ref in metadata.refs.items():
        snapshot = metadata.snapshot_by_id(ref.snapshot_id)
        ref_max_age = (
            ref.max_ref_age_ms if ref.max_ref_age_ms is not None else max_ref_age
        )
        # A ref's age is its snapshot's.
        if ref_name == MAIN_BRANCH or (
            snapshot is not None
            and (ref_max_age is None or now_ms - snapshot.timestamp_ms <= ref_max_age)
        ):
            kept_refs[ref_name] = ref
        else:
            removed_refs.append(ref_name)

    keep = {ref.snapshot_id for ref in kept_refs.values()}
    for snapshot_id in snapshot_ids:
        if snapshot_id in keep:
            refs = sorted(
                n for n, r in kept_refs.items() if r.snapshot_id == snapshot_id
            )
            msg = (
                f"snapshot {snapshot_id} of table '{name}' cannot be expired: it is the "
                f"snapshot of {', '.join(repr(r) for r in refs)}"
            )
            raise InvalidOperationError(msg)

    reached = set()
    for ref in kept_refs.values():
        head = metadata.snapshot_by_id(ref.snapshot_id)
        if ref.snapshot_ref_type != SnapshotRefType.BRANCH or head is None:
            reached.add(ref.snapshot_id)
            continue
        branch_older_than = (
            now_ms - ref.max_snapshot_age_ms
            if ref.max_snapshot_age_ms is not None
            else older_than_ms
        )
        branch_keep = (
            ref.min_snapshots_to_keep
            if ref.min_snapshots_to_keep is not None
            else retain_last
        )
        kept = 0
        keeping = True
        for ancestor in ancestors_of(head, metadata):
            reached.add(ancestor.snapshot_id)
            # The newest ancestors, until one is old enough and enough are kept.
            keeping = keeping and (
                kept < branch_keep or ancestor.timestamp_ms >= branch_older_than  # type: ignore[operator]
            )
            if keeping:
                keep.add(ancestor.snapshot_id)
                kept += 1
    for snapshot in metadata.snapshots:
        if (
            snapshot.snapshot_id not in reached
            and snapshot.timestamp_ms >= older_than_ms
        ):
            keep.add(snapshot.snapshot_id)

    expired = {s.snapshot_id for s in metadata.snapshots if s.snapshot_id not in keep}
    # Named snapshots go even if a branch's retention keeps them.
    removed = expired | set(snapshot_ids)
    return _Expiry(
        removed_refs,
        sorted(removed),
        # Not every catalog drops partition statistics of removed snapshots.
        sorted(
            s.snapshot_id
            for s in metadata.partition_statistics
            if s.snapshot_id in removed
        ),
    )


def _commit_expiry(
    table: pyiceberg.table.Table, metadata: TableMetadata, expiry: _Expiry
) -> None:
    """Commit `expiry`, planned on `metadata`, failing if a ref moved since."""
    from pyiceberg.table.refs import MAIN_BRANCH
    from pyiceberg.table.update import (
        AssertRefSnapshotId,
        RemovePartitionStatisticsUpdate,
        RemoveSnapshotRefUpdate,
        RemoveSnapshotsUpdate,
    )

    updates: list[Any] = [RemoveSnapshotRefUpdate(ref_name=r) for r in expiry.refs]  # type: ignore[call-arg]
    updates += [
        RemovePartitionStatisticsUpdate(snapshot_id=i)  # type: ignore[call-arg]
        for i in expiry.partition_statistics
    ]
    if expiry.snapshot_ids:
        updates.append(RemoveSnapshotsUpdate(snapshot_ids=expiry.snapshot_ids))  # type: ignore[call-arg]
    # Every ref, in one `_stage`: PyIceberg keeps one requirement per kind across calls.
    requirements = [
        AssertRefSnapshotId(ref=n, snapshot_id=r.snapshot_id)  # type: ignore[call-arg]
        for n, r in metadata.refs.items()
    ]
    if MAIN_BRANCH not in metadata.refs:
        requirements.append(AssertRefSnapshotId(ref=MAIN_BRANCH, snapshot_id=None))  # type: ignore[call-arg]
    tx = table.transaction()
    tx._stage(tuple(updates), tuple(requirements))
    tx.commit_transaction()


def _delete_expired_files(
    table: pyiceberg.table.Table,
    before: TableMetadata,
    expired_ids: list[int],
    name: str,
) -> ExpiredFiles:
    """
    Delete the files only snapshots of `expired_ids` referenced, best effort.

    A path goes if a removed snapshot referenced it (live in one of its manifests, its
    manifests, its manifest list, its statistics) and no snapshot of the table as loaded
    now uses it, in any role, under any name it may have (`_path_names`). Only the
    manifests of removed snapshots that no kept
    snapshot lists are read in full; kept manifests are read only while candidate files
    remain. If a kept snapshot's manifests cannot be read, nothing is deleted.
    """
    from pyiceberg.manifest import DataFileContent
    from pyiceberg.utils.concurrent import ExecutorFactory
    from pyiceberg.utils.properties import property_as_bool

    from polars._warnings import issue_warning

    def keep_files(reason: str) -> ExpiredFiles:
        issue_warning(
            f"expired snapshots of table '{name}', but deleted no files: {reason}",
            UserWarning,
        )
        return ExpiredFiles()

    # The table as it is now: a commit of another writer may reference the files again.
    try:
        current = table.catalog.load_table(table.name()).metadata
    except Exception as error:
        return keep_files(f"the table could not be loaded again ({error})")
    if current.table_uuid != before.table_uuid:
        return keep_files("another table has its name now")
    if not property_as_bool(current.properties, "gc.enabled", True):
        return keep_files("its property gc.enabled is false now")
    removed = [
        s
        for s in before.snapshots
        if s.snapshot_id in expired_ids
        and current.snapshot_by_id(s.snapshot_id) is None
    ]
    if not removed:
        return ExpiredFiles()
    io = table.io
    executor = ExecutorFactory.get_or_create()

    # Kept: the names of kept manifest lists, manifests and statistics.
    kept: set[str] = set()
    kept_manifests: dict[str, Any] = {}
    try:
        for snapshot in current.snapshots:
            for manifest in snapshot.manifests(io):
                kept_manifests.setdefault(manifest.manifest_path, manifest)
            kept |= _path_names(snapshot.manifest_list)
    except Exception as error:
        return keep_files(f"a kept snapshot's manifests could not be read ({error})")
    for path in [
        *kept_manifests,
        *(
            s.statistics_path
            for s in [*current.statistics, *current.partition_statistics]
        ),
    ]:
        kept |= _path_names(path)

    def unkept(path: str) -> bool:
        return _path_names(path).isdisjoint(kept)

    # Candidates: what removed snapshots referenced, that no kept one lists.
    lists: dict[str, str] = {}
    manifests: dict[str, Any] = {}
    for snapshot in removed:
        try:
            for manifest in snapshot.manifests(io):
                if unkept(manifest.manifest_path):
                    manifests.setdefault(manifest.manifest_path, manifest)
        except Exception:
            # An unreadable removed snapshot's files are not candidates.
            continue
        if unkept(snapshot.manifest_list):
            lists[snapshot.manifest_list] = snapshot.manifest_list

    def live(manifest: Any) -> list[tuple[str, DataFileContent]]:
        return [
            (e.data_file.file_path, e.data_file.content)
            for e in manifest.fetch_manifest_entry(io, discard_deleted=True)
        ]

    files: dict[str, tuple[str, DataFileContent]] = {}
    for entries in executor.map(_or_none(live), list(manifests.values())):
        for path, content in entries or []:
            if unkept(path):
                files.setdefault(path, (path, content))
    stats = {
        s.statistics_path: s.statistics_path
        for s in [*before.statistics, *before.partition_statistics]
        if s.snapshot_id in expired_ids and unkept(s.statistics_path)
    }

    # Kept manifests, read until no candidate is left: a path can be live in one (one
    # a removed snapshot also listed, or one added since), in any role.
    candidates = {
        path: _path_names(path) for path in [*files, *stats, *manifests, *lists]
    }
    pending = list(kept_manifests.values())
    while candidates and pending:
        batch, pending = pending[:64], pending[64:]
        try:
            for entries in executor.map(live, batch):
                live_names = {n for path, _ in entries for n in _path_names(path)}
                candidates = {
                    path: names
                    for path, names in candidates.items()
                    if names.isdisjoint(live_names)
                }
        except Exception as error:
            return keep_files(f"a kept manifest could not be read ({error})")
    files = {k: v for k, v in files.items() if k in candidates}
    stats = {k: v for k, v in stats.items() if k in candidates and k not in files}
    manifests = {k: v for k, v in manifests.items() if k in candidates}
    lists = {k: v for k, v in lists.items() if k in candidates}

    failed = 0

    def delete(paths: list[str]) -> int:
        nonlocal failed

        def one(path: str) -> bool:
            # FileIOs drop a query or fragment, and fsspec expands globs: both widen
            # the delete.
            if any(c in path for c in "?#*["):
                return False
            try:
                io.delete(path)
            except Exception:
                return False
            return True

        done = sum(executor.map(one, paths))
        failed += len(paths) - done
        return done

    def of(kind: DataFileContent) -> list[str]:
        return sorted(path for path, content in files.values() if content == kind)

    deleted = ExpiredFiles(
        deleted_data_files_count=delete(of(DataFileContent.DATA)),
        deleted_position_delete_files_count=delete(
            of(DataFileContent.POSITION_DELETES)
        ),
        deleted_equality_delete_files_count=delete(
            of(DataFileContent.EQUALITY_DELETES)
        ),
        # Metadata last: after a failure above, the metadata left still lists the files.
        deleted_manifest_files_count=delete(
            sorted(m.manifest_path for m in manifests.values())
        ),
        deleted_manifest_lists_count=delete(sorted(lists.values())),
        deleted_statistics_files_count=delete(sorted(stats.values())),
    )
    if failed:
        issue_warning(
            f"expired snapshots of table '{name}', but {failed} of their files could "
            "not be deleted",
            UserWarning,
        )
    return deleted


def _or_none(function: Any) -> Any:
    """`function`, returning None where it raises."""

    def call(*args: Any) -> Any:
        try:
            return function(*args)
        except Exception:
            return None

    return call


def _path_names(path: str) -> frozenset[str]:
    """
    Every name the file at `path` may have, as PyArrow's or fsspec's FileIO resolves it.

    `s3a://b/k` is `s3://b/k`, with or without a `?query` or `#fragment` (PyArrow drops
    them, fsspec keeps them), also as `s3:///b//./k`. Azure by container, whatever the
    account host (`c@account.host/k`, `account.host/c/k`, `c/k`). HDFS by path, whatever
    the name node. `file:///a/./b`, `file:/a/b`
    and `/a/b` are one local file. Percent-encoded or not. A file is kept when any of
    its names is a kept file's: more names only keep more files.
    """
    import posixpath
    import unicodedata
    from pathlib import Path
    from urllib.parse import unquote, urlparse

    from polars.io.iceberg._plugin import _AZURE_SCHEMES, _backend

    names = {path}
    parsed = urlparse(path)
    scheme = parsed.scheme.lower()
    if len(scheme) == 1:
        # A Windows drive.
        locals_ = [path]
    elif not scheme or scheme == "file":
        # Literal (fsspec), and parsed: without query or fragment (PyArrow).
        literal = re.sub(r"^file:(//(localhost)?)?(?=/)", "", path, flags=re.IGNORECASE)
        locals_ = [literal, f"{parsed.netloc}{parsed.path}"]
    else:
        schemes = _backend(path)
        canonical = schemes[0] if schemes else scheme
        for name in (
            f"{canonical}:{path.split(':', 1)[1]}",
            f"{canonical}://{parsed.netloc}{parsed.path}",
        ):
            names |= {name, unquote(name)}
        # Loosest: the bucket or container (no account, host, port), the key with
        # slashes collapsed and `.`/`..` resolved; HDFS by path alone.
        netloc = parsed.netloc.lower()
        if scheme in ("hdfs", "viewfs"):
            netloc = ""
        elif "@" in netloc:
            # Azure `container@account.host`.
            netloc = netloc.split("@", 1)[0]
        elif schemes == _AZURE_SCHEMES and "." in netloc:
            # Azure `account.host/container`.
            netloc = ""
        for key in {parsed.path, unquote(parsed.path)}:
            loose = posixpath.normpath(f"/{netloc}/{key}").lstrip("/")
            names.add(f"{canonical}:{loose}")
        return frozenset(names)
    locals_ += [str(Path(p).expanduser()) for p in locals_ if p.startswith("~")]
    local_names = set()
    for local in [*locals_, *map(unquote, locals_)]:
        if re.match(r"^/[A-Za-z]:", local):
            local = local[1:]
        local_names.add(local)
        with contextlib.suppress(OSError, ValueError):
            local_names.add(str(Path(local).resolve()))
    # A case-insensitive disk opens `A.parquet` as `a.parquet`, and some disks open a
    # name in any Unicode normalization.
    local_names |= {unicodedata.normalize("NFC", n) for n in local_names}
    return frozenset(names | local_names | {n.casefold() for n in local_names})


# Java's `remove_orphan_files` defaults and floor.
_ORPHAN_AGE_DEFAULT_MS = 3 * 24 * 60 * 60 * 1000
_ORPHAN_AGE_FLOOR_MS = 24 * 60 * 60 * 1000
_ORPHANS_REFUSED = "orphan files of table '{}' cannot be removed"


@dataclass(frozen=True)
class _Listed:
    """A file the listing found."""

    # The URI, the table location's scheme and authority with the listed key.
    uri: str
    # The path the listing's filesystem names it by, for an exact delete.
    fs_path: str
    # Modification time in milliseconds (UTC), None when unknown.
    mtime_ms: int | None


def remove_orphan_files(
    table: pyiceberg.table.Table,
    *,
    older_than_ms: int | None,
    location: str | None,
    dry_run: bool,
    equal_schemes: dict[str, str],
    equal_authorities: dict[str, str],
    prefix_mismatch_mode: str,
    name: str,
) -> list[str]:
    """
    Delete the files under the table location that the table does not reference.

    As Iceberg's `remove_orphan_files`, plus: a directory without referenced files keeps
    its files while one of them is newer than `older_than` (a write still running).
    Returns the URIs of the orphans, deleted or, with `dry_run`, found.
    """
    from polars._warnings import issue_warning
    from polars.exceptions import InvalidOperationError

    _refuse_without_gc(table.metadata, name, _ORPHANS_REFUSED)
    _refuse_refs_loading(table, name, _ORPHANS_REFUSED)
    now = _now_ms()
    older = now - _ORPHAN_AGE_DEFAULT_MS if older_than_ms is None else older_than_ms
    if not dry_run and older > now - _ORPHAN_AGE_FLOOR_MS:
        msg = (
            f"{_ORPHANS_REFUSED.format(name)} with an `older_than` less than 24 hours "
            "ago: files of writes still running may not be referenced yet; use "
            "`dry_run => true` to list them"
        )
        raise InvalidOperationError(msg)
    table_location = table.location().rstrip("/")
    root = table_location if location is None else location.rstrip("/")
    _check_location(table.io, root, table_location, name)

    before = table.metadata
    # The files referenced before the listing must all be in it.
    referenced = _readable_references(table, name)
    referenced_location = table.metadata_location
    try:
        listed = _list_files(table.io, root)
    except NotImplementedError:
        raise
    except Exception as error:
        msg = f"{_ORPHANS_REFUSED.format(name)}: listing '{root}' failed ({error})"
        raise InvalidOperationError(msg) from error
    _check_listing(
        table.io,
        listed,
        referenced,
        table.metadata_location,
        root,
        table_location,
        name,
    )
    retries = 3
    for attempt in range(retries + 1):
        # The valid set also from metadata loaded after the listing: a file listed and
        # committed meanwhile is referenced.
        current_table = table.catalog.load_table(table.name())
        current = current_table.metadata
        if current.table_uuid != before.table_uuid:
            msg = f"{_ORPHANS_REFUSED.format(name)}: another table has its name now"
            raise InvalidOperationError(msg)
        _refuse_without_gc(current, name, _ORPHANS_REFUSED)
        if current_table.metadata_location != referenced_location:
            referenced = referenced | _readable_references(current_table, name)
            referenced_location = current_table.metadata_location
        orphans = _orphans(
            table.io,
            listed,
            referenced,
            current,
            older,
            root,
            equal_schemes,
            equal_authorities,
            prefix_mismatch_mode,
            name,
        )
        # Plan again if a commit landed meanwhile: it may reference a listed file.
        latest = table.catalog.load_table(table.name())
        if latest.metadata_location == current_table.metadata_location:
            break
        if attempt == retries:
            msg = (
                f"{_ORPHANS_REFUSED.format(name)}: the table changed during each of "
                f"{retries + 1} attempts"
            )
            raise InvalidOperationError(msg)

    orphans = _without_local_links(orphans, root)
    if dry_run:
        return sorted(f.uri for f in orphans)
    from pyiceberg.utils.concurrent import ExecutorFactory

    deleted = ExecutorFactory.get_or_create().map(
        lambda f: _delete_listed(table.io, root, f), orphans
    )
    failed = sum(not done for done in deleted)
    if failed:
        issue_warning(
            f"removed orphan files of table '{name}', but {failed} of them could not "
            "be deleted",
            UserWarning,
        )
    return sorted(f.uri for f in orphans)


def _list_files(io: Any, root: str) -> list[_Listed]:
    """Every file under `root`, through `io`'s own filesystem (PyArrow or fsspec)."""
    from pyiceberg.io.fsspec import FsspecFileIO
    from pyiceberg.io.pyarrow import PyArrowFileIO

    if isinstance(io, PyArrowFileIO):
        from pyarrow.fs import FileSelector, FileType

        scheme, netloc, _ = io.parse_location(root, io.properties)
        fs = io.fs_by_scheme(scheme, netloc)
        base = _fs_base(io, root)
        infos = fs.get_file_info(
            FileSelector(base, recursive=True, allow_not_found=True)
        )
        files = []
        for info in infos:
            if info.type != FileType.File or not info.path.startswith(f"{base}/"):
                continue
            mtime = info.mtime
            files.append(
                _Listed(
                    uri=f"{root}/{info.path[len(base) + 1 :]}",
                    fs_path=info.path,
                    mtime_ms=None if mtime is None else _datetime_ms(mtime),
                )
            )
        return files

    if isinstance(io, FsspecFileIO):
        fs = _fsspec_fs(io, root)
        base = fs._strip_protocol(root).rstrip("/")
        # A cached listing would hide files committed since (and after this one).
        fs.invalidate_cache()
        # A listing error must not shorten the listing silently: fsspec's own walk
        # omits errors below the top level, backends with their own `find` raise them.
        from fsspec.asyn import AsyncFileSystem
        from fsspec.spec import AbstractFileSystem

        walks = (
            type(fs)._find is AsyncFileSystem._find
            if isinstance(fs, AsyncFileSystem)
            else type(fs).find is AbstractFileSystem.find
        )
        try:
            found = (
                _fsspec_walk(fs, base)
                if walks
                else fs.find(base, withdirs=False, detail=True)
            )
        finally:
            fs.invalidate_cache()
        files = []
        for path, info in found.items():
            if info.get("type") != "file" or path.endswith("/"):
                continue
            if not path.startswith(f"{base}/"):
                continue
            files.append(
                _Listed(
                    uri=f"{root}/{path[len(base) + 1 :]}",
                    fs_path=path,
                    mtime_ms=_fsspec_mtime_ms(info),
                )
            )
        return files

    msg = f"listing files with {type(io).__name__} (PyArrowFileIO or FsspecFileIO)"
    raise NotImplementedError(msg)


def _fsspec_walk(fs: Any, base: str) -> dict[str, dict[str, Any]]:
    """The files below `base`, raising on any error; symbolic links are not followed."""
    found: dict[str, dict[str, Any]] = {}
    pending = [base]
    while pending:
        for info in fs.ls(pending.pop(), detail=True):
            if info.get("type") == "directory":
                if not info.get("islink"):
                    pending.append(info["name"].rstrip("/"))
            else:
                found[info["name"]] = info
    return found


def _fsspec_fs(io: Any, location: str) -> Any:
    """The fsspec filesystem of `location` (a one-letter scheme is a Windows drive)."""
    from urllib.parse import urlparse

    uri = urlparse(location)
    scheme = uri.scheme if len(uri.scheme) > 1 else "file"
    return io.get_fs(scheme, uri.hostname if len(uri.scheme) > 1 else None)


def _exists(io: Any, path: str) -> bool:
    """Whether `path` exists, as the listing's filesystem sees it; unsure is yes."""
    from pyiceberg.io.fsspec import FsspecFileIO
    from pyiceberg.io.pyarrow import PyArrowFileIO

    try:
        if isinstance(io, PyArrowFileIO):
            from pyarrow.fs import FileType

            scheme, netloc, _ = io.parse_location(path, io.properties)
            info = io.fs_by_scheme(scheme, netloc).get_file_info(_fs_base(io, path))
            return bool(info.type != FileType.NotFound)
        if isinstance(io, FsspecFileIO):
            # `exists` answers no on any error: ask for the entry instead.
            _fsspec_fs(io, path).info(_fs_base(io, path))
            return True
    except FileNotFoundError:
        return False
    except Exception:
        return True
    return True


def _fs_scheme(io: Any, location: str) -> str:
    """The scheme `io` reads `location` with (a default for a scheme-less one)."""
    from pyiceberg.io.pyarrow import PyArrowFileIO

    if isinstance(io, PyArrowFileIO):
        return io.parse_location(location, io.properties)[0]
    return "file"


def _fs_base(io: Any, location: str) -> str:
    """`location` as the path `io`'s filesystem reads it."""
    from pyiceberg.io.fsspec import FsspecFileIO
    from pyiceberg.io.pyarrow import PyArrowFileIO

    if isinstance(io, PyArrowFileIO):
        # The filesystem's own form: forward slashes on Windows too.
        scheme, netloc, path = io.parse_location(location, io.properties)
        return str(io.fs_by_scheme(scheme, netloc).normalize_path(path)).rstrip("/")
    if isinstance(io, FsspecFileIO):
        return _fsspec_fs(io, location)._strip_protocol(location).rstrip("/")
    msg = f"listing files with {type(io).__name__} (PyArrowFileIO or FsspecFileIO)"
    raise NotImplementedError(msg)


def _datetime_ms(value: Any) -> int | None:
    from datetime import datetime

    if isinstance(value, datetime):
        if value.tzinfo is None:
            # A naive time from a listing: UTC, as PyArrow's.
            value = value.replace(tzinfo=UTC)
        return int(value.timestamp() * 1000)
    return None


def _fsspec_mtime_ms(info: dict[str, Any]) -> int | None:
    """The modification time of an fsspec listing entry, None when unknown."""
    for key in ("mtime", "LastModified", "last_modified"):
        value = info.get(key)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            return int(value * 1000)
        if (ms := _datetime_ms(value)) is not None:
            return ms
    return None


def _delete_listed(io: Any, root: str, file: _Listed) -> bool:
    """Delete `file` exactly, through the listing's filesystem."""
    from pyiceberg.io.fsspec import FsspecFileIO
    from pyiceberg.io.pyarrow import PyArrowFileIO

    try:
        if isinstance(io, PyArrowFileIO):
            scheme, netloc, _ = io.parse_location(root, io.properties)
            io.fs_by_scheme(scheme, netloc).delete_file(file.fs_path)
        elif isinstance(io, FsspecFileIO):
            # Object store backends read `?versionId=` and the like out of a key.
            if not _is_local(root) and any(c in file.fs_path for c in "?#"):
                return False
            _fsspec_fs(io, root).rm_file(file.fs_path)
        else:
            return False
    except Exception:
        return False
    return True


def _readable_references(table: pyiceberg.table.Table, name: str) -> set[str]:
    from polars.exceptions import InvalidOperationError

    try:
        return _referenced_files(table)
    except Exception as error:
        msg = (
            f"{_ORPHANS_REFUSED.format(name)}: a manifest list or manifest could not "
            f"be read ({error})"
        )
        raise InvalidOperationError(msg) from error


def _referenced_files(table: pyiceberg.table.Table) -> set[str]:
    """
    Every path the table references, as Iceberg's orphan removal counts them.

    Live content files of every manifest of every snapshot, the manifests, the manifest
    lists, the current metadata file and those of its log, `version-hint.text`, and
    statistics and partition statistics files. An unreadable file raises.
    """
    from pyiceberg.utils.concurrent import ExecutorFactory

    metadata = table.metadata
    io = table.io
    paths: set[str] = set()
    manifests: dict[str, Any] = {}
    for snapshot in metadata.snapshots:
        paths.add(snapshot.manifest_list)
        for manifest in snapshot.manifests(io):
            manifests.setdefault(manifest.manifest_path, manifest)
    paths |= set(manifests)

    def live(manifest: Any) -> list[str]:
        return [
            e.data_file.file_path
            for e in manifest.fetch_manifest_entry(io, discard_deleted=True)
        ]

    executor = ExecutorFactory.get_or_create()
    for entries in executor.map(live, list(manifests.values())):
        paths.update(entries)
    if table.metadata_location:
        paths.add(table.metadata_location)
    paths |= {entry.metadata_file for entry in metadata.metadata_log}
    paths.add(f"{table.location().rstrip('/')}/metadata/version-hint.text")
    paths |= {
        s.statistics_path
        for s in [*metadata.statistics, *metadata.partition_statistics]
    }
    return paths


def _check_location(io: Any, root: str, table_location: str, name: str) -> None:
    """Refuse a `location` that a listing could read as another directory."""
    import os
    from pathlib import Path
    from urllib.parse import urlparse

    from polars.exceptions import InvalidOperationError

    def refuse(why: str) -> None:
        msg = f"{_ORPHANS_REFUSED.format(name)} under '{root}': {why}"
        raise InvalidOperationError(msg)

    if root != table_location and not root.startswith(f"{table_location}/"):
        refuse(f"`location` must be the table location '{table_location}' or below it")
    if any(c in root for c in "?#"):
        refuse("a location with `?` or `#` names another directory to file IOs")
    relative = [p for p in root[len(table_location) :].split("/") if p]
    if any(part in (".", "..") for part in relative):
        refuse("a location with `.` or `..` may leave the table location")
    uri = urlparse(root)
    if not _scheme(root) and (scheme := _fs_scheme(io, root)) != "file":
        # References may use either form: no name rule covers both.
        refuse(
            f"the file IO reads this location without a scheme as a '{scheme}' one; "
            "orphan files are removed only from tables whose location has a scheme"
        )
    if not _is_local(root):
        return
    # Local: the path as the file IO reads it (a drive kept, `~` expanded) must be the
    # one the URI names, and reach the table location without a link.
    base = _fs_base(io, root)
    literal = f"{uri.netloc}{uri.path}" if uri.scheme.lower() == "file" else root
    if re.match(r"^/[A-Za-z]:", literal):
        literal = literal[1:]
    if os.path.normpath(base) != os.path.normpath(literal):
        refuse(f"the file IO reads it as '{base}'")
    table_base = _fs_base(io, table_location)
    if Path(base).resolve() != Path(table_base).resolve().joinpath(*relative):
        refuse("a location through a symbolic link may leave the table location")
    # On a case-insensitive disk `DATA` opens `data`, under names no reference has.
    parent = Path(table_base)
    for part in relative:
        if not parent.is_dir() or part not in {p.name for p in parent.iterdir()}:
            refuse(
                f"'{part}' is not a directory of '{parent}' (names are case-sensitive)"
            )
        parent = parent / part


def _scheme(location: str) -> str:
    """The scheme of `location`, as PyIceberg reads it (a Windows drive is none)."""
    import os
    from urllib.parse import urlparse

    return "" if os.path.splitdrive(location)[0] else urlparse(location).scheme


def _is_local(location: str) -> bool:
    """Whether `location` is a local path: no scheme, `file:`, or a Windows drive."""
    from urllib.parse import urlparse

    scheme = urlparse(location).scheme.lower()
    return scheme in ("", "file") or len(scheme) == 1


def _without_local_links(files: list[_Listed], root: str) -> list[_Listed]:
    """`files` without local files reached through a symbolic link below `root`."""
    from pathlib import Path

    if not _is_local(root):
        return files
    kept = []
    for f in files:
        base = f.fs_path[: len(f.fs_path) - len(f.uri) + len(root)]
        relative = f.uri[len(root) + 1 :].split("/")
        if Path(f.fs_path).resolve() == Path(base).resolve().joinpath(*relative):
            kept.append(f)
    return kept


def _check_listing(
    io: Any,
    listed: list[_Listed],
    referenced: set[str],
    metadata_file: str | None,
    root: str,
    table_location: str,
    name: str,
) -> None:
    """
    Refuse when the listing misses a file the table references under `root`.

    Such a listing would make every file an orphan. A dangling reference does not count.
    """
    from polars.exceptions import InvalidOperationError

    version_hint = f"{table_location}/metadata/version-hint.text"
    uris = {f.uri for f in listed}
    rest = [
        p
        for p in referenced
        if p.startswith(f"{root}/") and p != version_hint and p not in uris
    ]
    if not rest:
        return
    names = {n for f in listed for n in _path_names(f.uri)}
    missing = sorted(p for p in rest if _path_names(p).isdisjoint(names))
    # The current metadata file exists; others may be dangling references.
    present = next((p for p in missing if p == metadata_file or _exists(io, p)), None)
    if present is not None:
        msg = (
            f"{_ORPHANS_REFUSED.format(name)}: the listing of '{root}' misses "
            f"'{present}', which the table references and exists"
        )
        raise InvalidOperationError(msg)


def _orphans(
    io: Any,
    listed: list[_Listed],
    referenced: set[str],
    metadata: TableMetadata,
    older: int,
    root: str,
    equal_schemes: dict[str, str],
    equal_authorities: dict[str, str],
    prefix_mismatch_mode: str,
    name: str,
) -> list[_Listed]:
    """The listed files to delete."""
    import posixpath
    from urllib.parse import urlparse

    from polars.exceptions import InvalidOperationError

    hidden_ok = tuple(
        f"{field.name}="
        for spec in metadata.partition_specs
        for field in spec.fields
        if field.name.startswith(("_", "."))
    )

    def hidden(file: _Listed) -> bool:
        parts = file.uri[len(root) + 1 :].split("/")
        return any(
            part.startswith(("_", "."))
            and not (i < len(parts) - 1 and part.startswith(hidden_ok))
            for i, part in enumerate(parts)
        )

    def mapped(path: str) -> str:
        """`path` with the scheme and authority `equal_schemes` and co. map it to."""
        scheme, sep, rest = path.partition("://")
        if not sep:
            return path
        authority, slash, key = rest.partition("/")
        scheme = schemes.get(scheme.lower(), scheme)
        authority = authorities.get(authority, authority)
        return f"{scheme}://{authority}{slash}{key}"

    schemes = {
        s.strip().lower(): target
        for sources, target in equal_schemes.items()
        for s in sources.split(",")
    }
    authorities = {
        a.strip(): target
        for sources, target in equal_authorities.items()
        for a in sources.split(",")
    }

    def all_names(path: str) -> frozenset[str]:
        names = _path_names(path)
        for alias in (mapped(path), qualified(path), mapped(qualified(path))):
            if alias != path:
                names |= _path_names(alias)
        return names

    def qualified(path: str) -> str:
        """A path without a scheme, as the file IO reads it (`DEFAULT_SCHEME`)."""
        if _scheme(path) or (scheme := _fs_scheme(io, path)) == "file":
            return path
        return f"{scheme}://{_fs_base(io, path)}"

    # Unreferenced: not referenced exactly, then by no name (aliases).
    files = [f for f in listed if not hidden(f) and f.uri not in referenced]
    if files:
        names: set[str] = set()
        for path in referenced:
            names |= all_names(path)
        files = [f for f in files if all_names(f.uri).isdisjoint(names)]

    # A directory without referenced files where an unreferenced file is new (or of
    # unknown age): a write still running there. Its files are kept.
    unreferenced = {f.uri for f in files}
    referencing: set[str] = set()
    busy: set[str] = set()
    for f in listed:
        if hidden(f):
            continue
        directory = posixpath.dirname(f.uri)
        if f.uri not in unreferenced:
            referencing.add(directory)
        elif f.mtime_ms is None or f.mtime_ms >= older:
            busy.add(directory)
    candidates = [
        f
        for f in files
        if f.mtime_ms is not None
        and f.mtime_ms < older
        and (
            posixpath.dirname(f.uri) in referencing
            or posixpath.dirname(f.uri) not in busy
        )
    ]

    # The path of a referenced file under another scheme or authority: Java's conflict.
    if candidates:

        def path_of(uri: str) -> str:
            return posixpath.normpath(urlparse(qualified(uri)).path or "/")

        paths = {path_of(p) for p in referenced}
        conflicts = [f for f in candidates if path_of(f.uri) in paths]
        if conflicts and prefix_mismatch_mode == "ERROR":
            msg = (
                f"{_ORPHANS_REFUSED.format(name)}: {len(conflicts)} listed file(s) have "
                "the path of a referenced file under another scheme or authority, such "
                f"as '{conflicts[0].uri}'; map them with `equal_schemes` or "
                "`equal_authorities`, or keep them with `prefix_mismatch_mode => 'IGNORE'`"
            )
            raise InvalidOperationError(msg)
        conflicting = {f.uri for f in conflicts}
        candidates = [f for f in candidates if f.uri not in conflicting]
    return candidates
