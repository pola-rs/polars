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
