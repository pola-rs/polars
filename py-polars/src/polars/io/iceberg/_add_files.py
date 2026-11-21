"""Register existing Parquet files in an Iceberg table, for SQL `CALL ... add_files`.

Not PyIceberg's `Transaction.add_files`: it infers partition values from statistics
that may not cover every row (apache/iceberg-python#4191), checks for duplicates only
before a commit retry (apache/iceberg-python#4445) and maps the statistics of files with
field ids by name (apache/iceberg-python#4359). Its building blocks are used instead,
with the checks below. Footers are read with pyarrow; reading them in Rust is the
follow-up if imports of many files are slow.
"""

from __future__ import annotations

import contextlib
import re
import uuid
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn
from urllib.parse import unquote, urlparse

if TYPE_CHECKING:
    import pyarrow.parquet as pq
    import pyiceberg.table
    from pyiceberg.manifest import DataFile, ManifestFile
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.schema import Schema
    from pyiceberg.table import Transaction

# Hive's name for a NULL partition value in a folder name.
_HIVE_NULL = "__HIVE_DEFAULT_PARTITION__"
_HIDDEN_PREFIXES = ("_", ".")
_TIMESTAMP = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}[ T][0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]{1,6})?"
    r"(Z|[+-][0-9]{2}:[0-9]{2})?"
)
_INSERT_HINT = (
    "To copy the rows instead, use `INSERT INTO <table> SELECT * FROM "
    "read_parquet('<files>')`."
)


class _DuplicateFiles(Exception):
    """Files the table already references, found when a commit attempt is planned."""

    def __init__(self, paths: list[str]) -> None:
        self.paths = paths


def add_files(
    table: pyiceberg.table.Table,
    location: str,
    *,
    check_duplicate_files: bool,
) -> tuple[int, int | None]:
    """
    Register the Parquet files at `location` in `table`, in one append commit.

    `location` is a folder (its visible files, or `col=value` folders holding them) or
    one file, read with the table's FileIO. Nothing is copied: the table references the
    files where they are. Returns the numbers of added files and of changed partitions
    (None if the snapshot summary has none).
    """
    from pyiceberg.exceptions import (
        CommitFailedException,
        CommitStateUnknownException,
        ValidationException,
    )
    from pyiceberg.table import Table, TableProperties
    from pyiceberg.table.update import AssertCurrentSchemaId
    from pyiceberg.utils.concurrent import ExecutorFactory

    from polars.exceptions import DuplicateError
    from polars.io.iceberg._maintenance import _landed
    from polars.io.iceberg._sink import (
        _SINK_UUID_PROPERTY,
        _AlreadyCommitted,
        _copy_table_state,
        _sink_transaction,
        _SinkCatalog,
    )

    name = ".".join(table.name())
    if table.format_version > 2:
        msg = (
            f"add_files to an Iceberg v{table.format_version} table is not supported "
            f"yet: table '{name}'"
        )
        raise NotImplementedError(msg)

    schema = table.schema()
    spec = table.spec()
    root, files = _list_files(table, location)
    folders = _hive_values(files, root, spec, schema, name)
    data_files = list(
        ExecutorFactory.get_or_create().map(
            lambda pv: _data_file(table, pv[0], pv[1], name),
            zip(files, folders, strict=True),
        )
    )

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
    # the files were checked against this name mapping, or get the schema's
    mapping_key = TableProperties.DEFAULT_NAME_MAPPING
    loaded_mapping = table.metadata.properties.get(mapping_key)
    sets_mapping = loaded_mapping is None
    new_mapping = schema.name_mapping.model_dump_json()

    def same_mapping(metadata: Any) -> None:
        # the one checked against, or the one this commit sets
        if metadata.properties.get(mapping_key) not in (loaded_mapping, new_mapping):
            msg = f"the name mapping of table '{name}' changed during add_files"
            raise ValidationException(msg)

    try:
        with _sink_transaction(
            wrapped,
            commit_uuid,
            spec.spec_id,
            schema_id=schema.schema_id,
            check=same_mapping,
        ) as tx:
            # the files were checked against this schema
            tx._stage(
                (),
                (AssertCurrentSchemaId(current_schema_id=schema.schema_id),),  # type: ignore[call-arg]
            )
            # files without field ids are read through the name mapping
            if sets_mapping:
                tx.set_properties({mapping_key: new_mapping})
            producers.append(
                _append(
                    tx,
                    data_files,
                    {_SINK_UUID_PROPERTY: commit_uuid},
                    check_duplicate_files=check_duplicate_files,
                )
            )
            producers[0].commit()
    except _AlreadyCommitted as error:
        _copy_table_state(error.table, table)
        return _counts(table, commit_uuid)
    except CommitStateUnknownException:
        raise
    except Exception as error:
        # not the last request's own refusal: did the append commit?
        if catalog._snapshot_ids and error is not catalog._raised:
            if _landed(table, commit_uuid):
                table.refresh()
                return _counts(table, commit_uuid)
            if not isinstance(catalog._raised, CommitFailedException):
                msg = "a failed Iceberg commit may still land"
                raise CommitStateUnknownException(msg) from error
        for producer in producers:
            with contextlib.suppress(Exception):
                producer._clean_all_uncommitted()
        if isinstance(error, _DuplicateFiles):
            msg = (
                f"add_files: table '{name}' already references "
                f"{_some(error.paths)}; nothing was added"
            )
            raise DuplicateError(msg) from None
        # the catalog's refusal of the request
        if (
            sets_mapping
            and error is catalog._raised
            and not isinstance(error, CommitFailedException)
        ):
            error.add_note(
                f"add_files also set the name mapping of table '{name}' "
                f"({mapping_key}), which files without field "
                f"ids need; some catalogs refuse it. {_INSERT_HINT}"
            )
        raise
    _copy_table_state(wrapped, table)
    return _counts(table, commit_uuid)


def _counts(table: pyiceberg.table.Table, commit_uuid: str) -> tuple[int, int | None]:
    """The added files and changed partitions in the summary of the commit."""
    from polars.io.iceberg._sink import _SINK_UUID_PROPERTY

    for snapshot in reversed(table.metadata.snapshots):
        summary = snapshot.summary
        if summary is not None and summary.get(_SINK_UUID_PROPERTY) == commit_uuid:
            changed = summary.get("changed-partition-count")
            return (
                int(summary.get("added-data-files", 0)),
                None if changed is None else int(changed),
            )
    msg = "add_files: the committed snapshot is not in the table"
    raise AssertionError(msg)


def _append(
    tx: Transaction,
    data_files: list[DataFile],
    snapshot_properties: dict[str, str],
    *,
    check_duplicate_files: bool,
) -> Any:
    """A fast append of `data_files`, checked for duplicates on each commit attempt."""
    from pyiceberg.manifest import ManifestContent
    from pyiceberg.table.snapshots import Operation
    from pyiceberg.table.update.snapshot import _FastAppendFiles

    paths = {f.file_path for f in data_files}

    # private in PyIceberg 0.12, as its own producers. A fast append: a merging
    # append rewrites manifests, which PyIceberg 0.12 may round (#30832).
    class _AddFiles(_FastAppendFiles):  # type: ignore[misc]
        def _existing_manifests(self) -> list[ManifestFile]:
            # planned on each attempt, from the table as that attempt sees it
            manifests = super()._existing_manifests()
            if check_duplicate_files:
                taken = sorted(
                    entry.data_file.file_path
                    for manifest in manifests
                    if manifest.content == ManifestContent.DATA
                    for entry in self.fetch_manifest_entry(manifest, True)
                    if entry.data_file.file_path in paths
                )
                if taken:
                    raise _DuplicateFiles(taken)
            return manifests

    producer = _AddFiles(
        operation=Operation.APPEND,
        transaction=tx,
        io=tx._table.io,
        snapshot_properties=snapshot_properties,
    )
    for data_file in data_files:
        producer.append_data_file(data_file)
    return producer


def _list_files(table: pyiceberg.table.Table, location: str) -> tuple[str, list[str]]:
    """
    The folder `location` and its visible files, or `location` if it is a file.

    Listed with the table's FileIO, which also reads the files. Names starting with
    `_` or `.` are hidden, except `_col=value` folders.
    """
    from polars.exceptions import ComputeError, InvalidOperationError

    if any(c in location for c in "?#"):
        # a URL's query or fragment, not part of a path
        msg = f"add_files does not support '?' or '#' in the location '{location}'"
        raise InvalidOperationError(msg)
    # a local path as the location readers of the table will open
    local = location.removeprefix("file://")
    if re.match(r"^/?[A-Za-z]:[\\/]", local):
        location = "file://" + Path(local.lstrip("/")).resolve().as_posix()
    else:
        uri = urlparse(location)
        if not uri.scheme:
            location = "file://" + Path(location).resolve().as_posix()
        elif uri.scheme == "file" and not uri.netloc:
            location = "file://" + Path(uri.path).resolve().as_posix()
    root = location.rstrip("/")
    try:
        found = _list(table.io, root)
    except OSError as error:
        if not isinstance(error, (FileNotFoundError, PermissionError)):
            error.add_note(
                f"add_files could not list '{location}'; a single file can be given "
                "instead of a folder"
            )
        raise
    if found is None:
        return root, [root]
    files = sorted(f"{root}/{rel}" for rel in found)
    if odd := [f for f in files if any(c in f for c in "?#")]:
        msg = f"add_files does not support '?' or '#' in file names: '{odd[0]}'"
        raise InvalidOperationError(msg)
    if not files:
        msg = (
            f"add_files found no files under '{location}'; check the path (a missing "
            "folder on object storage lists as empty)"
        )
        raise ComputeError(msg)
    return root, files


def _list(io: Any, root: str) -> list[str] | None:
    """
    The visible files under the folder `root`, relative to it; None if it is a file.

    Hidden folders are not entered, and a folder that cannot be listed fails the call.
    """
    from pyiceberg.io.fsspec import FsspecFileIO
    from pyiceberg.io.pyarrow import PyArrowFileIO

    if isinstance(io, PyArrowFileIO):
        from pyarrow.fs import FileSelector, FileType

        scheme, netloc, path = io.parse_location(root)
        fs = io.fs_by_scheme(scheme, netloc)
        info = fs.get_file_info(path)
        if info.type == FileType.File:
            return None
        if info.type != FileType.Directory:
            return []

        def entries(folder: str) -> list[tuple[str, bool]]:
            return [
                (i.path, i.type == FileType.Directory)
                for i in fs.get_file_info(FileSelector(folder))
                if i.type in (FileType.File, FileType.Directory)
            ]

        base = path.rstrip("/") + "/"
    elif isinstance(io, FsspecFileIO):
        uri = urlparse(root)
        fs = io.get_fs(uri.scheme or "file", uri.hostname)
        # listings are cached across calls on the same filesystem
        fs.invalidate_cache()
        try:
            kind = fs.info(root)["type"]
        except FileNotFoundError:
            return []
        if kind == "file":
            return None

        def entries(folder: str) -> list[tuple[str, bool]]:
            return [
                (e["name"], e["type"] == "directory")
                for e in fs.ls(folder, detail=True)
                if e["type"] in ("file", "directory")
            ]

        path = fs._strip_protocol(root)
        base = path.rstrip("/") + "/"
    else:
        msg = f"add_files with a {type(io).__name__} is not supported"
        raise NotImplementedError(msg)

    found: list[str] = []
    pending = [path]
    while pending:
        for entry, is_folder in entries(pending.pop()):
            relative = entry[len(base) :]
            if not entry.startswith(base) or not relative:
                continue
            if not is_folder:
                found.append(relative)
            elif _visible(relative + "/x"):
                pending.append(entry)
    return [f for f in found if _visible(f)]


def _visible(relative: str) -> bool:
    *folders, file = relative.split("/")
    # `_col=value` folders are partitions
    return not file.startswith(_HIDDEN_PREFIXES) and all(
        not f.startswith(".") and (not f.startswith("_") or "=" in f) for f in folders
    )


def _hive_values(
    files: list[str], root: str, spec: PartitionSpec, schema: Schema, name: str
) -> list[dict[int, str | None]]:
    """
    The partition values of each file's `col=value` folders, by source field id.

    All files are directly in `root`, or all sit in folders naming the identity
    partition source columns of the table's spec, in order. Values are unescaped.
    """
    from pyiceberg.transforms import IdentityTransform, VoidTransform

    from polars.exceptions import InvalidOperationError

    folders = [
        [] if path == root else path[len(root) + 1 :].split("/")[:-1] for path in files
    ]
    if not any(folders):
        return [{} for _ in files]

    fields = [f for f in spec.fields if not isinstance(f.transform, VoidTransform)]
    names = [schema.find_column_name(f.source_id) for f in fields]
    if not fields or not all(
        isinstance(f.transform, IdentityTransform) for f in fields
    ):
        msg = (
            f"add_files: the files under '{root}' sit in subfolders, which only a "
            f"table partitioned by identity columns can take them from: table '{name}'"
        )
        raise InvalidOperationError(msg)
    values = []
    for path, parts in zip(files, folders, strict=True):
        pairs = [p.split("=", 1) for p in parts]
        if any(len(p) != 2 for p in pairs) or [unquote(k) for k, _ in pairs] != names:
            msg = (
                f"add_files: '{path}' is not in `col=value` folders for the partition "
                f"columns {names} of table '{name}', in that order. {_INSERT_HINT}"
            )
            raise InvalidOperationError(msg)
        values.append(
            {
                f.source_id: None if raw == _HIVE_NULL else unquote(raw)
                for f, (_, raw) in zip(fields, pairs, strict=True)
            }
        )
    return values


def _data_file(
    table: pyiceberg.table.Table,
    path: str,
    folder_values: dict[int, str | None],
    name: str,
) -> DataFile:
    """The data file of the Parquet file `path`, from one footer read."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyiceberg.io.pyarrow import (
        UnsupportedPyArrowTypeException,
        _check_pyarrow_schema_compatible,
        compute_statistics_plan,
        data_file_statistics_from_parquet_metadata,
        parquet_path_to_id_mapping,
    )
    from pyiceberg.manifest import DataFile, DataFileContent, FileFormat
    from pyiceberg.typedef import Record

    from polars.exceptions import ComputeError, InvalidOperationError, SchemaError

    metadata = table.metadata
    schema = metadata.schema()
    input_file = table.io.new_input(path)
    try:
        with input_file.open() as stream:
            footer = pq.read_metadata(stream)
            size = _size(stream)
    except pa.ArrowInvalid as error:
        msg = f"add_files: '{path}' is not a Parquet file"
        raise ComputeError(msg) from error
    arrow_schema = footer.schema.to_arrow_schema()
    if any(_has_field_id(field) for field in arrow_schema):
        msg = (
            f"add_files: '{path}' has Iceberg field ids, which are not supported yet: "
            f"only files without field ids can be added to table '{name}'. {_INSERT_HINT}"
        )
        raise InvalidOperationError(msg)
    try:
        _check_pyarrow_schema_compatible(
            schema, arrow_schema, format_version=metadata.format_version
        )
    except (ValueError, TypeError, UnsupportedPyArrowTypeException) as error:
        msg = (
            f"add_files: the columns of '{path}' do not fit table '{name}'. "
            f"{_INSERT_HINT}"
        )
        raise SchemaError(msg) from error
    # e.g. a legacy list layout (`list.item`) that the table's column paths do not name
    paths = parquet_path_to_id_mapping(schema)
    columns = [footer.schema.column(i).path for i in range(footer.num_columns)]
    if unnamed := [c for c in columns if c not in paths]:
        msg = (
            f"add_files: the columns {unnamed} of '{path}' are in a layout table "
            f"'{name}' does not name. {_INSERT_HINT}"
        )
        raise SchemaError(msg)
    # files without field ids are read through the name mapping the table has
    if (mapping := metadata.name_mapping()) is not None:
        _check_name_mapping(arrow_schema, mapping, metadata, path, name)
    try:
        statistics = data_file_statistics_from_parquet_metadata(
            parquet_metadata=footer,
            stats_columns=compute_statistics_plan(schema, metadata.properties),
            parquet_column_mapping=paths,
        )
        partition = [
            _partition_value(footer, field, schema, folder_values, path, name)
            for field in metadata.spec().fields
        ]
    except (ValueError, TypeError) as error:
        msg = (
            f"add_files: the statistics of '{path}' do not fit table '{name}'. "
            f"{_INSERT_HINT}"
        )
        raise SchemaError(msg) from error
    serialized = statistics.to_serialized_dict()
    # PyIceberg keeps the bounds of the row groups that have them: those would prune
    # the rows of a row group without
    for field_id in _incomplete_bounds(footer, paths):
        serialized["lower_bounds"].pop(field_id, None)
        serialized["upper_bounds"].pop(field_id, None)
    return DataFile.from_args(
        content=DataFileContent.DATA,
        file_path=path,
        file_format=FileFormat.PARQUET,
        partition=Record(*partition),
        file_size_in_bytes=len(input_file) if size is None else size,
        sort_order_id=None,
        spec_id=metadata.default_spec_id,
        equality_ids=None,
        key_metadata=None,
        **serialized,
    )


def _size(stream: Any) -> int | None:
    """The size of an open file, without another request."""
    size = getattr(stream, "size", None)
    if callable(size):
        size = size()
    return size if isinstance(size, int) else None


def _incomplete_bounds(footer: pq.FileMetaData, paths: dict[str, int]) -> set[int]:
    """The field ids of the columns with a non-null row group lacking a min or max."""
    incomplete = set()
    for r in range(footer.num_row_groups):
        row_group = footer.row_group(r)
        for i in range(row_group.num_columns):
            chunk = row_group.column(i)
            stats = chunk.statistics if chunk.is_stats_set else None
            if (
                stats is not None
                and stats.has_null_count
                and stats.null_count == chunk.num_values
            ):
                continue
            if stats is None or not stats.has_min_max:
                incomplete.add(paths[chunk.path_in_schema])
    return incomplete


def _check_name_mapping(
    arrow_schema: Any, mapping: Any, metadata: Any, path: str, name: str
) -> None:
    """Refuse a file that the table's name mapping maps otherwise than its schema."""
    from pyiceberg.io.pyarrow import pyarrow_to_schema
    from pyiceberg.schema import index_by_name

    from polars.exceptions import SchemaError

    msg = (
        f"add_files: the name mapping of table '{name}' does not map the columns of "
        f"'{path}' to the fields of its schema. {_INSERT_HINT}"
    )
    try:
        mapped = pyarrow_to_schema(
            arrow_schema, name_mapping=mapping, format_version=metadata.format_version
        )
    except ValueError as error:
        raise SchemaError(msg) from error
    expected = index_by_name(metadata.schema())
    if any(expected.get(column) != i for column, i in index_by_name(mapped).items()):
        raise SchemaError(msg)


def _has_field_id(field: Any) -> bool:
    if field.metadata and b"PARQUET:field_id" in field.metadata:
        return True
    # struct, list, large/fixed-size list and map (its entries struct) children
    return any(_has_field_id(field.type.field(i)) for i in range(field.type.num_fields))


def _partition_value(
    footer: pq.FileMetaData,
    field: PartitionField,
    schema: Schema,
    folder_values: dict[int, str | None],
    path: str,
    name: str,
) -> Any:
    """
    The value of partition `field` for the file `path`, proven or refused.

    From the `col=value` folder if there is one, else from the footer's statistics.
    When both exist, they must agree. Nothing is read beyond the footer.
    """
    from pyiceberg.partitioning import partition_record_value
    from pyiceberg.transforms import VoidTransform
    from pyiceberg.types import DoubleType, FloatType

    from polars.exceptions import InvalidOperationError

    if isinstance(field.transform, VoidTransform):
        return None
    source = schema.find_field(field.source_id)
    column = schema.find_column_name(field.source_id)
    assert column is not None
    if isinstance(source.field_type, (FloatType, DoubleType)):
        # NaN is left out of Parquet statistics; float folder values are not exact
        msg = (
            f"add_files: partition column '{column}' of table '{name}' is floating-point, "
            f"which add_files does not support. {_INSERT_HINT}"
        )
        raise InvalidOperationError(msg)
    in_file = (
        None
        if footer.num_rows == 0
        else _footer_values(footer, column, source, path, name)
    )
    if in_file is not None:
        in_file = [partition_record_value(field, v, schema) for v in in_file]

    if field.source_id in folder_values:
        raw = folder_values[field.source_id]
        folder = (
            None
            if raw is None
            else partition_record_value(
                field, _parse_folder_value(raw, source, path, name), schema
            )
        )
        if in_file is None or footer.num_rows == 0:
            # a nested column's value also depends on its parents being non-null
            if footer.num_rows and field.source_id not in {
                f.field_id for f in schema.fields
            }:
                msg = (
                    f"add_files: '{path}' lacks the partition column '{column}' of "
                    f"table '{name}', which its folder can only give for an optional "
                    f"top-level column. {_INSERT_HINT}"
                )
                raise InvalidOperationError(msg)
            return folder
        if any(v != folder for v in in_file):
            msg = (
                f"add_files: the folder of '{path}' says {column} = {raw!r}, but the "
                f"file's statistics do not show only that value: table '{name}'. "
                f"{_INSERT_HINT}"
            )
            raise InvalidOperationError(msg)
        return folder

    if footer.num_rows == 0:
        # no row to prove a value for
        return None
    if in_file is None:
        msg = (
            f"add_files: '{path}' lacks the partition column '{column}' of table "
            f"'{name}'. {_INSERT_HINT}"
        )
        raise InvalidOperationError(msg)
    if in_file == [None]:
        return None
    lower, upper = in_file
    if not field.transform.preserves_order and lower != upper:
        msg = (
            f"add_files: the partition of '{path}' by {field.transform}({column}) "
            f"cannot be proven from its statistics: table '{name}'. {_INSERT_HINT}"
        )
        raise InvalidOperationError(msg)
    transform = field.transform.transform(source.field_type)
    low, high = transform(lower), transform(upper)
    if low != high:
        msg = (
            f"add_files: the statistics of '{path}' do not show a single value of "
            f"partition field '{field.name}' (the file holds several, or its "
            f"statistics are truncated): table '{name}'. {_INSERT_HINT}"
        )
        raise InvalidOperationError(msg)
    return low


def _footer_values(
    footer: pq.FileMetaData, column: str, source: Any, path: str, name: str
) -> list[Any] | None:
    """
    The footer's proof of the values of `column`, or None if the file lacks it.

    `[None]` if every row is null, else `[min, max]`. Refuses when the statistics of
    a row group are missing or the file mixes nulls with values.
    """
    from pyiceberg.io.pyarrow import StatsAggregator

    from polars.exceptions import InvalidOperationError

    index = next(
        (
            i
            for i in range(footer.num_columns)
            if footer.schema.column(i).path == column
        ),
        None,
    )
    if index is None:
        return None

    def refuse(reason: str) -> NoReturn:
        msg = (
            f"add_files: the partition value of '{path}' cannot be proven: {reason} "
            f"(column '{column}' of table '{name}'). {_INSERT_HINT}"
        )
        raise InvalidOperationError(msg)

    nulls = 0
    aggregate: Any = None
    for r in range(footer.num_row_groups):
        chunk = footer.row_group(r).column(index)
        stats = chunk.statistics if chunk.is_stats_set else None
        if stats is None or not stats.has_null_count:
            refuse("a row group has no statistics")
        nulls += stats.null_count
        if stats.null_count == chunk.num_values:
            continue
        if not stats.has_min_max:
            refuse("a row group has no minimum and maximum")
        if aggregate is None:
            aggregate = StatsAggregator(source.field_type, stats.physical_type)
        aggregate.update_min(stats.min)
        aggregate.update_max(stats.max)
    if nulls == footer.num_rows:
        return [None]
    if nulls:
        refuse("the file holds nulls and values")
    assert aggregate is not None
    return [aggregate.current_min, aggregate.current_max]


def _parse_folder_value(raw: str, source: Any, path: str, name: str) -> Any:
    """The `col=value` folder value as a value of the source column's type."""
    from datetime import date, datetime
    from decimal import Decimal, InvalidOperation, localcontext

    from pyiceberg.types import (
        BooleanType,
        DateType,
        DecimalType,
        IntegerType,
        LongType,
        StringType,
        TimestampType,
        TimestamptzType,
    )

    from polars.exceptions import InvalidOperationError

    t = source.field_type
    value: Any = None
    if isinstance(t, StringType):
        value = raw
    elif isinstance(t, (IntegerType, LongType)):
        bits = 31 if isinstance(t, IntegerType) else 63
        if re.fullmatch(r"[+-]?[0-9]+", raw) and -(2**bits) <= int(raw) < 2**bits:
            value = int(raw)
    elif isinstance(t, BooleanType):
        value = {"true": True, "false": False}.get(raw.lower())
    elif isinstance(t, DateType):
        with contextlib.suppress(ValueError):
            if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", raw):
                value = date.fromisoformat(raw)
    elif isinstance(t, (TimestampType, TimestamptzType)):
        # Hive writes `2024-06-01 10:00:00`, others `2024-06-01T10:00:00`; at most
        # microseconds, an offset only as `Z` or `+hh:mm`
        with contextlib.suppress(ValueError, OverflowError):
            if not _TIMESTAMP.fullmatch(raw):
                raise ValueError
            parsed = datetime.fromisoformat(raw)
            # a time zone only for timestamptz, which must have one
            if (parsed.tzinfo is not None) == isinstance(t, TimestamptzType):
                value = parsed.astimezone(UTC) if parsed.tzinfo else parsed
    elif isinstance(t, DecimalType):
        with contextlib.suppress(InvalidOperation), localcontext() as context:
            # Iceberg decimals have up to 38 digits, more than the default context
            context.prec = 80
            number = Decimal(raw)
            exponent = number.as_tuple().exponent
            if (
                number.is_finite()
                and isinstance(exponent, int)
                and -exponent <= t.scale
            ):
                exact = number.quantize(Decimal(1).scaleb(-t.scale))
                if len(exact.as_tuple().digits) <= t.precision:
                    value = exact
    if value is None:
        msg = (
            f"add_files: the folder value {raw!r} of '{path}' is not a "
            f"{t} for column '{source.name}' of table '{name}', or that type is not "
            f"supported in folder names. {_INSERT_HINT}"
        )
        raise InvalidOperationError(msg)
    return value


def _some(paths: list[str]) -> str:
    shown = ", ".join(f"'{p}'" for p in paths[:3])
    return shown if len(paths) <= 3 else f"{shown} and {len(paths) - 3} more"
