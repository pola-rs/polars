"""Copy-on-write deletes of Iceberg rows (`DELETE`, `delete_iceberg`, overwrites)."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Collection

    import pyiceberg.expressions
    import pyiceberg.table
    from pyiceberg.manifest import DataFile
    from pyiceberg.schema import Schema
    from pyiceberg.table import FileScanTask

    from polars._typing import PolarsDataType, StorageOptionsDict
    from polars.expr import Expr
    from polars.io.iceberg._sink import _ReplaceFiles
    from polars.lazyframe import LazyFrame


def _delete_whole_files(
    table: pyiceberg.table.Table,
    delete_filter: pyiceberg.expressions.BooleanExpression,
    *,
    refused: str,
    unknown: str,
) -> bool:
    """
    Delete the rows of `table` matching `delete_filter` by dropping data files.

    One commit, from metadata only, if every file with a matching row matches whole.
    Returns False, committing nothing, if a file needs a rewrite. On failure, the note
    `refused` if nothing was committed, else `unknown`.
    """
    import uuid

    from pyiceberg.exceptions import (
        CommitFailedException,
        CommitStateUnknownException,
        ValidationException,
    )
    from pyiceberg.table import Table

    from polars.io.iceberg._sink import (
        _SINK_UUID_PROPERTY,
        _AlreadyCommitted,
        _copy_table_state,
        _raise_if_spec_changed,
        _sink_transaction,
        _SinkCatalog,
    )

    # as a sink's commit: a failed commit is checked for having landed before
    # PyIceberg retries it or deletes its manifests
    commit_uuid = str(uuid.uuid4())
    properties = {_SINK_UUID_PROPERTY: commit_uuid}

    catalog = _SinkCatalog(table.catalog, commit_uuid, verbose=False)
    wrapped = Table(
        table.name(),
        table.metadata,
        table.metadata_location,
        table.io,
        catalog,  # type: ignore[arg-type]
        table.config,
    )
    spec_id = table.metadata.default_spec_id
    try:
        try:
            with _sink_transaction(wrapped, commit_uuid, spec_id) as tx:
                # files whose rows all match, from column statistics; not a `with`:
                # committed only if no file needs a rewrite
                delete = tx.update_snapshot(snapshot_properties=properties).delete()
                delete.delete_by_predicate(delete_filter)
                try:
                    rewrite = delete.rewrites_needed
                except Exception:
                    with contextlib.suppress(Exception):
                        delete._clean_all_uncommitted()
                    raise
                if rewrite:
                    # computing that wrote manifests the commit will not use
                    delete._clean_all_uncommitted()
                    return False
                delete.commit()
        except (_AlreadyCommitted, CommitStateUnknownException, ValidationException):
            raise
        except Exception as error:
            if isinstance(error, CommitFailedException):
                _raise_if_spec_changed(table, spec_id, error)
            # a failure after a commit request, not of it: did the commit land?
            _copy_table_state(catalog.landed_after(table, error), table)
            return True
    except _AlreadyCommitted as error:
        _copy_table_state(error.table, table)
        return True
    except CommitStateUnknownException as error:
        error.add_note(unknown)
        raise
    except Exception as error:
        error.add_note(refused)
        raise
    _copy_table_state(wrapped, table)
    return True


def _delete_rows(
    table: pyiceberg.table.Table,
    delete_filter: pyiceberg.expressions.BooleanExpression,
    predicate: Expr,
    storage_options: StorageOptionsDict | None,
    *,
    refused: str,
    unknown: str,
) -> None:
    """
    Delete the rows of `table` matching `delete_filter` (`predicate`) in one commit.

    Data files whose rows all match are dropped. If a file matches only partly, the
    candidate files are read and those with a match replaced by files with the other
    rows (copy-on-write, as `UPDATE`). On failure, the note `refused` if nothing was
    committed, else `unknown`.
    """
    from pyiceberg.exceptions import CommitStateUnknownException
    from pyiceberg.expressions import AlwaysTrue

    from polars.io.iceberg._sink import _ReplaceFiles
    from polars.io.iceberg.functions import _scan_iceberg_files

    if _delete_whole_files(table, delete_filter, refused=refused, unknown=unknown):
        return

    partial: list[FileScanTask] = []
    try:
        if delete_filter == AlwaysTrue():
            # every file goes, also one whose row count was not recorded
            removed = _live_data_files(table)
        else:
            whole, partial = _matching_files(
                table,
                _candidate_tasks(table, delete_filter),
                predicate,
                storage_options,
            )
            removed = [t.file for t in whole + partial]
    except Exception as error:
        error.add_note(refused)
        raise
    if not removed:
        return
    replace = _ReplaceFiles(
        data_files=removed,
        conflict_filter=delete_filter,
        schema_id=table.metadata.current_schema_id,
        spec_id=table.metadata.default_spec_id,
        isolation="delete",
    )
    rows = (
        _scan_iceberg_files(table, partial, storage_options).remove(predicate)
        if partial
        else None
    )
    try:
        _write_rows(table, rows, "append", storage_options, replace=replace)
    except CommitStateUnknownException as error:
        error.add_note(unknown)
        raise
    except Exception as error:
        error.add_note(refused)
        raise


def _matching_files(
    table: pyiceberg.table.Table,
    tasks: list[FileScanTask],
    predicate: Expr,
    storage_options: StorageOptionsDict | None,
) -> tuple[list[FileScanTask], list[FileScanTask]]:
    """Of `tasks`, the files whose live rows all match `predicate`, and those with some.

    A NULL condition keeps a row.
    """
    from polars.functions import collect_all
    from polars.functions import len as count
    from polars.io.iceberg.functions import _scan_iceberg_files

    counts = collect_all(
        _scan_iceberg_files(table, [task], storage_options).select(
            matched=predicate.fill_null(False).sum(), total=count()
        )
        for task in tasks
    )
    whole, partial = [], []
    for task, n in zip(tasks, counts, strict=True):
        matched, total = n.row(0)
        if matched == total and matched > 0:
            whole.append(task)
        elif matched > 0:
            partial.append(task)
    return whole, partial


def _live_data_files(table: pyiceberg.table.Table) -> list[DataFile]:
    """The data files of `table`'s current snapshot, read from its manifests."""
    from pyiceberg.manifest import ManifestContent

    snapshot = table.current_snapshot()
    if snapshot is None:
        return []
    return [
        entry.data_file
        for manifest in snapshot.manifests(table.io)
        if manifest.content == ManifestContent.DATA
        for entry in manifest.fetch_manifest_entry(table.io, discard_deleted=True)
    ]


def _pruning_filter(
    table: pyiceberg.table.Table, row_filter: pyiceberg.expressions.BooleanExpression
) -> pyiceberg.expressions.BooleanExpression:
    """`row_filter` without the predicates PyIceberg would prune files wrongly with."""
    from pyiceberg.expressions import AlwaysTrue

    from polars.io.iceberg._utils import (
        filter_for_required_is_null,
        filter_for_truncate_overflow,
    )

    out = filter_for_truncate_overflow(row_filter, table)
    if out is not None:
        out = filter_for_required_is_null(out, table.schema())
    return AlwaysTrue() if out is None else out


def _classify_files(
    table: pyiceberg.table.Table,
    row_filter: pyiceberg.expressions.BooleanExpression,
    *,
    whole_partitions: Collection[Any] = (),
) -> tuple[list[DataFile], set[str]]:
    """
    Classify the live data files of `table` from metadata only.

    Returns the files whose rows all match `row_filter` (of `_exact_filter`), and the
    paths of the files that may hold a match. Files of the default spec whose partition
    is in `whole_partitions` match whole.
    """
    from pyiceberg.expressions.visitors import (
        _InclusiveMetricsEvaluator,
        _StrictMetricsEvaluator,
        expression_evaluator,
        inclusive_projection,
    )
    from pyiceberg.schema import Schema

    schema = table.schema()
    prune = _pruning_filter(table, row_filter)
    # private in PyIceberg 0.12, as its planning uses them
    strict = _StrictMetricsEvaluator(
        schema, _strict_filter(schema, row_filter), case_sensitive=True
    ).eval
    inclusive = _InclusiveMetricsEvaluator(schema, prune, case_sensitive=True).eval
    partition_evaluators: dict[int, Any] = {}

    float_partitions = _float_identity_partitions(table)

    def partition_may_match(data_file: DataFile) -> bool:
        spec_id = data_file.spec_id
        if float_partitions:
            return True
        if spec_id not in partition_evaluators:
            try:
                spec = table.specs()[spec_id]
                partition_schema = Schema(*spec.partition_type(schema).fields)
                projected = inclusive_projection(schema, spec, case_sensitive=True)(
                    prune
                )
                partition_evaluators[spec_id] = expression_evaluator(
                    partition_schema, projected, case_sensitive=True
                )
            except Exception:
                # eg: a partition source column that was dropped
                partition_evaluators[spec_id] = None
        evaluator = partition_evaluators[spec_id]
        return evaluator is None or evaluator(data_file.partition)

    default_spec_id = table.metadata.default_spec_id
    whole, may_match = [], set()
    for data_file in _live_data_files(table):
        if data_file.spec_id == default_spec_id and data_file.partition in (
            whole_partitions
        ):
            whole.append(data_file)
        elif not partition_may_match(data_file) or not inclusive(data_file):
            continue
        elif strict(data_file):
            whole.append(data_file)
        else:
            may_match.add(data_file.file_path)
    return whole, may_match


def _float_identity_partitions(table: pyiceberg.table.Table) -> bool:
    """Whether a spec has an identity partition on a float column.

    PyIceberg reads these partition values as float32: it would prune wrongly.
    """
    from pyiceberg.transforms import IdentityTransform
    from pyiceberg.types import DoubleType, FloatType

    schema = table.schema()
    for spec in table.metadata.partition_specs:
        for field in spec.fields:
            try:
                source = schema.find_field(field.source_id).field_type
            except ValueError:
                continue
            if isinstance(field.transform, IdentityTransform) and isinstance(
                source, (FloatType, DoubleType)
            ):
                return True
    return False


def _strict_filter(
    schema: Schema, row_filter: pyiceberg.expressions.BooleanExpression
) -> pyiceberg.expressions.BooleanExpression:
    """`row_filter` for proving files match whole, without `NotNull` of required fields.

    Binding makes those true, but files written before the field was added hold nulls.
    """
    from pyiceberg.expressions import AlwaysFalse, And, NotNull, Or

    def visit(e: Any) -> Any:
        if isinstance(e, (And, Or)):
            return type(e)(visit(e.left), visit(e.right))
        if isinstance(e, NotNull):
            try:
                field = schema.find_field(str(e.term.name), case_sensitive=True)  # type: ignore[attr-defined]
            except ValueError:
                return e
            if field.required:
                return AlwaysFalse()
        return e

    return visit(row_filter)


def _candidate_tasks(
    table: pyiceberg.table.Table, row_filter: pyiceberg.expressions.BooleanExpression
) -> list[FileScanTask]:
    """The scan tasks of `table`'s files that may hold a row matching `row_filter`.

    Each with all its delete files: planning with a filter also drops the delete files
    whose rows can't match it, which a rewrite of the whole file needs.
    """
    from pyiceberg.manifest import ManifestContent

    tasks = _filtered_tasks(table, row_filter)
    snapshot = table.current_snapshot()
    if not tasks or snapshot is None:
        return tasks
    # delete files are in delete manifests (spec), the summary's totals can be wrong
    manifests = snapshot.manifests(table.io)
    if all(m.content != ManifestContent.DELETES for m in manifests):
        return tasks
    complete = {t.file.file_path: t for t in table.scan().plan_files()}
    return [complete[t.file.file_path] for t in tasks]


def _filtered_tasks(
    table: pyiceberg.table.Table, row_filter: pyiceberg.expressions.BooleanExpression
) -> list[FileScanTask]:
    """`table`'s scan tasks pruned by `row_filter`, where PyIceberg can do that."""
    if _float_identity_partitions(table):
        return list(table.scan().plan_files())
    try:
        return list(
            table.scan(row_filter=_pruning_filter(table, row_filter)).plan_files()
        )
    except TypeError:
        # PyIceberg's residual evaluator compares null partition values (eg: under OR)
        return list(table.scan().plan_files())


def _write_rows(
    table: pyiceberg.table.Table,
    lf: LazyFrame | None,
    mode: Literal["append", "overwrite"],
    storage_options: StorageOptionsDict | None,
    *,
    replace: _ReplaceFiles | None = None,
) -> None:
    """Write the rows of `lf` to `table` in one commit, leaving no files on failure.

    With `replace`, the commit replaces those data files with the written ones; with no
    `lf`, it only removes them.
    """
    from pyiceberg.exceptions import CommitStateUnknownException

    from polars.io.iceberg._sink import IcebergSinkState

    state = None
    try:
        state = IcebergSinkState.new(
            table, mode=mode, storage_options=storage_options, replace_files=replace
        )
        if lf is None:
            state.commit([])
        else:
            state.attach_sink(lf).collect()
    except CommitStateUnknownException as error:
        assert state is not None
        if lf is not None:
            location, _ = state.written_files_dir()
            error.add_note(
                f"the rows may have landed; data files kept under '{location}'"
            )
        raise
    except Exception as error:
        if state is not None and (remaining := state.remove_written_files()):
            error.add_note(f"data files may remain under '{remaining}'")
        raise


def _exact_filter(
    schema: Schema, row_filter: pyiceberg.expressions.BooleanExpression
) -> pyiceberg.expressions.BooleanExpression:
    """
    `row_filter`, matching the rows for which it is true, under Iceberg's evaluation.

    Iceberg evaluates filters in two-valued logic, where a null value matches `!=` and
    `NOT IN`; here a NULL condition keeps a row, as in SQL. NaN orders above every
    number, as in Polars. So the metadata checks, PyIceberg's planning and
    `_filter_predicate` agree on the rows.
    """
    from pyiceberg.expressions import (
        AlwaysFalse,
        AlwaysTrue,
        And,
        EqualTo,
        GreaterThan,
        GreaterThanOrEqual,
        In,
        IsNaN,
        IsNull,
        LessThan,
        LessThanOrEqual,
        LiteralPredicate,
        NotNaN,
        NotNull,
        Or,
        Reference,
        SetPredicate,
        UnboundPredicate,
    )
    from pyiceberg.expressions.visitors import rewrite_not
    from pyiceberg.types import DoubleType, FloatType

    def exact(e: Any) -> Any:
        if isinstance(e, (And, Or)):
            return type(e)(exact(e.left), exact(e.right))
        if isinstance(e, (AlwaysTrue, AlwaysFalse)):
            return e
        if not isinstance(e, UnboundPredicate) or not isinstance(e.term, Reference):
            msg = f"unexpected Iceberg filter {e!r}"
            raise TypeError(msg)
        ref = e.term
        try:
            field = schema.find_field(ref.name, case_sensitive=True)
        except ValueError:
            # a column the write adds (`schema_mode="merge"`): null in every row
            return AlwaysTrue() if isinstance(e, IsNull) else AlwaysFalse()
        if _under_optional_struct(schema, field.field_id):
            msg = (
                "an Iceberg overwrite filter on a column nested within an optional "
                f"struct is not supported: '{ref.name}'"
            )
            raise NotImplementedError(msg)
        if isinstance(e, (IsNull, NotNull)):
            return e
        if isinstance(field.field_type, FloatType):
            # as the column's float32 values, which its statistics hold
            if isinstance(e, LiteralPredicate):
                e = type(e)(ref, _float32(e.literal))  # type: ignore[call-arg]
            elif isinstance(e, SetPredicate):
                e = type(e)(ref, {_float32(lit) for lit in e.literals})  # type: ignore[call-arg]
        # a required field binds `NotNull` to true, also where older files lack it
        guarded = e if field.required else And(e, NotNull(ref))  # type: ignore[call-arg]
        if isinstance(field.field_type, (FloatType, DoubleType)):
            if isinstance(e, (GreaterThan, GreaterThanOrEqual)):
                return Or(guarded, IsNaN(ref))  # type: ignore[call-arg]
            # statistics without NaN counts must not prove a NaN row matches
            if isinstance(e, (EqualTo, In, LessThan, LessThanOrEqual)):
                return And(guarded, NotNaN(ref))  # type: ignore[call-arg]
        return guarded

    return exact(rewrite_not(row_filter))


def _float32(literal: Any) -> Any:
    """A literal's value rounded to float32, or the literal if not finite there."""
    import math
    import struct

    value = float(literal.value)
    if not math.isfinite(value):
        return literal
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return literal


def _under_optional_struct(schema: Schema, field_id: int) -> bool:
    """Whether a struct the field is nested in is optional (null for the field)."""
    accessor = schema.accessor_for_field(field_id)
    field = schema.fields[accessor.position]
    while accessor.inner is not None:
        if not field.required:
            return True
        accessor = accessor.inner
        field = field.field_type.fields[accessor.position]  # type: ignore[attr-defined]
    return False


def _filter_predicate(
    schema: Schema, row_filter: pyiceberg.expressions.BooleanExpression
) -> Expr:
    """The rows matching `row_filter` (of `_exact_filter`), as a Polars predicate."""
    from functools import reduce
    from operator import or_

    from pyiceberg.expressions import (
        AlwaysFalse,
        AlwaysTrue,
        And,
        BoundEqualTo,
        BoundGreaterThan,
        BoundGreaterThanOrEqual,
        BoundIn,
        BoundIsNaN,
        BoundIsNull,
        BoundLessThan,
        BoundLessThanOrEqual,
        BoundNotEqualTo,
        BoundNotIn,
        BoundNotNaN,
        BoundNotNull,
        BoundNotStartsWith,
        BoundStartsWith,
        IsNull,
        NotNull,
        Or,
    )
    from pyiceberg.types import DoubleType, FloatType

    import polars as pl
    from polars.io.iceberg._sink import _iceberg_source_expr

    def column(name: str) -> Expr:
        field = schema.find_field(name, case_sensitive=True)
        return _iceberg_source_expr(schema, field.field_id)

    def predicate(e: Any) -> Expr:
        if isinstance(e, And):
            return predicate(e.left) & predicate(e.right)
        if isinstance(e, Or):
            return predicate(e.left) | predicate(e.right)
        if isinstance(e, (AlwaysTrue, AlwaysFalse)):
            return pl.lit(isinstance(e, AlwaysTrue))
        # not bound: binding folds `IsNull` of a required field to false
        if isinstance(e, (IsNull, NotNull)):
            c = column(str(e.term.name))  # type: ignore[attr-defined]
            return c.is_null() if isinstance(e, IsNull) else c.is_not_null()
        bound = e.bind(schema, case_sensitive=True)
        if isinstance(bound, (AlwaysTrue, AlwaysFalse)):
            return pl.lit(isinstance(bound, AlwaysTrue))
        field = bound.term.ref().field
        c = _iceberg_source_expr(schema, field.field_id)
        if isinstance(bound, (BoundIsNull, BoundNotNull, BoundIsNaN, BoundNotNaN)):
            return {
                BoundIsNull: c.is_null,
                BoundNotNull: c.is_not_null,
                BoundIsNaN: c.is_nan,
                BoundNotNaN: c.is_not_nan,
            }[type(bound)]()
        if isinstance(bound, (BoundIn, BoundNotIn)):
            values = [_value(field.field_type, lit.value) for lit in bound.literals]
            dtype = values[0][1]
            if isinstance(field.field_type, (FloatType, DoubleType)):
                # `==` matches 0.0 and -0.0
                found = reduce(or_, (c == pl.lit(v) for v, _ in values))
            else:
                found = c.is_in(
                    pl.Series([v for v, _ in values], dtype=dtype).implode()
                )
            return found if isinstance(bound, BoundIn) else ~found
        v = pl.lit(*_value(field.field_type, bound.literal.value))
        if isinstance(bound, (BoundStartsWith, BoundNotStartsWith)):
            found = c.str.starts_with(v)
            return found if isinstance(bound, BoundStartsWith) else ~found
        compare = {
            BoundEqualTo: c.__eq__,
            BoundNotEqualTo: c.__ne__,
            BoundLessThan: c.__lt__,
            BoundLessThanOrEqual: c.__le__,
            BoundGreaterThan: c.__gt__,
            BoundGreaterThanOrEqual: c.__ge__,
        }.get(type(bound))
        if compare is None:
            msg = f"unsupported Iceberg filter {e!r}"
            raise NotImplementedError(msg)
        return compare(v)

    return predicate(row_filter)


def _value(field_type: Any, value: Any) -> tuple[Any, PolarsDataType | None]:
    """A bound Iceberg literal's value as a Python value, and its Polars type."""
    from pyiceberg.types import (
        DateType,
        DecimalType,
        TimestampType,
        TimestamptzType,
        TimeType,
    )
    from pyiceberg.utils.datetime import (
        days_to_date,
        micros_to_time,
        micros_to_timestamp,
        micros_to_timestamptz,
    )

    import polars as pl

    if isinstance(field_type, DateType):
        return days_to_date(value), pl.Date()
    if isinstance(field_type, TimeType):
        return micros_to_time(value), pl.Time()
    if isinstance(field_type, TimestamptzType):
        return micros_to_timestamptz(value), pl.Datetime("us", "UTC")
    if isinstance(field_type, TimestampType):
        return micros_to_timestamp(value), pl.Datetime("us")
    if isinstance(field_type, DecimalType):
        return value, pl.Decimal(field_type.precision, field_type.scale)
    return value, None
