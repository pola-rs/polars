from __future__ import annotations

import abc
import ast
import contextlib
import re
import uuid
from _ast import GtE, Lt, LtE
from ast import (
    Attribute,
    BinOp,
    BitAnd,
    BitOr,
    Call,
    Compare,
    Constant,
    Eq,
    Gt,
    Invert,
    List,
    Name,
    NotEq,
    UnaryOp,
    USub,
)
from dataclasses import dataclass
from functools import cache, singledispatch
from typing import TYPE_CHECKING, Any

import polars._reexport as pl
from polars._utils.convert import to_py_date, to_py_datetime
from polars._utils.logging import eprint
from polars._utils.wrap import wrap_s
from polars.exceptions import ComputeError
from polars.io._utils import null_count_dtype

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from datetime import date, datetime

    import pyiceberg
    import pyiceberg.schema
    from pyiceberg.manifest import DataFile
    from pyiceberg.table import Table
    from pyiceberg.types import IcebergType, NestedField

    from polars import DataFrame
else:
    from polars._dependencies import pyiceberg


def _to_py_datetime_exact(value: int, time_unit: str, *args: Any) -> datetime:
    # Python datetimes have microsecond precision: truncating a nanosecond value
    # would change the predicate (e.g. narrow `<`), so it is not converted.
    if time_unit == "ns" and value % 1000 != 0:
        msg = f"nanosecond literal {value} cannot be represented exactly"
        raise ValueError(msg)
    return to_py_datetime(value, time_unit, *args)  # type: ignore[arg-type]


_temporal_conversions: dict[str, Callable[..., datetime | date]] = {
    "to_py_date": to_py_date,
    "to_py_datetime": _to_py_datetime_exact,
}

ICEBERG_TIME_TO_NS: int = 1000


def _new_pyiceberg_scan(
    tbl: Table,
    *,
    snapshot_id: int | None,
    from_snapshot_id_exclusive: int | None,
    to_snapshot_id_inclusive: int | None,
    selected_fields: tuple[str, ...] = ("*",),
    limit: int | None = None,
) -> Any:
    from polars.io.iceberg._cache import with_metadata_file_cache

    scan: Any

    if from_snapshot_id_exclusive is None and to_snapshot_id_inclusive is None:
        scan = tbl.scan(
            snapshot_id=snapshot_id,
            selected_fields=selected_fields,
            limit=limit,
        )
    else:
        scan = tbl.incremental_append_scan(
            from_snapshot_id_exclusive=from_snapshot_id_exclusive,
            to_snapshot_id_inclusive=to_snapshot_id_inclusive,
            selected_fields=selected_fields,
            limit=limit,
        )

    return with_metadata_file_cache(scan)


# PyIceberg on Windows uses `file://C:/` rather than `file:///C:/`.
def _normalize_windows_iceberg_file_uri(path: str) -> str:
    # `file://C:/x` has the drive letter as its authority. Other authorities (UNC hosts,
    # `localhost`) are kept.
    if re.match(r"file://[A-Za-z]:", path):
        return f"file:///{path.removeprefix('file://')}"

    return path


def _scan_pyarrow_dataset_impl(
    tbl: Table,
    with_columns: list[str] | None = None,
    iceberg_table_filter: Any | None = None,
    n_rows: int | None = None,
    snapshot_id: int | None = None,
    from_snapshot_id_exclusive: int | None = None,
    to_snapshot_id_inclusive: int | None = None,
    **kwargs: Any,  # noqa: ARG001
) -> tuple[Iterable[DataFrame], bool]:
    """
    Take the projected columns and materialize an arrow table.

    Parameters
    ----------
    tbl
        pyarrow dataset
    with_columns
        Columns that are projected
    iceberg_table_filter
        PyIceberg filter expression
    n_rows:
        Materialize only n rows from the arrow dataset.
    snapshot_id:
        The snapshot ID to scan from.
    from_snapshot_id_exclusive
        The exclusive start of an incremental append scan.
    to_snapshot_id_inclusive
        The inclusive end of an incremental append scan.
    batch_size
        The maximum row count for scanned pyarrow record batches.
    kwargs:
        For backward compatibility

    Returns
    -------
    tuple[Iterator[DataFrame], bool]
    A generator over the DataFrames and a boolean indicating if the
    predicates could be parsed.
    This boolean is always `False` as there might be some predicates
    that could not be converted
    to pyarrow and need to be applied as post-predicate.
    """
    scan = _new_pyiceberg_scan(
        tbl,
        snapshot_id=snapshot_id,
        from_snapshot_id_exclusive=from_snapshot_id_exclusive,
        to_snapshot_id_inclusive=to_snapshot_id_inclusive,
        limit=n_rows,
    )

    if with_columns is not None:
        if not with_columns:
            assert iceberg_table_filter is None

            def gen() -> Iterable[pl.DataFrame]:
                if hasattr(scan, "count"):
                    remaining = scan.count()
                else:
                    # E.g. incremental append scans.
                    remaining = sum(
                        batch.num_rows for batch in scan.to_arrow_batch_reader()
                    )

                if n_rows is not None:
                    remaining = min(remaining, n_rows)

                yield pl.DataFrame(height=remaining)

            return (gen(), False)

        scan = scan.select(*with_columns)

    if iceberg_table_filter is not None:
        scan = scan.filter(iceberg_table_filter)

    batches = scan.to_arrow_batch_reader()

    return ((pl.DataFrame(batch) for batch in batches), False)


def _ensure_boolean_expression(result: Any) -> Any:
    """Convert scalar booleans and bare fields into PyIceberg boolean expressions."""
    if result is True:
        return pyiceberg.expressions.AlwaysTrue()
    if result is False:
        return pyiceberg.expressions.AlwaysFalse()
    if isinstance(result, list) and len(result) == 1:
        return pyiceberg.expressions.EqualTo(result[0], True)  # type: ignore[misc, call-arg, arg-type]
    return result


def try_convert_pyarrow_predicate(pyarrow_predicate: str) -> Any | None:
    try:
        expr_ast = _to_ast(pyarrow_predicate)
    except Exception:
        return None

    # Polars hands us a conjunction of independently converted minterms, and
    # PyIceberg has no equivalent for some of them (arithmetic, for one). Keep
    # the conjuncts that do convert: dropping one only widens the filter, and
    # the engine re-applies the full predicate after the scan.
    converted: list[Any] = []

    for conjunct in _split_conjuncts(expr_ast):
        with contextlib.suppress(Exception):
            converted.append(_ensure_boolean_expression(_convert_predicate(conjunct)))

    if not converted:
        return None

    result = converted[0]

    for expr in converted[1:]:
        result = pyiceberg.expressions.And(result, expr)

    return result


def _split_conjuncts(a: ast.expr) -> Iterable[ast.expr]:
    """Yield the operands of a (possibly nested) top-level `&`."""
    if isinstance(a, BinOp) and isinstance(a.op, BitAnd):
        yield from _split_conjuncts(a.left)
        yield from _split_conjuncts(a.right)
    else:
        yield a


def _to_ast(expr: str) -> ast.expr:
    """
    Converts a Python string to an AST.

    This will take the Python Arrow expression (as a string), and it will
    be converted into a Python AST that can be traversed to convert it to a PyIceberg
    expression.

    The reason to convert it to an AST is because the PyArrow expression
    itself doesn't have any methods/properties to traverse the expression.
    We need this to convert it into a PyIceberg expression.

    Parameters
    ----------
    expr
        The string expression

    Returns
    -------
    The AST representing the Arrow expression
    """
    return ast.parse(expr, mode="eval").body


@singledispatch
def _convert_predicate(a: Any) -> Any:
    """Walks the AST to convert the PyArrow expression to a PyIceberg expression."""
    msg = f"Unexpected symbol: {a}"
    raise ValueError(msg)


@_convert_predicate.register(Constant)
def _(a: Constant) -> Any:
    return a.value


@_convert_predicate.register(Name)
def _(a: Name) -> Any:
    if a.id == "NaN":
        msg = "NaN literal is not supported in this predicate position"
        raise ValueError(msg)
    return a.id


@_convert_predicate.register(UnaryOp)
def _(a: UnaryOp) -> Any:
    if isinstance(a.op, Invert):
        operand = _ensure_boolean_expression(_convert_predicate(a.operand))
        return pyiceberg.expressions.Not(operand)
    elif (
        isinstance(a.op, USub)
        and isinstance(a.operand, Constant)
        and isinstance(a.operand.value, (int, float))
        and not isinstance(a.operand.value, bool)
    ):
        # Negative literals, e.g. `-5` or `to_py_datetime(-123, 'us')`.
        return -a.operand.value
    else:
        msg = f"Unexpected UnaryOp: {a}"
        raise TypeError(msg)


@_convert_predicate.register(Call)
def _(a: Call) -> Any:
    args = [_convert_predicate(arg) for arg in a.args]
    f = _convert_predicate(a.func)
    if f == "field":
        return args
    elif f == "scalar":
        return args[0]
    elif f in _temporal_conversions:
        # convert from polars-native i64 to ISO8601 string
        return _temporal_conversions[f](*args).isoformat()
    elif f == "starts_with":
        pattern = _convert_predicate(a.keywords[0].value)
        return pyiceberg.expressions.StartsWith(args[0][0], pattern)  # type: ignore[misc, call-arg]
    else:
        ref = _convert_predicate(a.func.value)[0]  # type: ignore[attr-defined]
        if f == "isin":
            return pyiceberg.expressions.In(ref, args[0])  # type: ignore[misc, call-arg]
        elif f == "is_null":
            return pyiceberg.expressions.IsNull(ref)  # type: ignore[misc]
        elif f == "is_nan":
            return pyiceberg.expressions.IsNaN(ref)  # type: ignore[misc]

    msg = f"Unknown call: {f!r}"
    raise ValueError(msg)


@_convert_predicate.register(Attribute)
def _(a: Attribute) -> Any:
    return a.attr


@_convert_predicate.register(BinOp)
def _(a: BinOp) -> Any:
    lhs = _ensure_boolean_expression(_convert_predicate(a.left))
    rhs = _ensure_boolean_expression(_convert_predicate(a.right))

    op = a.op
    if isinstance(op, BitAnd):
        return pyiceberg.expressions.And(lhs, rhs)
    if isinstance(op, BitOr):
        return pyiceberg.expressions.Or(lhs, rhs)
    else:
        msg = f"Unknown: {lhs} {op} {rhs}"
        raise TypeError(msg)


@_convert_predicate.register(Compare)
def _(a: Compare) -> Any:
    op = a.ops[0]
    lhs = _convert_predicate(a.left)[0]
    rhs_ast = a.comparators[0]

    if isinstance(rhs_ast, Name) and rhs_ast.id == "NaN":
        if isinstance(op, Eq):
            return pyiceberg.expressions.IsNaN(lhs)  # type: ignore[misc]
        if isinstance(op, NotEq):
            return pyiceberg.expressions.NotNaN(lhs)  # type: ignore[misc]

    rhs = _convert_predicate(rhs_ast)

    if isinstance(op, Gt):
        return pyiceberg.expressions.GreaterThan(lhs, rhs)  # type: ignore[misc, call-arg]
    if isinstance(op, GtE):
        return pyiceberg.expressions.GreaterThanOrEqual(lhs, rhs)  # type: ignore[misc, call-arg]
    if isinstance(op, Eq):
        return pyiceberg.expressions.EqualTo(lhs, rhs)  # type: ignore[misc, call-arg]
    if isinstance(op, NotEq):
        return pyiceberg.expressions.NotEqualTo(lhs, rhs)  # type: ignore[misc, call-arg]
    if isinstance(op, Lt):
        return pyiceberg.expressions.LessThan(lhs, rhs)  # type: ignore[misc, call-arg]
    if isinstance(op, LtE):
        return pyiceberg.expressions.LessThanOrEqual(lhs, rhs)  # type: ignore[misc, call-arg]
    else:
        msg = f"Unknown comparison: {op}"
        raise TypeError(msg)


@_convert_predicate.register(List)
def _(a: List) -> Any:
    return [_convert_predicate(e) for e in a.elts]


def extract_field_initial_default(field: NestedField) -> pl.Series | None:
    from pyiceberg.types import (
        UUIDType,
    )

    if field.initial_default is None:
        return None

    value = field.initial_default

    if isinstance(field.field_type, UUIDType):
        assert isinstance(value, uuid.UUID)
        value = value.bytes

    return pl.Series([value], dtype=pl_dtype_from_iceberg_field(field))


def pl_dtype_from_iceberg_field(field: NestedField) -> pl.DataType:
    from pyiceberg.io.pyarrow import schema_to_pyarrow

    _, field_polars_dtype = pl.Schema(
        schema_to_pyarrow(pyiceberg.schema.Schema(field))
    ).popitem()

    return field_polars_dtype


def load_puffin_deletion_file(puffin_bytes: bytes) -> dict[str, pl.Series]:
    import pyiceberg.table.puffin

    import polars as pl

    positions = pyiceberg.table.puffin.PuffinFile(puffin_bytes).to_vector()

    return {k: pl.Series(v).reinterpret(dtype=pl.UInt64) for k, v in positions.items()}


def filter_for_pyiceberg_reader(
    expr: pyiceberg.expressions.BooleanExpression,
    schema: pyiceberg.schema.Schema,
) -> pyiceberg.expressions.BooleanExpression | None:
    """
    The part of a filter that PyIceberg can apply when reading rows, or None.

    PyIceberg drops rows that do not match the filter, so it must select a superset
    of the rows of Polars' predicate. On float columns, predicates that NaN
    satisfies in Polars (where NaN is greater than all values) are left out: NaN
    does not compare in Iceberg, and Parquet row group statistics used by PyArrow
    exclude NaN.
    """
    import math

    from pyiceberg.expressions import (
        And,
        EqualTo,
        In,
        IsNull,
        LessThan,
        LessThanOrEqual,
        Not,
        NotNaN,
        NotNull,
        Or,
        Reference,
        UnboundPredicate,
    )
    from pyiceberg.expressions.literals import Literal
    from pyiceberg.expressions.visitors import rewrite_not
    from pyiceberg.types import DoubleType, FloatType

    # Not satisfied by NaN, given non-NaN literals.
    nan_excluding = (
        LessThan,
        LessThanOrEqual,
        EqualTo,
        In,
        IsNull,
        NotNull,
        NotNaN,
    )

    def is_nan_literal(v: Any) -> bool:
        v = v.value if isinstance(v, Literal) else v
        return isinstance(v, float) and math.isnan(v)

    def is_zero_literal(v: Any) -> bool:
        v = v.value if isinstance(v, Literal) else v
        return v == 0

    def visit(
        e: pyiceberg.expressions.BooleanExpression,
    ) -> pyiceberg.expressions.BooleanExpression | None:
        if isinstance(e, And):
            left, right = visit(e.left), visit(e.right)
            if left is None or right is None:
                return left if right is None else right
            return And(left, right)
        if isinstance(e, Or):
            left, right = visit(e.left), visit(e.right)
            return None if left is None or right is None else Or(left, right)
        if isinstance(e, Not):
            # Not left by `rewrite_not`.
            return None
        if isinstance(e, UnboundPredicate):
            if not isinstance(e.term, Reference):
                return None
            try:
                field = schema.find_field(e.term.name, case_sensitive=True)
            except ValueError:
                return None
            if not field.field_type.is_primitive:
                # PyIceberg cannot project list / map columns for row filters.
                return None
            if isinstance(field.field_type, (FloatType, DoubleType)):
                literals = [
                    *getattr(e, "literals", ()),
                    *([e.literal] if hasattr(e, "literal") else []),
                ]
                if (
                    not isinstance(e, nan_excluding)
                    or any(is_nan_literal(v) for v in literals)
                    # PyArrow `is_in` and Parquet row group statistics distinguish
                    # -0.0 from 0.0.
                    or (isinstance(e, In) and any(is_zero_literal(v) for v in literals))
                ):
                    return None
        return e

    return visit(rewrite_not(expr))


def filter_for_scan_schema(
    expr: pyiceberg.expressions.BooleanExpression,
    scan_schema: pyiceberg.schema.Schema,
    current_schema: pyiceberg.schema.Schema,
) -> pyiceberg.expressions.BooleanExpression | None:
    """
    The filter to plan a scan of an older schema with, or None.

    PyIceberg binds scan filters to the current schema, also when time travelling;
    after schema changes, a column name may refer to another field (or none). The
    filter is kept only if it binds to the same fields in both schemas.
    """
    from pyiceberg.expressions.visitors import bind

    try:
        if bind(scan_schema, expr, case_sensitive=True) == bind(
            current_schema, expr, case_sensitive=True
        ):
            return expr
    except Exception:
        pass

    return None


def filter_with_nan_ordering(
    expr: pyiceberg.expressions.BooleanExpression,
    schema: pyiceberg.schema.Schema,
) -> pyiceberg.expressions.BooleanExpression:
    """
    Adapt a filter to Polars' NaN ordering.

    In Polars, NaN is greater than all other values, whereas Iceberg comparisons
    are false for NaN. `>` / `>=` on a float column therefore also select NaN.
    """
    from pyiceberg.expressions import (
        And,
        GreaterThan,
        GreaterThanOrEqual,
        IsNaN,
        Or,
        Reference,
    )
    from pyiceberg.expressions.visitors import rewrite_not
    from pyiceberg.types import DoubleType, FloatType

    def visit(
        e: pyiceberg.expressions.BooleanExpression,
    ) -> pyiceberg.expressions.BooleanExpression:
        if isinstance(e, (And, Or)):
            return type(e)(visit(e.left), visit(e.right))
        if isinstance(e, (GreaterThan, GreaterThanOrEqual)) and isinstance(
            e.term, Reference
        ):
            try:
                field = schema.find_field(e.term.name, case_sensitive=True)
            except ValueError:
                return e
            if isinstance(field.field_type, (FloatType, DoubleType)):
                return Or(e, IsNaN(e.term))  # type: ignore[call-arg]
        return e

    # Negations are pushed down first, as `~(x < v)` also selects NaN.
    return visit(rewrite_not(expr))


class IdentityTransformedPartitionValuesBuilder:
    def __init__(
        self,
        table: Table,
        projected_schema: pyiceberg.schema.Schema,
    ) -> None:
        import pyiceberg.schema
        from pyiceberg.io.pyarrow import schema_to_pyarrow
        from pyiceberg.transforms import IdentityTransform
        from pyiceberg.types import (
            DecimalType,
            DoubleType,
            FloatType,
            IntegerType,
            LongType,
        )

        projected_ids: set[int] = projected_schema.field_ids

        # {source_field_id: [values] | error_message}
        self.partition_values: dict[int, list[Any] | str] = {}
        # Logical types will have length-2 list [<constructor type>, <cast type>].
        # E.g. for Datetime it will be [Int64, Datetime]
        self.partition_values_dtypes: dict[int, pl.DataType] = {}

        # {spec_id: [partition_value_index, source_field_id]}
        self.partition_spec_id_to_identity_transforms: dict[
            int, list[tuple[int, int]]
        ] = {}

        partition_specs = table.specs()

        for spec_id, spec in partition_specs.items():
            out = []

            for field_index, field in enumerate(spec.fields):
                if field.source_id in projected_ids and isinstance(
                    field.transform, IdentityTransform
                ):
                    out.append((field_index, field.source_id))
                    self.partition_values[field.source_id] = []

            self.partition_spec_id_to_identity_transforms[spec_id] = out

        for field_id in self.partition_values:
            projected_field = projected_schema.find_field(field_id)
            projected_type = projected_field.field_type

            _, output_dtype = pl.Schema(
                schema_to_pyarrow(pyiceberg.schema.Schema(projected_field))
            ).popitem()

            self.partition_values_dtypes[field_id] = output_dtype

            if not projected_type.is_primitive or output_dtype.is_nested():
                self.partition_values[field_id] = (
                    f"non-primitive type: {projected_type = } {output_dtype = }"
                )

            for schema in table.schemas().values():
                try:
                    type_this_schema = schema.find_field(field_id).field_type
                except ValueError:
                    continue

                # Type promotions, in either direction: a scan of an older snapshot
                # projects the type from before later promotions.
                if not (
                    projected_type == type_this_schema
                    or (
                        isinstance(projected_type, (LongType, IntegerType))
                        and isinstance(type_this_schema, (LongType, IntegerType))
                    )
                    or (
                        isinstance(projected_type, (DoubleType, FloatType))
                        and isinstance(type_this_schema, (DoubleType, FloatType))
                    )
                    or (
                        # Precision changes; unscaled values are unchanged.
                        isinstance(projected_type, DecimalType)
                        and isinstance(type_this_schema, DecimalType)
                        and projected_type.scale == type_this_schema.scale
                    )
                ):
                    self.partition_values[field_id] = (
                        f"unsupported type change: from: {type_this_schema}, "
                        f"to: {projected_type}"
                    )

    def push_partition_values(
        self,
        *,
        current_index: int,
        partition_spec_id: int,
        partition_values: pyiceberg.typedef.Record,
    ) -> None:
        try:
            identity_transforms = self.partition_spec_id_to_identity_transforms[
                partition_spec_id
            ]
        except KeyError:
            self.partition_values = dict.fromkeys(
                self.partition_values,
                f"partition spec ID not found: {partition_spec_id}",
            )
            return

        for i, source_field_id in identity_transforms:
            partition_value = partition_values[i]

            if isinstance(values := self.partition_values[source_field_id], list):
                # extend() - there can be gaps from partitions being
                # added/removed/re-added
                values.extend(None for _ in range(current_index - len(values)))
                values.append(partition_value)

    def finish(self) -> dict[int, pl.Series | str]:
        from polars.datatypes import (
            Binary,
            Date,
            Datetime,
            Duration,
            Int32,
            Int64,
            Time,
        )

        out: dict[int, pl.Series | str] = {}

        for field_id, v in self.partition_values.items():
            if isinstance(v, str):
                out[field_id] = v
            else:
                try:
                    output_dtype = self.partition_values_dtypes[field_id]

                    constructor_dtype = (
                        Int64
                        if isinstance(output_dtype, (Datetime, Duration, Time))
                        else Int32
                        if isinstance(output_dtype, Date)
                        else output_dtype
                    )

                    if constructor_dtype == Binary:
                        # E.g. UUIDs.
                        v = [x.bytes if isinstance(x, uuid.UUID) else x for x in v]

                    s = pl.Series(v, dtype=constructor_dtype)

                    assert not s.dtype.is_nested()

                    if isinstance(output_dtype, Time):
                        # Physical from PyIceberg is in microseconds, physical
                        # used by polars is in nanoseconds.
                        s = s * ICEBERG_TIME_TO_NS

                    s = s.cast(output_dtype)

                    out[field_id] = s

                except Exception as e:
                    out[field_id] = f"failed to load partition values: {e}"

        return out


class IcebergStatisticsLoader:
    def __init__(
        self,
        table: Table,
        statistics_schema: pyiceberg.schema.Schema,
        *,
        best_effort_columns: Sequence[str] = (),
    ) -> None:
        import polars._utils.logging

        verbose = polars._utils.logging.verbose()

        self.file_column_statistics: dict[int, IcebergColumnStatisticsLoader] = {}
        self.load_as_empty_statistics: list[str] = []
        self.file_lengths: list[int] = []
        self.statistics_schema = statistics_schema
        # Columns whose statistics are loaded as nulls when they cannot be loaded.
        self.best_effort_columns = set(best_effort_columns)

        for field in statistics_schema.fields:
            field_all_types = set()

            for schema in table.schemas().values():
                with contextlib.suppress(ValueError):
                    field_all_types.add(schema.find_field(field.field_id).field_type)

            field_polars_dtype = pl_dtype_from_iceberg_field(field)

            load_from_bytes_impl = LoadFromBytesImpl.init_for_field_type(
                field.field_type,
                field_all_types,
                field_polars_dtype,
            )

            if verbose:
                _load_from_bytes_impl = (
                    type(load_from_bytes_impl).__name__
                    if load_from_bytes_impl is not None
                    else "None"
                )

                eprint(
                    "IcebergStatisticsLoader: "
                    f"{field.name = }, "
                    f"{field.field_id = }, "
                    f"{field.field_type = }, "
                    f"{field_all_types = }, "
                    f"{field_polars_dtype = }, "
                    f"{_load_from_bytes_impl = }"
                )

            self.file_column_statistics[field.field_id] = IcebergColumnStatisticsLoader(
                field_id=field.field_id,
                column_name=field.name,
                column_dtype=field_polars_dtype,
                load_from_bytes_impl=load_from_bytes_impl,
                min_values=[],
                max_values=[],
                null_count=[],
            )

    def push_file_statistics(self, file: DataFile) -> None:
        self.file_lengths.append(file.record_count)

        for stats in self.file_column_statistics.values():
            stats.push_file_statistics(file)

    def finish(
        self,
        expected_height: int,
        identity_transformed_values: dict[int, pl.Series | str],
    ) -> pl.DataFrame:
        import polars as pl
        import polars._utils.logging

        verbose = polars._utils.logging.verbose()

        out: list[pl.DataFrame] = [
            # A record count that does not fit (or is unknown, -1 in some format v1
            # tables) is null, which only disables skipping that file.
            pl.Series(
                "len",
                [x if 0 <= x < 2**32 else None for x in self.file_lengths],
                dtype=pl.UInt32,
            ).to_frame()
        ]

        for field_id, stat_builder in self.file_column_statistics.items():
            try:
                column_stats_df = stat_builder.finish(
                    expected_height, identity_transformed_values.get(field_id)
                )
            except Exception as e:
                if stat_builder.column_name not in self.best_effort_columns:
                    raise

                if verbose:
                    eprint(
                        "IcebergStatisticsLoader: statistics load failed for column "
                        f"{stat_builder.column_name!r}: {e!r}"
                    )

                column_stats_df = stat_builder.null_statistics(expected_height)

            out.append(column_stats_df)

        return pl.concat(out, how="horizontal")


@dataclass
class IcebergColumnStatisticsLoader:
    column_name: str
    column_dtype: pl.DataType
    field_id: int
    load_from_bytes_impl: LoadFromBytesImpl | None
    null_count: list[int | None]
    min_values: list[bytes | None]
    max_values: list[bytes | None]

    def push_file_statistics(self, file: DataFile) -> None:
        self.null_count.append(file.null_value_counts.get(self.field_id))

        if self.load_from_bytes_impl is not None:
            self.min_values.append(file.lower_bounds.get(self.field_id))
            self.max_values.append(file.upper_bounds.get(self.field_id))

    def null_statistics(self, height: int) -> pl.DataFrame:
        import polars as pl

        c = self.column_name

        return pl.DataFrame(
            schema={
                f"{c}_nc": null_count_dtype(self.column_dtype),
                f"{c}_min": self.column_dtype,
                f"{c}_max": self.column_dtype,
            }
        ).clear(height)

    def finish(
        self,
        expected_height: int,
        identity_transformed_values: pl.Series | str | None,
    ) -> pl.DataFrame:
        import polars as pl

        if isinstance(identity_transformed_values, str):
            msg = f"statistics load failure for filter column: {identity_transformed_values}"
            raise ComputeError(msg)

        c = self.column_name
        assert len(self.null_count) == expected_height

        out = pl.Series(
            f"{c}_nc", self.null_count, dtype=null_count_dtype(self.column_dtype)
        ).to_frame()

        if self.load_from_bytes_impl is None:
            # Can be shorter if the identity partition field was removed.
            s = (
                identity_transformed_values.extend_constant(
                    None, expected_height - identity_transformed_values.len()
                )
                if identity_transformed_values is not None
                else pl.repeat(
                    None, expected_height, dtype=self.column_dtype, eager=True
                )
            )

            return out.with_columns(s.alias(f"{c}_min"), s.alias(f"{c}_max"))

        assert len(self.min_values) == expected_height
        assert len(self.max_values) == expected_height

        if self.column_dtype.is_nested():
            raise NotImplementedError

        min_values = self.load_from_bytes_impl.load_from_bytes(self.min_values)
        max_values = self.load_from_bytes_impl.load_from_bytes(self.max_values)

        if identity_transformed_values is not None:
            assert identity_transformed_values.dtype == self.column_dtype

            identity_transformed_values = identity_transformed_values.extend_constant(
                None, expected_height - identity_transformed_values.len()
            )

            min_values = identity_transformed_values.fill_null(min_values)
            max_values = identity_transformed_values.fill_null(max_values)

        return out.with_columns(
            min_values.alias(f"{c}_min"), max_values.alias(f"{c}_max")
        )


# Lazy init instead of global const as PyIceberg is an optional dependency
@cache
def _bytes_loader_lookup() -> dict[
    type[IcebergType],
    tuple[type[LoadFromBytesImpl], type[IcebergType] | Sequence[type[IcebergType]]],
]:
    from pyiceberg.types import (
        BinaryType,
        BooleanType,
        DateType,
        DecimalType,
        FixedType,
        IntegerType,
        LongType,
        StringType,
        TimestampType,
        TimestamptzType,
        TimeType,
    )

    # TODO: Float statistics
    return {
        BooleanType: (LoadBooleanFromBytes, BooleanType),
        DateType: (LoadDateFromBytes, DateType),
        TimeType: (LoadTimeFromBytes, TimeType),
        TimestampType: (LoadTimestampFromBytes, TimestampType),
        TimestamptzType: (LoadTimestamptzFromBytes, TimestamptzType),
        IntegerType: (LoadInt32FromBytes, IntegerType),
        LongType: (LoadInt64FromBytes, (LongType, IntegerType)),
        StringType: (LoadStringFromBytes, StringType),
        BinaryType: (LoadBinaryFromBytes, BinaryType),
        DecimalType: (LoadDecimalFromBytes, DecimalType),
        FixedType: (LoadFixedFromBytes, FixedType),
    }


class LoadFromBytesImpl(abc.ABC):
    def __init__(self, polars_dtype: pl.DataType) -> None:
        self.polars_dtype = polars_dtype

    @staticmethod
    def init_for_field_type(
        current_field_type: IcebergType,
        # All types that this field ID has been set to across schema changes.
        all_field_types: set[IcebergType],
        field_polars_dtype: pl.DataType,
    ) -> LoadFromBytesImpl | None:
        if (v := _bytes_loader_lookup().get(type(current_field_type))) is None:
            return None

        loader_impl, allowed_field_types = v

        return (
            loader_impl(field_polars_dtype)
            if all(isinstance(x, allowed_field_types) for x in all_field_types)  # type: ignore[arg-type]
            else None
        )

    @abc.abstractmethod
    def load_from_bytes(self, byte_values: list[bytes | None]) -> pl.Series:
        """`bytes_values` should be of binary type."""


class LoadBinaryFromBytes(LoadFromBytesImpl):
    def load_from_bytes(self, byte_values: list[bytes | None]) -> pl.Series:
        import polars as pl

        return pl.Series(byte_values, dtype=pl.Binary)


class LoadDateFromBytes(LoadFromBytesImpl):
    def load_from_bytes(self, byte_values: list[bytes | None]) -> pl.Series:
        import polars as pl

        return (
            pl.Series(byte_values, dtype=pl.Binary)
            .bin.reinterpret(dtype=pl.Int32, endianness="little")
            .cast(pl.Date)
        )


class LoadTimeFromBytes(LoadFromBytesImpl):
    def load_from_bytes(self, byte_values: list[bytes | None]) -> pl.Series:
        import polars as pl

        return (
            pl.Series(byte_values, dtype=pl.Binary).bin.reinterpret(
                dtype=pl.Int64, endianness="little"
            )
            * ICEBERG_TIME_TO_NS
        ).cast(pl.Time)


class LoadTimestampFromBytes(LoadFromBytesImpl):
    def load_from_bytes(self, byte_values: list[bytes | None]) -> pl.Series:
        import polars as pl

        return (
            pl.Series(byte_values, dtype=pl.Binary)
            .bin.reinterpret(dtype=pl.Int64, endianness="little")
            .cast(pl.Datetime("us"))
        )


class LoadTimestamptzFromBytes(LoadFromBytesImpl):
    def load_from_bytes(self, byte_values: list[bytes | None]) -> pl.Series:
        import polars as pl

        return (
            pl.Series(byte_values, dtype=pl.Binary)
            .bin.reinterpret(dtype=pl.Int64, endianness="little")
            .cast(pl.Datetime("us", time_zone="UTC"))
        )


class LoadBooleanFromBytes(LoadFromBytesImpl):
    def load_from_bytes(self, byte_values: list[bytes | None]) -> pl.Series:
        import polars as pl

        return (
            pl.Series(byte_values, dtype=pl.Binary)
            .bin.reinterpret(dtype=pl.UInt8, endianness="little")
            .cast(pl.Boolean)
        )


class LoadDecimalFromBytes(LoadFromBytesImpl):
    def load_from_bytes(self, byte_values: list[bytes | None]) -> pl.Series:
        import polars as pl
        from polars._plr import PySeries

        dtype = self.polars_dtype
        assert isinstance(dtype, pl.Decimal)
        assert dtype.precision is not None

        return wrap_s(
            PySeries._import_decimal_from_iceberg_binary_repr(
                bytes_list=byte_values,
                precision=dtype.precision,
                scale=dtype.scale,
            )
        )


class LoadFixedFromBytes(LoadBinaryFromBytes): ...


class LoadInt32FromBytes(LoadFromBytesImpl):
    def load_from_bytes(self, byte_values: list[bytes | None]) -> pl.Series:
        import polars as pl

        return pl.Series(byte_values, dtype=pl.Binary).bin.reinterpret(
            dtype=pl.Int32, endianness="little"
        )


class LoadInt64FromBytes(LoadFromBytesImpl):
    def load_from_bytes(self, byte_values: list[bytes | None]) -> pl.Series:
        import polars as pl

        s = pl.Series(byte_values, dtype=pl.Binary)

        return s.bin.reinterpret(dtype=pl.Int64, endianness="little").fill_null(
            s.bin.reinterpret(dtype=pl.Int32, endianness="little").cast(pl.Int64)
        )


class LoadStringFromBytes(LoadFromBytesImpl):
    def load_from_bytes(self, byte_values: list[bytes | None]) -> pl.Series:
        import polars as pl

        return pl.Series(byte_values, dtype=pl.Binary).cast(pl.String)
