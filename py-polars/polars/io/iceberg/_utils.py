from __future__ import annotations

import ast
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
    UnaryOp,
)
from functools import singledispatch
from typing import TYPE_CHECKING, Any, Callable

import polars._reexport as pl
from polars._utils.convert import to_py_date, to_py_datetime
from polars.dependencies import pyiceberg

if TYPE_CHECKING:
    from datetime import date, datetime

    from pyiceberg.table import Table

    from polars import DataFrame, Series

_temporal_conversions: dict[str, Callable[..., datetime | date]] = {
    "to_py_date": to_py_date,
    "to_py_datetime": to_py_datetime,
}


def _scan_pyarrow_dataset_impl(
    tbl: Table,
    with_columns: list[str] | None = None,
    predicate: str | None = None,
    n_rows: int | None = None,
    snapshot_id: int | None = None,
    **kwargs: Any,
) -> DataFrame | Series:
    """
    Take the projected columns and materialize an arrow table.

    Parameters
    ----------
    tbl
        pyarrow dataset
    with_columns
        Columns that are projected
    predicate
        pyarrow expression that can be evaluated with eval
    n_rows:
        Materialize only n rows from the arrow dataset.
    snapshot_id:
        The snapshot ID to scan from.
    batch_size
        The maximum row count for scanned pyarrow record batches.
    kwargs:
        For backward compatibility

    Returns
    -------
    DataFrame
    """
    from polars import from_arrow

    scan = tbl.scan(limit=n_rows, snapshot_id=snapshot_id)

    if with_columns is not None:
        scan = scan.select(*with_columns)

    if predicate is not None:
        try:
            expr_ast = _to_ast(predicate)
            pyiceberg_expr = _convert_predicate(expr_ast)
        except ValueError as e:
            msg = f"Could not convert predicate to PyIceberg: {predicate}"
            raise ValueError(msg) from e

        scan = scan.filter(pyiceberg_expr)

    return from_arrow(scan.to_arrow())


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
    return a.id


@_convert_predicate.register(UnaryOp)
def _(a: UnaryOp) -> Any:
    if isinstance(a.op, Invert):
        return pyiceberg.expressions.Not(_convert_predicate(a.operand))
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
    else:
        ref = _convert_predicate(a.func.value)[0]  # type: ignore[attr-defined]
        if f == "isin":
            return pyiceberg.expressions.In(ref, args[0])
        elif f == "is_null":
            return pyiceberg.expressions.IsNull(ref)
        elif f == "is_nan":
            return pyiceberg.expressions.IsNaN(ref)

    msg = f"Unknown call: {f!r}"
    raise ValueError(msg)


@_convert_predicate.register(Attribute)
def _(a: Attribute) -> Any:
    return a.attr


@_convert_predicate.register(BinOp)
def _(a: BinOp) -> Any:
    lhs = _convert_predicate(a.left)
    rhs = _convert_predicate(a.right)

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
    rhs = _convert_predicate(a.comparators[0])

    if isinstance(op, Gt):
        return pyiceberg.expressions.GreaterThan(lhs, rhs)
    if isinstance(op, GtE):
        return pyiceberg.expressions.GreaterThanOrEqual(lhs, rhs)
    if isinstance(op, Eq):
        return pyiceberg.expressions.EqualTo(lhs, rhs)
    if isinstance(op, Lt):
        return pyiceberg.expressions.LessThan(lhs, rhs)
    if isinstance(op, LtE):
        return pyiceberg.expressions.LessThanOrEqual(lhs, rhs)
    else:
        msg = f"Unknown comparison: {op}"
        raise TypeError(msg)


@_convert_predicate.register(List)
def _(a: List) -> Any:
    return [_convert_predicate(e) for e in a.elts]


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

            if not projected_type.is_primitive:
                self.partition_values[field_id] = (
                    f"non-primitive type: {projected_type}"
                )

            _, output_dtype = pl.Schema(
                schema_to_pyarrow(pyiceberg.schema.Schema(projected_field))
            ).popitem()

            self.partition_values_dtypes[field_id] = output_dtype

            for schema in table.schemas().values():
                try:
                    type_this_schema = schema.find_field(field_id).field_type
                except ValueError:
                    continue

                if not (
                    projected_type == type_this_schema
                    or (
                        isinstance(projected_type, LongType)
                        and isinstance(type_this_schema, IntegerType)
                    )
                    or (
                        isinstance(projected_type, (DoubleType, FloatType))
                        and isinstance(type_this_schema, (DoubleType, FloatType))
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
            self.partition_values = {
                k: f"partition spec ID not found: {partition_spec_id}"
                for k in self.partition_values
            }
            return

        for i, source_field_id in identity_transforms:
            partition_value = partition_values[i]

            if isinstance(values := self.partition_values[source_field_id], list):
                # extend() - there can be gaps from partitions being
                # added/removed/re-added
                values.extend(None for _ in range(current_index - len(values)))
                values.append(partition_value)

    def finish(self) -> dict[int, pl.Series | str]:
        from polars.datatypes import Date, Datetime, Duration, Int32, Int64, Time

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

                    s = pl.Series(v, dtype=constructor_dtype)

                    if isinstance(output_dtype, Time):
                        # Physical from PyIceberg is in microseconds, physical
                        # used by polars is in nanoseconds.
                        s = s * 1000

                    s = s.cast(output_dtype)

                    out[field_id] = s

                except Exception as e:
                    out[field_id] = f"failed to load partition values: {e}"

        return out
