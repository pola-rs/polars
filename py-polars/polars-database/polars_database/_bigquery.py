"""Typed row batches from the Google BigQuery DB-API cursor."""

from __future__ import annotations

from decimal import Decimal
from functools import partial
from typing import TYPE_CHECKING, Any

import polars as pl
from polars.exceptions import ComputeError, DuplicateError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from polars._typing import PolarsDataType


_SCALAR_DTYPES = {
    "BOOLEAN": pl.Boolean,
    "BOOL": pl.Boolean,
    "BYTES": pl.Binary,
    "DATE": pl.Date,
    "DATETIME": pl.Datetime("us"),
    "FLOAT": pl.Float64,
    "FLOAT64": pl.Float64,
    "GEOGRAPHY": pl.String,
    "INTEGER": pl.Int64,
    "INT64": pl.Int64,
    "STRING": pl.String,
    "TIME": pl.Time,
    "TIMESTAMP": pl.Datetime("us", "UTC"),
}


def _decimal_text(value: Decimal) -> str:
    # The Storage API pads to the Arrow decimal's scale. Emit the same fixed-point
    # text for a value whether it came from Storage or REST, without a decimal context.
    if not value.is_finite():
        msg = "BigQuery returned a non-finite decimal"
        raise ComputeError(msg)
    text = format(value, "f") if value else "0"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _decimal_value(value: Decimal, *, name: str, precision: int, scale: int) -> Decimal:
    # Rescale exactly before Polars decodes the Python Decimal as Decimal128. A
    # Storage value can have more than 38 coefficient digits because of padding.
    sign, digits, exponent = value.as_tuple()
    if isinstance(exponent, int):
        if not value:
            return Decimal((0, (0,), -scale))
        shift = exponent + scale
        if len(digits) + shift <= precision and (shift >= 0 or not any(digits[shift:])):
            if shift < 0:
                digits = digits[:shift]
            elif shift > 0:
                digits += (0,) * shift
            else:
                return value
            return Decimal((sign, digits, -scale))

    msg = f"BigQuery field {name!r} has a value outside Decimal({precision}, {scale})"
    raise ComputeError(msg)


def _convert(value: Any, convert: Callable[[Any], Any] | None) -> Any:
    return value if value is None or convert is None else convert(value)


def _field_conversion(
    field: Any, name: str
) -> tuple[PolarsDataType, Callable[[Any], Any] | None]:
    kind = field.field_type.upper()
    if kind == "TIMESTAMP":
        # The newer drivers expose an enum whose microsecond value is None.
        precision = getattr(field, "timestamp_precision", None)
        if getattr(precision, "value", precision) not in (None, 6):
            msg = (
                f"BigQuery field {name!r} has unsupported TIMESTAMP precision; "
                "use CAST(<timestamp> AS STRING) in the source query"
            )
            raise ComputeError(msg)
    if kind in {"NUMERIC", "DECIMAL", "BIGNUMERIC", "BIGDECIMAL"}:
        wide = kind in {"BIGNUMERIC", "BIGDECIMAL"}
        precision = field.precision
        scale = field.scale
        if precision is None and scale is None and not wide:
            precision, scale = 38, 9
        if precision is not None and scale is None:
            scale = 0
        max_integer_digits, max_scale = (38, 38) if wide else (29, 9)
        if precision is None and scale is None:
            dtype, convert = pl.String, _decimal_text
        elif (
            precision is None
            or scale is None
            or not (
                0 <= scale <= max_scale
                and max(1, scale) <= precision <= scale + max_integer_digits
            )
        ):
            msg = f"BigQuery field {name!r} has invalid {kind} precision or scale"
            raise ComputeError(msg)
        elif precision > 38:
            dtype, convert = pl.String, _decimal_text
        else:
            dtype = pl.Decimal(precision, scale)
            convert = partial(
                _decimal_value, name=name, precision=precision, scale=scale
            )
    elif kind in {"RECORD", "STRUCT"}:
        schema, conversions = _schema(field.fields, prefix=f"{name}.")
        dtype = pl.Struct(schema)
        convert = (
            (
                lambda value: {
                    key: _convert(value[key], conversion)
                    for key, conversion in zip(schema, conversions, strict=True)
                }
            )
            if any(conversions)
            else None
        )
    elif kind == "RANGE":
        element_type = getattr(field.range_element_type, "element_type", None)
        if element_type in {"DATE", "DATETIME", "TIMESTAMP"}:
            element_dtype = _SCALAR_DTYPES[element_type]
            dtype = pl.Struct({"start": element_dtype, "end": element_dtype})
        else:
            dtype = pl.Unknown
        convert = None
    elif kind in _SCALAR_DTYPES:
        dtype, convert = _SCALAR_DTYPES[kind], None
    else:
        # Keep row inference for types with no fixed Polars mapping. For example,
        # REST decodes JSON as Python objects and Storage returns JSON text.
        dtype, convert = pl.Unknown, None

    if field.mode == "REPEATED":
        return pl.List(dtype), (
            None
            if convert is None
            else lambda values: [_convert(v, convert) for v in values]
        )
    return dtype, convert


def _schema(
    fields: Sequence[Any], *, prefix: str = ""
) -> tuple[dict[str, PolarsDataType], list[Callable[[Any], Any] | None]]:
    schema = {}
    conversions = []
    for field in fields:
        name = f"{prefix}{field.name}"
        if field.name in schema:
            msg = f"BigQuery field {name!r} appears more than once in the result schema"
            raise DuplicateError(msg)
        dtype, convert = _field_conversion(field, name)
        schema[field.name] = dtype
        conversions.append(convert)
    return schema, conversions


def _has_unknown(dtype: PolarsDataType) -> bool:
    if isinstance(dtype, pl.Struct):
        return any(_has_unknown(field.dtype) for field in dtype.fields)
    if isinstance(dtype, pl.List):
        return _has_unknown(dtype.inner)
    return dtype == pl.Unknown


def _resolve_unknown(dtype: PolarsDataType, values: Sequence[Any]) -> PolarsDataType:
    """Infer unknown leaves using row construction, even inside empty containers."""
    if isinstance(dtype, pl.Struct):
        return pl.Struct(
            {
                field.name: _resolve_unknown(
                    field.dtype,
                    [value[field.name] for value in values if value is not None],
                )
                if _has_unknown(field.dtype)
                else field.dtype
                for field in dtype.fields
            }
        )
    if isinstance(dtype, pl.List):
        elements = [item for value in values if value is not None for item in value]
        return pl.List(_resolve_unknown(dtype.inner, elements))
    if dtype == pl.Unknown:
        if not values:
            return pl.Null
        return pl.DataFrame(
            [(value,) for value in values], orient="row", infer_schema_length=None
        ).dtypes[0]
    return dtype


def result_frames(
    cursor: Any,
    batch_size: int,
    *,
    fetchmany: Callable[[int], Sequence[Any]] | None = None,
) -> tuple[pl.DataFrame, Iterator[pl.DataFrame], list[str]]:
    # BigQuery's DB-API description flattens REPEATED fields and drops nested fields
    # and decimal parameters. The cursor uses its RowIterator's schema to build that
    # description (older versions use the cached QueryJob results). Access the schema
    # directly: cursor.query_job can make a second API request.
    fields = getattr(getattr(cursor, "_query_rows", None), "schema", None)
    if fields is None:
        job = getattr(cursor, "_query_job", None)
        fields = getattr(getattr(job, "_query_results", None), "schema", None)
    if fields is None:
        msg = "the BigQuery DB-API cursor did not provide a result schema"
        raise ComputeError(msg)
    schema, conversions = _schema(fields)
    text_columns = [
        field.name
        for field in fields
        if field.field_type == "STRING" and field.mode != "REPEATED"
    ]
    converts = any(conversions)
    unknown = [
        (i, name, dtype)
        for i, (name, dtype) in enumerate(schema.items())
        if _has_unknown(dtype)
    ]
    nested_unknown = [entry for entry in unknown if entry[2] != pl.Unknown]
    # SQLAlchemy owns any buffered rows and applies its own result processors.
    read_rows = cursor.fetchmany if fetchmany is None else fetchmany

    def batches() -> Iterator[pl.DataFrame]:
        while rows := read_rows(batch_size):
            data = (
                [
                    tuple(
                        _convert(v, fn) for v, fn in zip(row, conversions, strict=True)
                    )
                    for row in rows
                ]
                if converts
                else [tuple(row) for row in rows]
            )
            batch_schema = schema
            if nested_unknown:
                # A nested Unknown can reach Arrow before it is resolved, for
                # example on a null list. Infer just its leaves first.
                batch_schema = schema | {
                    name: _resolve_unknown(dtype, [row[i] for row in data])
                    for i, name, dtype in nested_unknown
                }
            yield pl.DataFrame(
                data,
                schema=batch_schema,
                orient="row",
                infer_schema_length=None,
            )

    empty_schema = schema.copy() if unknown else schema
    for _, name, dtype in unknown:
        empty_schema[name] = _resolve_unknown(dtype, ())
    return pl.DataFrame(schema=empty_schema), batches(), text_columns
