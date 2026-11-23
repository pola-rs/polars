"""Typed row batches from the Google BigQuery DB-API cursor."""

from __future__ import annotations

from decimal import Decimal
from enum import IntFlag, auto
from functools import partial
from json import loads
from operator import itemgetter
from typing import TYPE_CHECKING, Any

import polars as pl
from polars.exceptions import ComputeError, DuplicateError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from polars._typing import PolarsDataType


_SCALAR_DTYPES: dict[str, PolarsDataType] = {
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


class _JsonKind(IntFlag):
    BOOLEAN = auto()
    INTEGER = auto()
    INEXACT_AS_FLOAT = auto()
    FLOAT = auto()
    STRING = auto()
    ARRAY = auto()
    OBJECT = auto()


_JSON_KINDS = {
    bool: _JsonKind.BOOLEAN,
    int: _JsonKind.INTEGER,
    float: _JsonKind.FLOAT,
    str: _JsonKind.STRING,
    list: _JsonKind.ARRAY,
    dict: _JsonKind.OBJECT,
}


class _JsonObservations:
    def __init__(self) -> None:
        self.kinds: dict[tuple[str | None, ...], _JsonKind] = {}
        self.object_keys: dict[tuple[str | None, ...], set[str]] = {}
        self.new_objects: set[tuple[str | None, ...]] = set()


class _JsonBatch:
    def __init__(self) -> None:
        self.kinds: dict[tuple[str | None, ...], _JsonKind] = {}
        self.object_keys: dict[tuple[str | None, ...], set[str]] = {}
        self.wide_integers: set[tuple[str | None, ...]] = set()
        self.dtypes: dict[tuple[str | None, ...], PolarsDataType] = {}
        self.prepare_paths: set[tuple[str | None, ...]] = set()

    def prepare_at(self, path: tuple[str | None, ...]) -> None:
        if path not in self.prepare_paths:
            self.prepare_paths.update(path[:i] for i in range(1, len(path) + 1))

    def dtype(self, path: tuple[str | None, ...]) -> PolarsDataType:
        if path not in self.dtypes:
            self.dtypes[path] = self._infer_dtype(path)
        return self.dtypes[path]

    def _infer_dtype(self, path: tuple[str | None, ...]) -> PolarsDataType:
        kind = self.kinds.get(path)
        if kind is None:
            return pl.Null
        if kind == _JsonKind.BOOLEAN:
            return pl.Boolean
        if kind == _JsonKind.STRING:
            return pl.String
        if kind == _JsonKind.ARRAY:
            return pl.List(self.dtype((*path, None)))
        if kind == _JsonKind.OBJECT:
            # JSON object keys have no declared order. Use the same order when
            # inferring a sample and when reading batches from the result.
            return pl.Struct(
                {
                    key: self.dtype((*path, key))
                    for key in sorted(self.object_keys[path])
                }
            )
        if kind & _JsonKind.FLOAT:
            return pl.Float64
        # BigQuery JSON integers span the signed and unsigned 64-bit ranges.
        return pl.Int128 if path in self.wide_integers else pl.Int64


def _observe_json(
    value: Any,
    path: tuple[str | None, ...],
    batch: _JsonBatch,
    history: _JsonObservations | None,
    *,
    in_list: bool,
) -> None:
    if value is None:
        return
    kind = _JSON_KINDS[type(value)]
    if kind == _JsonKind.INTEGER:
        if not -(2**63) <= value < 2**63:
            batch.wide_integers.add(path)
        if abs(value) > 2**53:
            bits = value.bit_length()
            if bits > 1024 or value % (1 << (bits - 53)):
                kind = _JsonKind.INEXACT_AS_FLOAT
    batch_combined = batch.kinds.get(path, kind) | kind
    combined = batch_combined | (history.kinds.get(path, kind) if history else kind)
    if combined != kind and combined not in (
        _JsonKind.INTEGER | _JsonKind.FLOAT,
        _JsonKind.INTEGER | _JsonKind.INEXACT_AS_FLOAT,
    ):
        location = str(path[0]) + "".join(
            "[]" if key is None else f".{key}" for key in path[1:]
        )
        msg = (
            f"BigQuery JSON field {location!r} cannot be inferred losslessly "
            "from mixed JSON types; use TO_JSON_STRING(<json>) in the source query"
        )
        raise ComputeError(msg)
    batch.kinds[path] = batch_combined
    if history is not None:
        history.kinds[path] = combined
    if in_list and batch_combined == (_JsonKind.INTEGER | _JsonKind.FLOAT):
        batch.prepare_at(path)
    if kind == _JsonKind.ARRAY:
        batch.prepare_at(path)
        item_path = (*path, None)
        for item in value:
            _observe_json(item, item_path, batch, history, in_list=True)
    elif kind == _JsonKind.OBJECT:
        keys = batch.object_keys.setdefault(path, set())
        # Keep all keys observed in the object's first nonempty batch. A result with
        # changing keys may grow only the current batch's observations after that.
        known_keys = None
        if history is not None and value:
            known_keys = history.object_keys.get(path)
            if known_keys is None:
                known_keys = history.object_keys[path] = set(value)
                history.new_objects.add(path)
            elif path in history.new_objects:
                known_keys.update(value)
        for key, item in value.items():
            keys.add(key)
            _observe_json(
                item,
                (*path, key),
                batch,
                history if known_keys is not None and key in known_keys else None,
                in_list=in_list,
            )


def _prepare_json(value: Any, path: tuple[str | None, ...], batch: _JsonBatch) -> Any:
    if value is None or path not in batch.prepare_paths:
        return value
    dtype = batch.dtype(path)
    if isinstance(dtype, pl.List):
        # Row construction infers a Python list's dtype from its own first item,
        # even when a schema was supplied. Pass the typed elements instead.
        item_path = (*path, None)
        return pl.Series(
            [_prepare_json(item, item_path, batch) for item in value], dtype=dtype.inner
        )
    if isinstance(dtype, pl.Struct):
        return {
            key: _prepare_json(item, (*path, key), batch) for key, item in value.items()
        }
    if dtype == pl.Float64:
        # The typed Series constructor needs floats even when the integer fits exactly.
        return float(value)
    return value


def _field_conversion(
    field: Any,
    path: tuple[str, ...],
    convert_json: Callable[[tuple[str, ...], Any], Any],
    json_fields: dict[tuple[str, ...], bool],
    *,
    in_list: bool,
) -> tuple[PolarsDataType, Callable[[Any], Any] | None]:
    kind = field.field_type.upper()
    name = ".".join(path)
    in_list = in_list or field.mode == "REPEATED"
    dtype: PolarsDataType
    convert: Callable[[Any], Any] | None
    if kind == "TIMESTAMP":
        # The newer drivers expose an enum whose microsecond value is None.
        precision = getattr(field, "timestamp_precision", None)
        if getattr(precision, "value", precision) not in (None, 6):
            msg = (
                f"BigQuery field {name!r} has unsupported TIMESTAMP precision; "
                "use CAST(<timestamp> AS STRING) in the source query"
            )
            raise ComputeError(msg)
    if kind == "INTERVAL":
        msg = (
            f"BigQuery field {name!r} has unsupported type INTERVAL; "
            "use CAST(<interval> AS STRING) in the source query"
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
        schema, conversions = _schema(
            field.fields, convert_json, json_fields, prefix=path, in_list=in_list
        )
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
    elif kind == "JSON":
        json_fields[path] = in_list
        dtype, convert = pl.Unknown, partial(convert_json, path)
    else:
        # Keep row inference for types with no fixed Polars mapping.
        dtype, convert = pl.Unknown, None

    if field.mode == "REPEATED":
        return pl.List(dtype), (
            None
            if convert is None
            else lambda values: [_convert(v, convert) for v in values]
        )
    return dtype, convert


def _schema(
    fields: Sequence[Any],
    convert_json: Callable[[tuple[str, ...], Any], Any],
    json_fields: dict[tuple[str, ...], bool],
    *,
    prefix: tuple[str, ...] = (),
    in_list: bool = False,
) -> tuple[dict[str, PolarsDataType], list[Callable[[Any], Any] | None]]:
    schema = {}
    conversions = []
    for field in fields:
        path = (*prefix, field.name)
        if field.name in schema:
            msg = f"BigQuery field {'.'.join(path)!r} appears more than once in the result schema"
            raise DuplicateError(msg)
        dtype, convert = _field_conversion(
            field, path, convert_json, json_fields, in_list=in_list
        )
        schema[field.name] = dtype
        conversions.append(convert)
    return schema, conversions


def _fetch_first_batch(
    cursor: Any, fetchmany: Callable[[int], list[Any]], batch_size: int
) -> tuple[list[Any], bool]:
    storage_fetch = getattr(cursor, "_bqstorage_fetch", None)
    if storage_fetch is None:
        return fetchmany(batch_size), False
    # Both REST and Storage use a generic row iterator and the driver may pick
    # REST with a Storage client configured. Observe its actual Arrow fetch.
    used_storage = False

    def fetch_storage(*args: Any, **kwargs: Any) -> Any:
        nonlocal used_storage
        used_storage = True
        return storage_fetch(*args, **kwargs)

    instance_method = "_bqstorage_fetch" in vars(cursor)
    cursor._bqstorage_fetch = fetch_storage
    try:
        rows = fetchmany(batch_size)
    finally:
        if instance_method:
            cursor._bqstorage_fetch = storage_fetch
        else:
            del cursor._bqstorage_fetch
    return rows, used_storage


def _has_unknown(dtype: PolarsDataType) -> bool:
    if isinstance(dtype, pl.Struct):
        return any(_has_unknown(field.dtype) for field in dtype.fields)
    if isinstance(dtype, pl.List):
        return _has_unknown(dtype.inner)
    return dtype == pl.Unknown


def _resolve_unknown(
    dtype: PolarsDataType,
    values: Callable[[], Iterable[Any]],
    path: tuple[str, ...],
    batch: _JsonBatch,
    json_fields: dict[tuple[str, ...], bool],
) -> PolarsDataType:
    """Resolve nested unknowns, reading values for leaves without native JSON types."""
    if isinstance(dtype, pl.Struct):

        def field_values(name: str) -> Iterator[Any]:
            return (value[name] for value in values() if value is not None)

        return pl.Struct(
            {
                field.name: _resolve_unknown(
                    field.dtype,
                    partial(field_values, field.name),
                    (*path, field.name),
                    batch,
                    json_fields,
                )
                if _has_unknown(field.dtype)
                else field.dtype
                for field in dtype.fields
            }
        )
    if isinstance(dtype, pl.List):
        return pl.List(
            _resolve_unknown(
                dtype.inner,
                lambda: (
                    item for value in values() if value is not None for item in value
                ),
                path,
                batch,
                json_fields,
            )
        )
    if dtype == pl.Unknown:
        if path in json_fields:
            return batch.dtype(path)
        rows = [(value,) for value in values()]
        if not rows:
            return pl.Null
        return pl.DataFrame(rows, orient="row", infer_schema_length=None).dtypes[0]
    return dtype


def _prepare_unknown(
    value: Any,
    declared: PolarsDataType,
    resolved: PolarsDataType,
    path: tuple[str, ...],
    batch: _JsonBatch,
) -> Any:
    if value is None or path not in batch.prepare_paths:
        return value
    if isinstance(declared, pl.Struct) and isinstance(resolved, pl.Struct):
        return {
            field.name: _prepare_unknown(
                value[field.name], field.dtype, typed.dtype, (*path, field.name), batch
            )
            for field, typed in zip(declared.fields, resolved.fields, strict=True)
        }
    if isinstance(declared, pl.List) and isinstance(resolved, pl.List):
        return pl.Series(
            [
                _prepare_unknown(item, declared.inner, resolved.inner, path, batch)
                for item in value
            ],
            dtype=resolved.inner,
        )
    return _prepare_json(value, path, batch)


def result_frames(
    cursor: Any,
    batch_size: int,
    *,
    fetchmany: Callable[[int], list[Any]] | None = None,
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
    fetch_rows = cursor.fetchmany if fetchmany is None else fetchmany
    storage = False
    json_history = _JsonObservations()
    json_batch = _JsonBatch()
    # Tracks which native JSON fields are themselves in a BigQuery REPEATED list.
    json_fields: dict[tuple[str, ...], bool] = {}

    def convert_json(path: tuple[str, ...], value: Any) -> Any:
        # REST already decodes JSON, including JSON null as Python None.
        if storage:
            value = loads(value)
        in_list = json_fields[path]
        if value is not None and in_list:
            json_batch.prepare_at(path)
        _observe_json(value, path, json_batch, json_history, in_list=in_list)
        return value

    schema, conversions = _schema(fields, convert_json, json_fields)
    text_columns = [
        field.name
        for field in fields
        if field.field_type == "STRING" and field.mode != "REPEATED"
    ]
    unknown = [
        (i, name, dtype)
        for i, (name, dtype) in enumerate(schema.items())
        if _has_unknown(dtype)
    ]
    infer_before_rows = [
        entry
        for entry in unknown
        if entry[2] != pl.Unknown or (entry[1],) in json_fields
    ]

    def batches() -> Iterator[pl.DataFrame]:
        nonlocal storage, json_batch
        if unknown:
            rows, storage = _fetch_first_batch(cursor, fetch_rows, batch_size)
        else:
            rows = fetch_rows(batch_size)
        converts = any(conversions)
        while rows:
            if json_fields:
                json_batch = _JsonBatch()
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
            json_history.new_objects.clear()
            batch_schema = schema
            if infer_before_rows:
                # A nested Unknown can reach Arrow before it is resolved, for
                # example on a null list. Infer just its leaves first.
                batch_schema = schema | {
                    name: _resolve_unknown(
                        dtype,
                        partial(map, itemgetter(i), data),
                        (name,),
                        json_batch,
                        json_fields,
                    )
                    for i, name, dtype in infer_before_rows
                }
            if json_batch.prepare_paths:
                prepared = [
                    entry
                    for entry in unknown
                    if (entry[1],) in json_batch.prepare_paths
                ]
                for i, row in enumerate(data):
                    values = list(row)
                    for j, name, dtype in prepared:
                        values[j] = _prepare_unknown(
                            values[j], dtype, batch_schema[name], (name,), json_batch
                        )
                    data[i] = tuple(values)
            yield pl.DataFrame(
                data,
                schema=batch_schema,
                orient="row",
                infer_schema_length=None,
            )
            rows = fetch_rows(batch_size)

    empty_schema = schema.copy() if unknown else schema
    for _, name, dtype in unknown:
        empty_schema[name] = _resolve_unknown(
            dtype, tuple, (name,), json_batch, json_fields
        )
    return pl.DataFrame(schema=empty_schema), batches(), text_columns
