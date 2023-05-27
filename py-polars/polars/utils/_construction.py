from __future__ import annotations

import contextlib
import warnings
from datetime import date, datetime, time, timedelta
from decimal import Decimal as PyDecimal
from functools import lru_cache, partial, singledispatch
from itertools import islice, zip_longest
from sys import version_info
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
    get_type_hints,
)

import polars._reexport as pl
from polars import functions as F
from polars.datatypes import (
    FLOAT_DTYPES,
    INTEGER_DTYPES,
    N_INFER_DEFAULT,
    TEMPORAL_DTYPES,
    Boolean,
    Categorical,
    Date,
    Datetime,
    Duration,
    Float32,
    List,
    Object,
    Struct,
    Time,
    Unknown,
    Utf8,
    dtype_to_py_type,
    is_polars_dtype,
    py_type_to_dtype,
)
from polars.datatypes.constructor import (
    numpy_type_to_constructor,
    numpy_values_and_dtype,
    polars_type_to_constructor,
    py_type_to_constructor,
)
from polars.dependencies import (
    _NUMPY_AVAILABLE,
    _check_for_numpy,
    _check_for_pandas,
    _check_for_pydantic,
    dataclasses,
    pydantic,
)
from polars.dependencies import numpy as np
from polars.dependencies import pandas as pd
from polars.dependencies import pyarrow as pa
from polars.exceptions import ComputeError, ShapeError, TimeZoneAwareConstructorWarning
from polars.utils._wrap import wrap_df, wrap_s
from polars.utils.meta import threadpool_size
from polars.utils.various import _is_generator, arrlen, find_stacklevel, range_to_series

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import PyDataFrame, PySeries

if TYPE_CHECKING:
    from polars import DataFrame, Series
    from polars.type_aliases import (
        Orientation,
        PolarsDataType,
        SchemaDefinition,
        SchemaDict,
    )


def _get_annotations(obj: type) -> dict[str, Any]:
    return getattr(obj, "__annotations__", {})


if version_info >= (3, 10):

    def type_hints(obj: type) -> dict[str, Any]:
        try:
            # often the same as obj.__annotations__, but handles forward references
            # encoded as string literals, adds Optional[t] if a default value equal
            # to None is set and recursively replaces 'Annotated[T, ...]' with 'T'.
            return get_type_hints(obj)
        except TypeError:
            # fallback on edge-cases (eg: InitVar inference on python 3.10).
            return _get_annotations(obj)

else:
    type_hints = _get_annotations


@lru_cache(64)
def is_namedtuple(cls: Any, annotated: bool = False) -> bool:
    """Check whether given class derives from NamedTuple."""
    if all(hasattr(cls, attr) for attr in ("_fields", "_field_defaults", "_replace")):
        if len(cls.__annotations__) == len(cls._fields) if annotated else True:
            return all(isinstance(fld, str) for fld in cls._fields)
    return False


def is_pydantic_model(value: Any) -> bool:
    """Check whether value derives from a pydantic.BaseModel."""
    return _check_for_pydantic(value) and isinstance(value, pydantic.BaseModel)


def contains_nested(value: Any, is_nested: Callable[[Any], bool]) -> bool:
    """Determine if value contains (or is) nested structured data."""
    if is_nested(value):
        return True
    elif isinstance(value, dict):
        return any(contains_nested(v, is_nested) for v in value.values())
    elif isinstance(value, (list, tuple)):
        return any(contains_nested(v, is_nested) for v in value)
    return False


def include_unknowns(
    schema: SchemaDict, cols: Sequence[str]
) -> MutableMapping[str, PolarsDataType]:
    """Complete partial schema dict by including Unknown type."""
    return {
        col: (schema.get(col, Unknown) or Unknown)  # type: ignore[truthy-bool]
        for col in cols
    }


def nt_unpack(obj: Any) -> Any:
    """Recursively unpack a nested NamedTuple."""
    if isinstance(obj, dict):
        return {key: nt_unpack(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [nt_unpack(value) for value in obj]
    elif is_namedtuple(obj.__class__):
        return {key: nt_unpack(value) for key, value in obj._asdict().items()}
    elif isinstance(obj, tuple):
        return tuple(nt_unpack(value) for value in obj)
    else:
        return obj


################################
# Series constructor interface #
################################


def series_to_pyseries(name: str, values: Series) -> PySeries:
    """Construct a PySeries from a Polars Series."""
    py_s = values._s
    py_s.rename(name)
    return py_s


def arrow_to_pyseries(name: str, values: pa.Array, rechunk: bool = True) -> PySeries:
    """Construct a PySeries from an Arrow array."""
    array = coerce_arrow(values)

    # special handling of empty categorical arrays
    if (
        len(array) == 0
        and isinstance(array.type, pa.DictionaryType)
        and array.type.value_type
        in (
            pa.utf8(),
            pa.large_utf8(),
        )
    ):
        pys = pl.Series(name, [], dtype=Categorical)._s

    elif not hasattr(array, "num_chunks"):
        pys = PySeries.from_arrow(name, array)
    else:
        if array.num_chunks > 1:
            # somehow going through ffi with a structarray
            # returns the first chunk everytime
            if isinstance(array.type, pa.StructType):
                pys = PySeries.from_arrow(name, array.combine_chunks())
            else:
                it = array.iterchunks()
                pys = PySeries.from_arrow(name, next(it))
                for a in it:
                    pys.append(PySeries.from_arrow(name, a))
        elif array.num_chunks == 0:
            pys = PySeries.from_arrow(name, pa.array([], array.type))
        else:
            pys = PySeries.from_arrow(name, array.chunks[0])

        if rechunk:
            pys.rechunk(in_place=True)

    return pys


def numpy_to_pyseries(
    name: str,
    values: np.ndarray[Any, Any],
    strict: bool = True,
    nan_to_null: bool = False,
) -> PySeries:
    """Construct a PySeries from a numpy array."""
    if not values.flags["C_CONTIGUOUS"]:
        values = np.array(values)

    if len(values.shape) == 1:
        values, dtype = numpy_values_and_dtype(values)
        constructor = numpy_type_to_constructor(dtype)
        return constructor(
            name, values, nan_to_null if dtype in (np.float32, np.float64) else strict
        )
    elif len(values.shape) == 2:
        pyseries_container = []
        for row in range(values.shape[0]):
            pyseries_container.append(
                numpy_to_pyseries("", values[row, :], strict, nan_to_null)
            )
        return PySeries.new_series_list(name, pyseries_container, False)
    else:
        return PySeries.new_object(name, values, strict)


def _get_first_non_none(values: Sequence[Any | None]) -> Any:
    """
    Return the first value from a sequence that isn't None.

    If sequence doesn't contain non-None values, return None.

    """
    if values is not None:
        return next((v for v in values if v is not None), None)


def sequence_from_anyvalue_or_object(name: str, values: Sequence[Any]) -> PySeries:
    """
    Last resort conversion.

    AnyValues are most flexible and if they fail we go for object types

    """
    try:
        return PySeries.new_from_anyvalues(name, values, strict=True)
    # raised if we cannot convert to Wrap<AnyValue>
    except RuntimeError:
        return PySeries.new_object(name, values, False)
    except ComputeError as e:
        if "mixed dtypes" in str(e):
            return PySeries.new_object(name, values, False)
        raise e


def iterable_to_pyseries(
    name: str,
    values: Iterable[Any],
    dtype: PolarsDataType | None = None,
    strict: bool = True,
    dtype_if_empty: PolarsDataType | None = None,
    chunk_size: int = 1_000_000,
) -> PySeries:
    """Construct a PySeries from an iterable/generator."""
    if not isinstance(values, Generator):
        values = iter(values)

    def to_series_chunk(values: list[Any], dtype: PolarsDataType | None) -> Series:
        return pl.Series(
            name=name,
            values=values,
            dtype=dtype,
            strict=strict,
            dtype_if_empty=dtype_if_empty,
        )

    n_chunks = 0
    series: Series = None  # type: ignore[assignment]
    while True:
        slice_values = list(islice(values, chunk_size))
        if not slice_values:
            break
        schunk = to_series_chunk(slice_values, dtype)
        if series is None:
            series = schunk
            dtype = series.dtype
        else:
            series.append(schunk, append_chunks=True)
            n_chunks += 1

    if series is None:
        series = to_series_chunk([], dtype)
    if n_chunks > 0:
        series.rechunk(in_place=True)

    return series._s


def _construct_series_with_fallbacks(
    constructor: Callable[[str, Sequence[Any], bool], PySeries],
    name: str,
    values: Sequence[Any],
    target_dtype: PolarsDataType | None,
    strict: bool,
) -> PySeries:
    """Construct Series, with fallbacks for basic type mismatch (eg: bool/int)."""
    while True:
        try:
            return constructor(name, values, strict)
        except TypeError as exc:
            str_exc = str(exc)

            # from x to float
            # error message can be:
            #   - integers: "'float' object cannot be interpreted as an integer"
            if "'float'" in str_exc and (
                # we do not accept float values as int/temporal, as it causes silent
                # information loss; the caller should explicitly cast in this case.
                target_dtype
                not in (INTEGER_DTYPES | TEMPORAL_DTYPES)
            ):
                constructor = py_type_to_constructor(float)

            # from x to string
            # error message can be:
            #   - integers: "'str' object cannot be interpreted as an integer"
            #   - floats: "must be real number, not str"
            elif "'str'" in str_exc or str_exc == "must be real number, not str":
                constructor = py_type_to_constructor(str)

            # from x to int
            # error message can be:
            #   - bools: "'int' object cannot be converted to 'PyBool'"
            elif str_exc == "'int' object cannot be converted to 'PyBool'":
                constructor = py_type_to_constructor(int)

            elif "decimal.Decimal" in str_exc:
                constructor = py_type_to_constructor(PyDecimal)
            else:
                raise exc


def sequence_to_pyseries(
    name: str,
    values: Sequence[Any],
    dtype: PolarsDataType | None = None,
    strict: bool = True,
    dtype_if_empty: PolarsDataType | None = None,
    nan_to_null: bool = False,
) -> PySeries:
    """Construct a PySeries from a sequence."""
    python_dtype: type | None = None

    # empty sequence
    if not values and dtype is None:
        # if dtype for empty sequence could be guessed
        # (e.g comparisons between self and other), default to Float32
        dtype = dtype_if_empty or Float32

    # lists defer to subsequent handling; identify nested type
    elif dtype == List:
        getattr(dtype, "inner", None)
        python_dtype = list

    # infer temporal type handling
    py_temporal_types = {date, datetime, timedelta, time}
    pl_temporal_types = {Date, Datetime, Duration, Time}

    value = _get_first_non_none(values)
    if value is not None:
        if (
            dataclasses.is_dataclass(value)
            or is_pydantic_model(value)
            or is_namedtuple(value.__class__, annotated=True)
        ):
            return pl.DataFrame(values).to_struct(name)._s
        elif isinstance(value, range):
            values = [range_to_series("", v) for v in values]
        else:
            # for temporal dtypes:
            # * if the values are integer, we take the physical branch.
            # * if the values are python types, take the temporal branch.
            # * if the values are ISO-8601 strings, init then convert via strptime.
            # * if the values are floats/other dtypes, this is an error.
            if dtype in py_temporal_types and isinstance(value, int):
                dtype = py_type_to_dtype(dtype)  # construct from integer
            elif (
                dtype in pl_temporal_types or type(dtype) in pl_temporal_types
            ) and not isinstance(value, int):
                python_dtype = dtype_to_py_type(dtype)  # type: ignore[arg-type]

    # physical branch
    # flat data
    if (
        dtype is not None
        and dtype not in (List, Struct, Unknown)
        and is_polars_dtype(dtype)
        and (python_dtype is None)
    ):
        constructor = polars_type_to_constructor(dtype)
        pyseries = _construct_series_with_fallbacks(
            constructor, name, values, dtype, strict
        )
        if dtype in (Date, Datetime, Duration, Time, Categorical, Boolean):
            if pyseries.dtype() != dtype:
                pyseries = pyseries.cast(dtype, True)
        return pyseries

    elif dtype == Struct:
        struct_schema = dtype.to_schema() if isinstance(dtype, Struct) else None
        empty = {}  # type: ignore[var-annotated]
        return sequence_to_pydf(
            data=[(empty if v is None else v) for v in values],
            schema=struct_schema,
            orient="row",
        ).to_struct(name)
    else:
        if python_dtype is None:
            if value is None:
                # Create a series with a dtype_if_empty dtype (if set) or Float32
                # (if not set) for a sequence which contains only None values.
                constructor = polars_type_to_constructor(
                    dtype_if_empty if dtype_if_empty else Float32
                )
                return _construct_series_with_fallbacks(
                    constructor, name, values, dtype, strict
                )

            # generic default dtype
            python_dtype = type(value)

        # temporal branch
        if python_dtype in py_temporal_types:
            if dtype is None:
                dtype = py_type_to_dtype(python_dtype)  # construct from integer
            elif dtype in py_temporal_types:
                dtype = py_type_to_dtype(dtype)

            values_dtype = (
                None
                if value is None
                else py_type_to_dtype(type(value), raise_unmatched=False)
            )
            if values_dtype in FLOAT_DTYPES:
                raise TypeError(
                    # we do not accept float values as temporal; if this is
                    # required, the caller should explicitly cast to int first.
                    f"'float' object cannot be interpreted as a {python_dtype.__name__}"
                )

            # we use anyvalue builder to create the datetime array
            # we store the values internally as UTC and set the timezone
            py_series = PySeries.new_from_anyvalues(name, values, strict)
            time_unit = getattr(dtype, "time_unit", None)
            if time_unit is None:
                s = wrap_s(py_series)
            else:
                s = wrap_s(py_series).dt.cast_time_unit(time_unit)
            if dtype == Datetime and value.tzinfo is not None:
                tz = str(value.tzinfo)
                dtype_tz = dtype.time_zone  # type: ignore[union-attr]
                if dtype_tz is not None and tz != dtype_tz:
                    raise ValueError(
                        "Given time_zone is different from that of timezone aware datetimes."
                        f" Given: '{dtype_tz}', got: '{tz}'."
                    )
                if tz != "UTC":
                    warnings.warn(
                        "Constructing a Series with time-zone-aware "
                        "datetimes results in a Series with UTC time zone. "
                        "To silence this warning, you can filter "
                        "warnings of class TimeZoneAwareConstructorWarning.",
                        TimeZoneAwareConstructorWarning,
                        stacklevel=find_stacklevel(),
                    )
                return s.dt.replace_time_zone("UTC")._s
            return s._s

        elif (
            _check_for_numpy(value)
            and isinstance(value, np.ndarray)
            and len(value.shape) == 1
        ):
            return PySeries.new_series_list(
                name,
                [numpy_to_pyseries("", v, strict, nan_to_null) for v in values],
                strict,
            )

        elif python_dtype in (list, tuple):
            if isinstance(dtype, Object):
                return PySeries.new_object(name, values, strict)
            if dtype:
                srs = sequence_from_anyvalue_or_object(name, values)
                if dtype.is_not(srs.dtype()):
                    srs = srs.cast(dtype, strict=False)
                return srs
            return sequence_from_anyvalue_or_object(name, values)

        elif python_dtype == pl.Series:
            return PySeries.new_series_list(name, [v._s for v in values], strict)

        elif python_dtype == PySeries:
            return PySeries.new_series_list(name, values, strict)
        else:
            constructor = py_type_to_constructor(python_dtype)
            if constructor == PySeries.new_object:
                try:
                    return PySeries.new_from_anyvalues(name, values, strict)
                # raised if we cannot convert to Wrap<AnyValue>
                except RuntimeError:
                    return sequence_from_anyvalue_or_object(name, values)

            return _construct_series_with_fallbacks(
                constructor, name, values, dtype, strict
            )


def _pandas_series_to_arrow(
    values: pd.Series | pd.DatetimeIndex,
    nan_to_null: bool = True,
    length: int | None = None,
) -> pa.Array:
    """
    Convert a pandas Series to an Arrow Array.

    Parameters
    ----------
    values : :class:`pandas.Series` or :class:`pandas.DatetimeIndex`.
        Series to convert to arrow
    nan_to_null : bool, default = True
        Interpret `NaN` as missing values.
    length : int, optional
        in case all values are null, create a null array of this length.
        if unset, length is inferred from values.

    Returns
    -------
    :class:`pyarrow.Array`

    """
    dtype = getattr(values, "dtype", None)
    if dtype == "object":
        first_non_none = _get_first_non_none(values.values)  # type: ignore[arg-type]
        if isinstance(first_non_none, str):
            return pa.array(values, pa.large_utf8(), from_pandas=nan_to_null)
        elif first_non_none is None:
            return pa.nulls(length or len(values), pa.large_utf8())
        return pa.array(values, from_pandas=nan_to_null)
    elif dtype:
        return pa.array(values, from_pandas=nan_to_null)
    else:
        # Pandas Series is actually a Pandas DataFrame when the original dataframe
        # contains duplicated columns and a duplicated column is requested with df["a"].
        raise ValueError(
            "Duplicate column names found: "
            + f"{str(values.columns.tolist())}"  # type: ignore[union-attr]
        )


def pandas_to_pyseries(
    name: str, values: pd.Series | pd.DatetimeIndex, nan_to_null: bool = True
) -> PySeries:
    """Construct a PySeries from a pandas Series or DatetimeIndex."""
    # TODO: Change `if not name` to `if name is not None` once name is Optional[str]
    if not name and values.name is not None:
        name = str(values.name)
    return arrow_to_pyseries(
        name, _pandas_series_to_arrow(values, nan_to_null=nan_to_null)
    )


###################################
# DataFrame constructor interface #
###################################


def _handle_columns_arg(
    data: list[PySeries], columns: Sequence[str] | None = None, from_dict: bool = False
) -> list[PySeries]:
    """Rename data according to columns argument."""
    if not columns:
        return data
    else:
        if not data:
            return [pl.Series(c, None)._s for c in columns]
        elif len(data) == len(columns):
            if from_dict:
                series_map = {s.name(): s for s in data}
                if all((col in series_map) for col in columns):
                    return [series_map[col] for col in columns]
            for i, c in enumerate(columns):
                if c != data[i].name():
                    data[i] = data[i].clone()
                    data[i].rename(c)
            return data
        else:
            raise ValueError("Dimensions of columns arg must match data dimensions.")


def _post_apply_columns(
    pydf: PyDataFrame,
    columns: SchemaDefinition | None,
    structs: dict[str, Struct] | None = None,
    schema_overrides: SchemaDict | None = None,
) -> PyDataFrame:
    """Apply 'columns' param _after_ PyDataFrame creation (if no alternative)."""
    pydf_columns, pydf_dtypes = pydf.columns(), pydf.dtypes()
    columns, dtypes = _unpack_schema(
        (columns or pydf_columns), schema_overrides=schema_overrides
    )
    column_subset: list[str] = []
    if columns != pydf_columns:
        if len(columns) < len(pydf_columns) and columns == pydf_columns[: len(columns)]:
            column_subset = columns
        else:
            pydf.set_column_names(columns)

    column_casts = []
    for i, col in enumerate(columns):
        if dtypes.get(col) == Categorical != pydf_dtypes[i]:
            column_casts.append(F.col(col).cast(Categorical)._pyexpr)
        elif structs and col in structs and structs[col] != pydf_dtypes[i]:
            column_casts.append(F.col(col).cast(structs[col])._pyexpr)
        elif dtypes.get(col) not in (None, Unknown) and dtypes[col] != pydf_dtypes[i]:
            column_casts.append(F.col(col).cast(dtypes[col])._pyexpr)

    if column_casts or column_subset:
        pydf = pydf.lazy()
        if column_casts:
            pydf = pydf.with_columns(column_casts)
        if column_subset:
            pydf = pydf.select([F.col(col)._pyexpr for col in column_subset])
        pydf = pydf.collect()

    return pydf


def _unpack_schema(
    schema: SchemaDefinition | None,
    schema_overrides: SchemaDict | None = None,
    n_expected: int | None = None,
    lookup_names: Iterable[str] | None = None,
    include_overrides_in_columns: bool = False,
) -> tuple[list[str], SchemaDict]:
    """
    Unpack column names and create dtype lookup.

    Works for any (name, dtype) pairs or schema dict input,
    overriding any inferred dtypes with explicit dtypes if supplied.
    """
    if isinstance(schema, dict):
        schema = list(schema.items())
    column_names = [
        (col or f"column_{i}") if isinstance(col, str) else col[0]
        for i, col in enumerate(schema or [])
    ]
    if not column_names and n_expected:
        column_names = [f"column_{i}" for i in range(n_expected)]
    lookup = {
        col: name for col, name in zip_longest(column_names, lookup_names or []) if name
    }
    column_dtypes = {
        lookup.get(col[0], col[0]): col[1]
        for col in (schema or [])
        if not isinstance(col, str) and col[1] is not None
    }
    if schema_overrides:
        column_dtypes.update(schema_overrides)
        if schema and include_overrides_in_columns:
            column_names = column_names + [
                col for col in column_dtypes if col not in column_names
            ]
    for col, dtype in column_dtypes.items():
        if not is_polars_dtype(dtype, include_unknown=True) and dtype is not None:
            column_dtypes[col] = py_type_to_dtype(dtype)

    return (
        column_names,  # type: ignore[return-value]
        column_dtypes,
    )


def _expand_dict_scalars(
    data: Mapping[str, Sequence[object] | Mapping[str, Sequence[object]] | Series],
    schema_overrides: SchemaDict | None = None,
    order: Sequence[str] | None = None,
    nan_to_null: bool = False,
) -> dict[str, Series]:
    """Expand any scalar values in dict data (propagate literal as array)."""
    updated_data = {}
    if data:
        dtypes = schema_overrides or {}
        array_len = max((arrlen(val) or 0) for val in data.values())
        if array_len > 0:
            for name, val in data.items():
                dtype = dtypes.get(name)
                if isinstance(val, dict) and dtype != Struct:
                    updated_data[name] = pl.DataFrame(val).to_struct(name)

                elif isinstance(val, pl.Series):
                    s = val.rename(name) if name != val.name else val
                    if dtype and dtype != s.dtype:
                        s = s.cast(dtype)
                    updated_data[name] = s

                elif arrlen(val) is not None or _is_generator(val):
                    updated_data[name] = pl.Series(
                        name=name, values=val, dtype=dtype, nan_to_null=nan_to_null
                    )
                elif val is None or isinstance(  # type: ignore[redundant-expr]
                    val, (int, float, str, bool, date, datetime, time, timedelta)
                ):
                    updated_data[name] = pl.Series(
                        name=name, values=[val], dtype=dtype
                    ).extend_constant(val, array_len - 1)
                else:
                    updated_data[name] = pl.Series(
                        name=name, values=[val] * array_len, dtype=dtype
                    )

        elif all((arrlen(val) == 0) for val in data.values()):
            for name, val in data.items():
                updated_data[name] = pl.Series(name, values=val, dtype=dtypes.get(name))

        elif all((arrlen(val) is None) for val in data.values()):
            for name, val in data.items():
                updated_data[name] = pl.Series(
                    name,
                    values=(val if _is_generator(val) else [val]),
                    dtype=dtypes.get(name),
                )
    if order and list(updated_data) != order:
        return {col: updated_data.pop(col) for col in order}
    return updated_data


def dict_to_pydf(
    data: Mapping[str, Sequence[object] | Mapping[str, Sequence[object]] | Series],
    schema: SchemaDefinition | None = None,
    schema_overrides: SchemaDict | None = None,
    nan_to_null: bool = False,
) -> PyDataFrame:
    """Construct a PyDataFrame from a dictionary of sequences."""
    if isinstance(schema, dict) and data:
        if not all((col in schema) for col in data):
            raise ValueError(
                "The given column-schema names do not match the data dictionary"
            )
        data = {col: data[col] for col in schema}

    column_names, schema_overrides = _unpack_schema(
        schema, lookup_names=data.keys(), schema_overrides=schema_overrides
    )
    if not column_names:
        column_names = list(data)

    if data and _NUMPY_AVAILABLE:
        # if there are 3 or more numpy arrays of sufficient size, we multi-thread:
        count_numpy = sum(
            int(
                _check_for_numpy(val)
                and isinstance(val, np.ndarray)
                and len(val) > 1000
            )
            for val in data.values()
        )
        if count_numpy >= 3:
            # yes, multi-threading was easier in python here; we cannot have multiple
            # threads running python and release the gil in pyo3 (it will deadlock).

            # (note: 'dummy' is threaded)
            import multiprocessing.dummy

            pool_size = threadpool_size()
            with multiprocessing.dummy.Pool(pool_size) as pool:
                data = dict(
                    zip(
                        column_names,
                        pool.map(
                            lambda t: pl.Series(t[0], t[1])
                            if isinstance(t[1], np.ndarray)
                            else t[1],
                            [(k, v) for k, v in data.items()],
                        ),
                    )
                )

    if not data and schema_overrides:
        data_series = [
            pl.Series(
                name, [], dtype=schema_overrides.get(name), nan_to_null=nan_to_null
            )._s
            for name in column_names
        ]
    else:
        data_series = [
            s._s
            for s in _expand_dict_scalars(
                data, schema_overrides, nan_to_null=nan_to_null
            ).values()
        ]

    data_series = _handle_columns_arg(data_series, columns=column_names, from_dict=True)
    return PyDataFrame(data_series)


def sequence_to_pydf(
    data: Sequence[Any],
    schema: SchemaDefinition | None = None,
    schema_overrides: SchemaDict | None = None,
    orient: Orientation | None = None,
    infer_schema_length: int | None = N_INFER_DEFAULT,
) -> PyDataFrame:
    """Construct a PyDataFrame from a sequence."""
    if len(data) == 0:
        return dict_to_pydf({}, schema=schema, schema_overrides=schema_overrides)

    return _sequence_to_pydf_dispatcher(
        data[0],
        data=data,
        schema=schema,
        schema_overrides=schema_overrides,
        orient=orient,
        infer_schema_length=infer_schema_length,
    )


def _sequence_of_series_to_pydf(
    first_element: Series,
    data: Sequence[Any],
    schema: SchemaDefinition | None,
    schema_overrides: SchemaDict | None,
    **kwargs: Any,
) -> PyDataFrame:
    series_names = [s.name for s in data]
    column_names, schema_overrides = _unpack_schema(
        schema or series_names,
        schema_overrides=schema_overrides,
        n_expected=len(data),
    )
    data_series: list[PySeries] = []
    for i, s in enumerate(data):
        if not s.name:
            s = s.alias(column_names[i])
        new_dtype = schema_overrides.get(column_names[i])
        if new_dtype and new_dtype != s.dtype:
            s = s.cast(new_dtype)
        data_series.append(s._s)

    data_series = _handle_columns_arg(data_series, columns=column_names)
    return PyDataFrame(data_series)


@singledispatch
def _sequence_to_pydf_dispatcher(
    first_element: Any,
    data: Sequence[Any],
    schema: SchemaDefinition | None,
    schema_overrides: SchemaDict | None,
    orient: Orientation | None,
    infer_schema_length: int | None,
) -> PyDataFrame:
    # note: ONLY python-native data should participate in singledispatch registration
    # via top-level decorators. third-party libraries (such as numpy/pandas) should
    # first be identified inline (here) and THEN registered for dispatch dynamically
    # so as not to break lazy-loading behaviour.

    common_params = {
        "data": data,
        "schema": schema,
        "schema_overrides": schema_overrides,
        "orient": orient,
        "infer_schema_length": infer_schema_length,
    }

    to_pydf: Callable[..., PyDataFrame]
    register_with_singledispatch = True

    if isinstance(first_element, Generator):
        to_pydf = _sequence_of_sequence_to_pydf
        data = [list(row) for row in data]
        first_element = data[0]
        register_with_singledispatch = False

    elif isinstance(first_element, pl.Series):
        to_pydf = _sequence_of_series_to_pydf

    elif _check_for_numpy(first_element) and isinstance(first_element, np.ndarray):
        to_pydf = _sequence_of_numpy_to_pydf

    elif _check_for_pandas(first_element) and isinstance(
        first_element, (pd.Series, pd.DatetimeIndex)
    ):
        to_pydf = _sequence_of_pandas_to_pydf

    elif dataclasses.is_dataclass(first_element):
        to_pydf = _dataclasses_or_models_to_pydf

    elif is_pydantic_model(first_element):
        to_pydf = partial(_dataclasses_or_models_to_pydf, pydantic_model=True)
    else:
        to_pydf = _sequence_of_elements_to_pydf

    if register_with_singledispatch:
        _sequence_to_pydf_dispatcher.register(type(first_element), to_pydf)

    common_params["first_element"] = first_element
    return to_pydf(**common_params)


@_sequence_to_pydf_dispatcher.register(list)
def _sequence_of_sequence_to_pydf(
    first_element: Sequence[Any] | np.ndarray[Any, Any],
    data: Sequence[Any],
    schema: SchemaDefinition | None,
    schema_overrides: SchemaDict | None,
    orient: Orientation | None,
    infer_schema_length: int | None,
) -> PyDataFrame:
    if orient is None:
        # note: limit type-checking to smaller data; larger values are much more
        # likely to indicate col orientation anyway, so minimise extra checks.
        if len(first_element) > 1000:
            orient = "col" if schema and len(schema) == len(data) else "row"
        elif (schema is not None and len(schema) == len(data)) or not schema:
            # check if element types in the first 'row' resolve to a single dtype.
            row_types = {type(value) for value in first_element if value is not None}
            if int in row_types and float in row_types:
                row_types.discard(int)
            orient = "col" if len(row_types) == 1 else "row"
        else:
            orient = "row"

    if orient == "row":
        column_names, schema_overrides = _unpack_schema(
            schema, schema_overrides=schema_overrides, n_expected=len(first_element)
        )
        local_schema_override = (
            include_unknowns(schema_overrides, column_names) if schema_overrides else {}
        )
        if column_names and len(first_element) != len(column_names):
            raise ShapeError("The row data does not match the number of columns")

        unpack_nested = False
        for col, tp in local_schema_override.items():
            if tp == Categorical:
                local_schema_override[col] = Utf8
            elif not unpack_nested and (tp.base_type() in (Unknown, Struct)):
                unpack_nested = contains_nested(
                    getattr(first_element, col, None).__class__, is_namedtuple
                )

        if unpack_nested:
            dicts = [nt_unpack(d) for d in data]
            pydf = PyDataFrame.read_dicts(dicts, infer_schema_length)
        else:
            pydf = PyDataFrame.read_rows(
                data,
                infer_schema_length,
                local_schema_override or None,
            )
        if column_names or schema_overrides:
            pydf = _post_apply_columns(
                pydf, column_names, schema_overrides=schema_overrides
            )
        return pydf

    if orient == "col" or orient is None:
        column_names, schema_overrides = _unpack_schema(
            schema, schema_overrides=schema_overrides, n_expected=len(data)
        )
        data_series: list[PySeries] = [
            pl.Series(
                column_names[i], element, schema_overrides.get(column_names[i])
            )._s
            for i, element in enumerate(data)
        ]
        return PyDataFrame(data_series)

    raise ValueError(
        f"orient must be one of {{'col', 'row', None}}, got {orient} instead."
    )


@_sequence_to_pydf_dispatcher.register(tuple)
def _sequence_of_tuple_to_pydf(
    first_element: tuple[Any, ...],
    data: Sequence[Any],
    schema: SchemaDefinition | None,
    schema_overrides: SchemaDict | None,
    orient: Orientation | None,
    infer_schema_length: int | None,
) -> PyDataFrame:
    # infer additional meta information if named tuple
    if is_namedtuple(first_element.__class__):
        if schema is None:
            schema = first_element._fields  # type: ignore[attr-defined]
            if len(first_element.__annotations__) == len(schema):
                schema = [
                    (name, py_type_to_dtype(tp, raise_unmatched=False))
                    for name, tp in first_element.__annotations__.items()
                ]
        elif orient is None:
            orient = "row"

    # ...then defer to generic sequence processing
    return _sequence_of_sequence_to_pydf(
        first_element,
        data=data,
        schema=schema,
        schema_overrides=schema_overrides,
        orient=orient,
        infer_schema_length=infer_schema_length,
    )


@_sequence_to_pydf_dispatcher.register(dict)
def _sequence_of_dict_to_pydf(
    first_element: Any,
    data: Sequence[Any],
    schema: SchemaDefinition | None,
    schema_overrides: SchemaDict | None,
    infer_schema_length: int | None,
    **kwargs: Any,
) -> PyDataFrame:
    column_names, schema_overrides = _unpack_schema(
        schema, schema_overrides=schema_overrides
    )
    dicts_schema = (
        include_unknowns(schema_overrides, column_names or list(schema_overrides))
        if schema_overrides and column_names
        else None
    )
    pydf = PyDataFrame.read_dicts(data, infer_schema_length, dicts_schema)

    if column_names and set(column_names).intersection(pydf.columns()):
        column_names = []
    if column_names or schema_overrides:
        pydf = _post_apply_columns(
            pydf,
            columns=column_names,
            schema_overrides=schema_overrides,
        )
    return pydf


@_sequence_to_pydf_dispatcher.register(str)
def _sequence_of_elements_to_pydf(
    first_element: Any,
    data: Sequence[Any],
    schema: SchemaDefinition | None,
    schema_overrides: SchemaDict | None,
    **kwargs: Any,
) -> PyDataFrame:
    column_names, schema_overrides = _unpack_schema(
        schema, schema_overrides=schema_overrides, n_expected=1
    )
    data_series: list[PySeries] = [
        pl.Series(column_names[0], data, schema_overrides.get(column_names[0]))._s
    ]
    data_series = _handle_columns_arg(data_series, columns=column_names)
    return PyDataFrame(data_series)


def _sequence_of_numpy_to_pydf(
    first_element: np.ndarray[Any, Any],
    **kwargs: Any,
) -> PyDataFrame:
    to_pydf = (
        _sequence_of_sequence_to_pydf
        if first_element.ndim == 1
        else _sequence_of_elements_to_pydf
    )
    return to_pydf(first_element, **kwargs)  # type: ignore[operator]


def _sequence_of_pandas_to_pydf(
    first_element: pd.Series | pd.DatetimeIndex,
    data: Sequence[Any],
    schema: SchemaDefinition | None,
    schema_overrides: SchemaDict | None,
    **kwargs: Any,
) -> PyDataFrame:
    if schema is None:
        column_names: list[str] = []
    else:
        column_names, schema_overrides = _unpack_schema(
            schema, schema_overrides=schema_overrides, n_expected=1
        )

    schema_overrides = schema_overrides or {}
    data_series: list[PySeries] = []
    for i, s in enumerate(data):
        name = column_names[i] if column_names else s.name
        dtype = schema_overrides.get(name, None)
        pyseries = pandas_to_pyseries(name=name, values=s)
        if dtype is not None and dtype != pyseries.dtype():
            pyseries = pyseries.cast(dtype, strict=True)
        data_series.append(pyseries)

    return PyDataFrame(data_series)


def _dataclasses_or_models_to_pydf(
    first_element: Any,
    data: Sequence[Any],
    schema: SchemaDefinition | None,
    schema_overrides: SchemaDict | None,
    infer_schema_length: int | None,
    **kwargs: Any,
) -> PyDataFrame:
    """Initialise DataFrame from python dataclass and/or pydantic model objects."""
    from dataclasses import asdict, astuple

    from_model = kwargs.get("pydantic_model")
    unpack_nested = False
    if schema:
        column_names, schema_overrides = _unpack_schema(schema, schema_overrides)
        schema_override = {
            col: schema_overrides.get(col, Unknown) for col in column_names
        }
    else:
        column_names = []
        schema_override = {
            col: (py_type_to_dtype(tp, raise_unmatched=False) or Unknown)
            for col, tp in type_hints(first_element.__class__).items()
            if col != "__slots__"
        }
        if schema_overrides:
            schema_override.update(schema_overrides)
        elif not from_model:
            dc_fields = set(asdict(first_element))
            schema_overrides = schema_override = {
                nm: tp for nm, tp in schema_override.items() if nm in dc_fields
            }
        else:
            schema_overrides = schema_override

    for col, tp in schema_override.items():
        if tp == Categorical:
            schema_override[col] = Utf8
        elif not unpack_nested and (tp.base_type() in (Unknown, Struct)):
            unpack_nested = contains_nested(
                getattr(first_element, col, None),
                is_pydantic_model if from_model else dataclasses.is_dataclass,  # type: ignore[arg-type]
            )

    if unpack_nested:
        if from_model:
            dicts = (
                [md.model_dump(mode="python") for md in data]
                if hasattr(first_element, "model_dump")
                else [md.dict() for md in data]
            )
        else:
            dicts = [asdict(md) for md in data]
        pydf = PyDataFrame.read_dicts(dicts, infer_schema_length)
    else:
        rows = (
            [tuple(md.__dict__.values()) for md in data]
            if from_model
            else [astuple(dc) for dc in data]
        )
        pydf = PyDataFrame.read_rows(rows, infer_schema_length, schema_override or None)

    if schema_override:
        structs = {c: tp for c, tp in schema_override.items() if isinstance(tp, Struct)}
        pydf = _post_apply_columns(pydf, column_names, structs, schema_overrides)

    return pydf


def numpy_to_pydf(
    data: np.ndarray[Any, Any],
    schema: SchemaDefinition | None = None,
    schema_overrides: SchemaDict | None = None,
    orient: Orientation | None = None,
    nan_to_null: bool = False,
) -> PyDataFrame:
    """Construct a PyDataFrame from a numpy ndarray (including structured ndarrays)."""
    shape = data.shape

    if data.dtype.names is not None:
        structured_array, orient = True, "col"
        record_names = list(data.dtype.names)
        n_columns = len(record_names)
        for nm in record_names:
            shape = data[nm].shape
            if len(data[nm].shape) > 2:
                raise ValueError(
                    f"Cannot create DataFrame from structured array with elements > 2D; shape[{nm!r}] = {shape}"
                )
        if not schema:
            schema = record_names
    else:
        # Unpack columns
        structured_array, record_names = False, []
        if shape == (0,):
            n_columns = 0

        elif len(shape) == 1:
            n_columns = 1

        elif len(shape) == 2:
            if orient is None and schema is None:
                # default convention; first axis is rows, second axis is columns
                n_columns = shape[1]
                orient = "row"

            elif orient is None and schema is not None:
                # infer orientation from 'schema' param
                if len(schema) == shape[0]:
                    orient = "col"
                    n_columns = shape[0]
                else:
                    orient = "row"
                    n_columns = shape[1]

            elif orient == "row":
                n_columns = shape[1]
            elif orient == "col":
                n_columns = shape[0]
            else:
                raise ValueError(
                    f"orient must be one of {{'col', 'row', None}}; found {orient!r} instead."
                )
        else:
            raise ValueError(
                f"Cannot create DataFrame from array with more than two dimensions; shape = {shape}"
            )

    if schema is not None and len(schema) != n_columns:
        raise ValueError("Dimensions of 'schema' arg must match data dimensions.")

    column_names, schema_overrides = _unpack_schema(
        schema, schema_overrides=schema_overrides, n_expected=n_columns
    )

    # Convert data to series
    if structured_array:
        data_series = [
            pl.Series(
                name=series_name,
                values=data[record_name],
                dtype=schema_overrides.get(record_name),
                nan_to_null=nan_to_null,
            )._s
            for series_name, record_name in zip(column_names, record_names)
        ]
    elif shape == (0,):
        data_series = []

    elif len(shape) == 1:
        data_series = [
            pl.Series(
                name=column_names[0],
                values=data,
                dtype=schema_overrides.get(column_names[0]),
                nan_to_null=nan_to_null,
            )._s
        ]
    else:
        if orient == "row":
            data_series = [
                pl.Series(
                    name=column_names[i],
                    values=data[:, i],
                    dtype=schema_overrides.get(column_names[i]),
                    nan_to_null=nan_to_null,
                )._s
                for i in range(n_columns)
            ]
        else:
            data_series = [
                pl.Series(
                    name=column_names[i],
                    values=data[i],
                    dtype=schema_overrides.get(column_names[i]),
                    nan_to_null=nan_to_null,
                )._s
                for i in range(n_columns)
            ]

    data_series = _handle_columns_arg(data_series, columns=column_names)
    return PyDataFrame(data_series)


def arrow_to_pydf(
    data: pa.Table,
    schema: SchemaDefinition | None = None,
    schema_overrides: SchemaDict | None = None,
    rechunk: bool = True,
) -> PyDataFrame:
    """Construct a PyDataFrame from an Arrow Table."""
    original_schema = schema
    column_names, schema_overrides = _unpack_schema(
        (schema or data.column_names), schema_overrides=schema_overrides
    )
    try:
        if column_names != data.column_names:
            data = data.rename_columns(column_names)
    except pa.lib.ArrowInvalid as e:
        raise ValueError("Dimensions of columns arg must match data dimensions.") from e

    data_dict = {}
    # dictionaries cannot be built in different batches (categorical does not allow
    # that) so we rechunk them and create them separately.
    dictionary_cols = {}
    # struct columns don't work properly if they contain multiple chunks.
    struct_cols = {}
    names = []
    for i, column in enumerate(data):
        # extract the name before casting
        name = f"column_{i}" if column._name is None else column._name
        names.append(name)

        column = coerce_arrow(column)
        if pa.types.is_dictionary(column.type):
            ps = arrow_to_pyseries(name, column, rechunk)
            dictionary_cols[i] = wrap_s(ps)
        elif isinstance(column.type, pa.StructType) and column.num_chunks > 1:
            ps = arrow_to_pyseries(name, column, rechunk)
            struct_cols[i] = wrap_s(ps)
        else:
            data_dict[name] = column

    if len(data_dict) > 0:
        tbl = pa.table(data_dict)

        # path for table without rows that keeps datatype
        if tbl.shape[0] == 0:
            pydf = pl.DataFrame(
                [pl.Series(name, c) for (name, c) in zip(tbl.column_names, tbl.columns)]
            )._df
        else:
            pydf = PyDataFrame.from_arrow_record_batches(tbl.to_batches())
    else:
        pydf = pl.DataFrame([])._df
    if rechunk:
        pydf = pydf.rechunk()

    reset_order = False
    if len(dictionary_cols) > 0:
        df = wrap_df(pydf)
        df = df.with_columns([F.lit(s).alias(s.name) for s in dictionary_cols.values()])
        reset_order = True

    if len(struct_cols) > 0:
        df = wrap_df(pydf)
        df = df.with_columns([F.lit(s).alias(s.name) for s in struct_cols.values()])
        reset_order = True

    if reset_order:
        df = df[names]
        pydf = df._df

    if column_names != original_schema and (schema_overrides or original_schema):
        pydf = _post_apply_columns(
            pydf, original_schema, schema_overrides=schema_overrides
        )
    elif schema_overrides:
        for col, dtype in zip(pydf.columns(), pydf.dtypes()):
            override_dtype = schema_overrides.get(col)
            if override_dtype is not None and dtype != override_dtype:
                pydf = _post_apply_columns(
                    pydf, original_schema, schema_overrides=schema_overrides
                )
                break

    return pydf


def series_to_pydf(
    data: Series,
    schema: SchemaDefinition | None = None,
    schema_overrides: SchemaDict | None = None,
) -> PyDataFrame:
    """Construct a PyDataFrame from a Polars Series."""
    data_series = [data._s]
    series_name = [s.name() for s in data_series]
    column_names, schema_overrides = _unpack_schema(
        schema or series_name, schema_overrides=schema_overrides, n_expected=1
    )
    if schema_overrides:
        new_dtype = list(schema_overrides.values())[0]
        if new_dtype != data.dtype:
            data_series[0] = data_series[0].cast(new_dtype, True)

    data_series = _handle_columns_arg(data_series, columns=column_names)
    return PyDataFrame(data_series)


def iterable_to_pydf(
    data: Iterable[Any],
    schema: SchemaDefinition | None = None,
    schema_overrides: SchemaDict | None = None,
    orient: Orientation | None = None,
    chunk_size: int | None = None,
    infer_schema_length: int | None = N_INFER_DEFAULT,
) -> PyDataFrame:
    """Construct a PyDataFrame from an iterable/generator."""
    original_schema = schema
    column_names: list[str] = []
    dtypes_by_idx: dict[int, PolarsDataType] = {}
    if schema is not None:
        column_names, schema_overrides = _unpack_schema(
            schema, schema_overrides=schema_overrides
        )
    elif schema_overrides:
        _, schema_overrides = _unpack_schema(schema, schema_overrides=schema_overrides)

    if not isinstance(data, Generator):
        data = iter(data)

    if orient == "col":
        if column_names and schema_overrides:
            dtypes_by_idx = {
                idx: schema_overrides.get(col, Unknown)
                for idx, col in enumerate(column_names)
            }

        return pl.DataFrame(
            {
                (column_names[idx] if column_names else f"column_{idx}"): pl.Series(
                    coldata, dtype=dtypes_by_idx.get(idx)
                )
                for idx, coldata in enumerate(data)
            }
        )._df

    def to_frame_chunk(values: list[Any], schema: SchemaDefinition | None) -> DataFrame:
        return pl.DataFrame(
            data=values,
            schema=schema,
            orient="row",
            infer_schema_length=infer_schema_length,
        )

    n_chunks = 0
    n_chunk_elems = 1_000_000

    if chunk_size:
        adaptive_chunk_size = chunk_size
    elif column_names:
        adaptive_chunk_size = n_chunk_elems // len(column_names)
    else:
        adaptive_chunk_size = None

    df: DataFrame = None  # type: ignore[assignment]
    chunk_size = max(
        (infer_schema_length or 0),
        (adaptive_chunk_size or 1000),
    )
    while True:
        values = list(islice(data, chunk_size))
        if not values:
            break
        frame_chunk = to_frame_chunk(values, original_schema)
        if df is None:
            df = frame_chunk
            if not original_schema:
                original_schema = list(df.schema.items())
            if chunk_size != adaptive_chunk_size:
                chunk_size = adaptive_chunk_size = n_chunk_elems // len(df.columns)
        else:
            df.vstack(frame_chunk, in_place=True)
            n_chunks += 1

    if df is None:
        df = to_frame_chunk([], original_schema)

    return (df.rechunk() if n_chunks > 0 else df)._df


def pandas_has_default_index(df: pd.DataFrame) -> bool:
    """Identify if the pandas frame only has a default (or equivalent) index."""
    from pandas.core.indexes.range import RangeIndex

    index_cols = df.index.names

    if len(index_cols) > 1 or index_cols not in ([None], [""]):
        # not default: more than one index, or index is named
        return False
    elif df.index.equals(RangeIndex(start=0, stop=len(df), step=1)):
        # is default: simple range index
        return True
    else:
        # finally, is the index _equivalent_ to a default unnamed
        # integer index with frame data that was previously sorted
        return (
            str(df.index.dtype).startswith("int")
            and (df.index.sort_values() == np.arange(len(df))).all()
        )


def pandas_to_pydf(
    data: pd.DataFrame,
    schema: SchemaDefinition | None = None,
    schema_overrides: SchemaDict | None = None,
    rechunk: bool = True,
    nan_to_null: bool = True,
    include_index: bool = False,
) -> PyDataFrame:
    """Construct a PyDataFrame from a pandas DataFrame."""
    arrow_dict = {}
    length = data.shape[0]

    if include_index and not pandas_has_default_index(data):
        for idxcol in data.index.names:
            arrow_dict[str(idxcol)] = _pandas_series_to_arrow(
                data.index.get_level_values(idxcol),
                nan_to_null=nan_to_null,
                length=length,
            )

    for col in data.columns:
        arrow_dict[str(col)] = _pandas_series_to_arrow(
            data[col], nan_to_null=nan_to_null, length=length
        )

    arrow_table = pa.table(arrow_dict)
    return arrow_to_pydf(
        arrow_table, schema=schema, schema_overrides=schema_overrides, rechunk=rechunk
    )


def coerce_arrow(array: pa.Array, rechunk: bool = True) -> pa.Array:
    import pyarrow.compute as pc

    if hasattr(array, "num_chunks") and array.num_chunks > 1 and rechunk:
        # small integer keys can often not be combined, so let's already cast
        # to the uint32 used by polars
        if pa.types.is_dictionary(array.type) and (
            pa.types.is_int8(array.type.index_type)
            or pa.types.is_uint8(array.type.index_type)
            or pa.types.is_int16(array.type.index_type)
            or pa.types.is_uint16(array.type.index_type)
            or pa.types.is_int32(array.type.index_type)
        ):
            array = pc.cast(
                array, pa.dictionary(pa.uint32(), pa.large_string())
            ).combine_chunks()
    return array
