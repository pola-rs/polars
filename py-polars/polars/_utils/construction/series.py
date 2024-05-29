from __future__ import annotations

import contextlib
import warnings
from datetime import date, datetime, time, timedelta
from decimal import Decimal as PyDecimal
from itertools import islice
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Iterable,
    Iterator,
    Sequence,
)

import polars._reexport as pl
import polars._utils.construction as plc
from polars._utils.construction.utils import (
    get_first_non_none,
    is_namedtuple,
    is_pydantic_model,
    is_simple_numpy_backed_pandas_series,
)
from polars._utils.various import (
    find_stacklevel,
    range_to_series,
)
from polars._utils.wrap import wrap_s
from polars.datatypes import (
    INTEGER_DTYPES,
    TEMPORAL_DTYPES,
    Array,
    Boolean,
    Categorical,
    Date,
    Datetime,
    Decimal,
    Duration,
    Enum,
    List,
    Null,
    Object,
    Struct,
    Time,
    Unknown,
    dtype_to_py_type,
    is_polars_dtype,
    numpy_char_code_to_dtype,
    py_type_to_dtype,
)
from polars.datatypes.constructor import (
    numpy_type_to_constructor,
    numpy_values_and_dtype,
    polars_type_to_constructor,
    py_type_to_constructor,
)
from polars.dependencies import (
    _PYARROW_AVAILABLE,
    _check_for_numpy,
    dataclasses,
)
from polars.dependencies import numpy as np
from polars.dependencies import pandas as pd
from polars.dependencies import pyarrow as pa
from polars.exceptions import TimeZoneAwareConstructorWarning

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import PySeries

if TYPE_CHECKING:
    from polars import DataFrame, Series
    from polars.dependencies import pandas as pd
    from polars.type_aliases import PolarsDataType


def sequence_to_pyseries(
    name: str,
    values: Sequence[Any],
    dtype: PolarsDataType | None = None,
    *,
    strict: bool = True,
    nan_to_null: bool = False,
) -> PySeries:
    """Construct a PySeries from a sequence."""
    python_dtype: type | None = None

    if isinstance(values, range):
        return range_to_series(name, values, dtype=dtype)._s

    # empty sequence
    if len(values) == 0 and dtype is None:
        # if dtype for empty sequence could be guessed
        # (e.g comparisons between self and other), default to Null
        dtype = Null

    # lists defer to subsequent handling; identify nested type
    elif dtype in (List, Array):
        python_dtype = list

    # infer temporal type handling
    py_temporal_types = {date, datetime, timedelta, time}
    pl_temporal_types = {Date, Datetime, Duration, Time}

    value = get_first_non_none(values)
    if value is not None:
        if (
            dataclasses.is_dataclass(value)
            or is_pydantic_model(value)
            or is_namedtuple(value.__class__)
        ) and dtype != Object:
            return pl.DataFrame(values).to_struct(name)._s
        elif isinstance(value, range) and dtype is None:
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
        and is_polars_dtype(dtype)
        and not dtype.is_nested()
        and dtype != Unknown
        and (python_dtype is None)
    ):
        constructor = polars_type_to_constructor(dtype)
        pyseries = _construct_series_with_fallbacks(
            constructor, name, values, dtype, strict=strict
        )
        if dtype in (
            Date,
            Datetime,
            Duration,
            Time,
            Categorical,
            Boolean,
            Enum,
            Decimal,
        ):
            if pyseries.dtype() != dtype:
                pyseries = pyseries.cast(dtype, strict=strict)
        return pyseries

    elif dtype == Struct:
        struct_schema = dtype.to_schema() if isinstance(dtype, Struct) else None
        empty = {}  # type: ignore[var-annotated]
        return plc.sequence_to_pydf(
            data=[(empty if v is None else v) for v in values],
            schema=struct_schema,
            orient="row",
        ).to_struct(name)

    if python_dtype is None:
        if value is None:
            constructor = polars_type_to_constructor(Null)
            return constructor(name, values, strict)

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
        if values_dtype is not None and values_dtype.is_float():
            msg = f"'float' object cannot be interpreted as a {python_dtype.__name__!r}"
            raise TypeError(
                # we do not accept float values as temporal; if this is
                # required, the caller should explicitly cast to int first.
                msg
            )

        # We use the AnyValue builder to create the datetime array
        # We store the values internally as UTC and set the timezone
        py_series = PySeries.new_from_any_values(name, values, strict)

        time_unit = getattr(dtype, "time_unit", None)
        time_zone = getattr(dtype, "time_zone", None)

        if time_unit is None or values_dtype == Date:
            s = wrap_s(py_series)
        else:
            s = wrap_s(py_series).dt.cast_time_unit(time_unit)

        if (values_dtype == Date) & (dtype == Datetime):
            return (
                s.cast(Datetime(time_unit or "us"))
                .dt.replace_time_zone(
                    time_zone,
                    ambiguous="raise" if strict else "null",
                    non_existent="raise" if strict else "null",
                )
                ._s
            )

        if (dtype == Datetime) and (value.tzinfo is not None or time_zone is not None):
            values_tz = str(value.tzinfo) if value.tzinfo is not None else None
            dtype_tz = time_zone
            if values_tz is not None and (dtype_tz is not None and dtype_tz != "UTC"):
                msg = (
                    "time-zone-aware datetimes are converted to UTC"
                    "\n\nPlease either drop the time zone from the dtype, or set it to 'UTC'."
                    " To convert to a different time zone, please use `.dt.convert_time_zone`."
                )
                raise ValueError(msg)
            if values_tz != "UTC" and dtype_tz is None:
                warnings.warn(
                    "Constructing a Series with time-zone-aware "
                    "datetimes results in a Series with UTC time zone. "
                    "To silence this warning, you can filter "
                    "warnings of class TimeZoneAwareConstructorWarning, or "
                    "set 'UTC' as the time zone of your datatype.",
                    TimeZoneAwareConstructorWarning,
                    stacklevel=find_stacklevel(),
                )
            return s.dt.replace_time_zone(
                dtype_tz or "UTC",
                ambiguous="raise" if strict else "null",
                non_existent="raise" if strict else "null",
            )._s
        return s._s

    elif (
        _check_for_numpy(value)
        and isinstance(value, np.ndarray)
        and len(value.shape) == 1
    ):
        n_elems = len(value)
        if all(len(v) == n_elems for v in values):
            # can take (much) faster path if all lists are the same length
            return numpy_to_pyseries(
                name,
                np.vstack(values),
                strict=strict,
                nan_to_null=nan_to_null,
            )
        else:
            return PySeries.new_series_list(
                name,
                [
                    numpy_to_pyseries("", v, strict=strict, nan_to_null=nan_to_null)
                    for v in values
                ],
                strict,
            )

    elif python_dtype in (list, tuple):
        if dtype is None:
            return PySeries.new_from_any_values(name, values, strict=strict)
        elif dtype == Object:
            return PySeries.new_object(name, values, strict)
        else:
            if (inner_dtype := getattr(dtype, "inner", None)) is not None:
                pyseries_list = [
                    None
                    if value is None
                    else sequence_to_pyseries(
                        "",
                        value,
                        inner_dtype,
                        strict=strict,
                        nan_to_null=nan_to_null,
                    )
                    for value in values
                ]
                pyseries = PySeries.new_series_list(name, pyseries_list, strict)
            else:
                pyseries = PySeries.new_from_any_values_and_dtype(
                    name, values, dtype, strict=strict
                )
            if dtype != pyseries.dtype():
                pyseries = pyseries.cast(dtype, strict=False)
            return pyseries

    elif python_dtype == pl.Series:
        return PySeries.new_series_list(
            name, [v._s if v is not None else None for v in values], strict
        )

    elif python_dtype == PySeries:
        return PySeries.new_series_list(name, values, strict)
    else:
        constructor = py_type_to_constructor(python_dtype)
        if constructor == PySeries.new_object:
            try:
                srs = PySeries.new_from_any_values(name, values, strict)
                if _check_for_numpy(python_dtype, check_type=False) and isinstance(
                    np.bool_(True), np.generic
                ):
                    dtype = numpy_char_code_to_dtype(np.dtype(python_dtype).char)
                    return srs.cast(dtype, strict=strict)
                else:
                    return srs

            except RuntimeError:
                return PySeries.new_from_any_values(name, values, strict=strict)

        return _construct_series_with_fallbacks(
            constructor, name, values, dtype, strict=strict
        )


def _construct_series_with_fallbacks(
    constructor: Callable[[str, Sequence[Any], bool], PySeries],
    name: str,
    values: Sequence[Any],
    target_dtype: PolarsDataType | None,
    *,
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
                target_dtype not in (INTEGER_DTYPES | TEMPORAL_DTYPES)
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
                raise


def iterable_to_pyseries(
    name: str,
    values: Iterable[Any],
    dtype: PolarsDataType | None = None,
    *,
    chunk_size: int = 1_000_000,
    strict: bool = True,
) -> PySeries:
    """Construct a PySeries from an iterable/generator."""
    if not isinstance(values, (Generator, Iterator)):
        values = iter(values)

    def to_series_chunk(values: list[Any], dtype: PolarsDataType | None) -> Series:
        return pl.Series(
            name=name,
            values=values,
            dtype=dtype,
            strict=strict,
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
            series.append(schunk)
            n_chunks += 1

    if series is None:
        series = to_series_chunk([], dtype)
    if n_chunks > 0:
        series.rechunk(in_place=True)

    return series._s


def pandas_to_pyseries(
    name: str,
    values: pd.Series[Any] | pd.Index[Any] | pd.DatetimeIndex,
    dtype: PolarsDataType | None = None,
    *,
    strict: bool = True,
    nan_to_null: bool = True,
) -> PySeries:
    """Construct a PySeries from a pandas Series or DatetimeIndex."""
    if not name and values.name is not None:
        name = str(values.name)
    if is_simple_numpy_backed_pandas_series(values):
        return pl.Series(
            name, values.to_numpy(), dtype=dtype, nan_to_null=nan_to_null, strict=strict
        )._s
    if not _PYARROW_AVAILABLE:
        msg = (
            "pyarrow is required for converting a pandas series to Polars, "
            "unless it is a simple numpy-backed one "
            "(e.g. 'int64', 'bool', 'float32' - not 'Int64')"
        )
        raise ImportError(msg)
    return arrow_to_pyseries(
        name,
        plc.pandas_series_to_arrow(values, nan_to_null=nan_to_null),
        dtype=dtype,
        strict=strict,
    )


def arrow_to_pyseries(
    name: str,
    values: pa.Array,
    dtype: PolarsDataType | None = None,
    *,
    strict: bool = True,
    rechunk: bool = True,
) -> PySeries:
    """Construct a PySeries from an Arrow array."""
    array = plc.coerce_arrow(values)

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
            # returns the first chunk every time
            if isinstance(array.type, pa.StructType):
                pys = PySeries.from_arrow(name, array.combine_chunks())
            else:
                it = array.iterchunks()
                pys = PySeries.from_arrow(name, next(it))
                for a in it:
                    pys.append(PySeries.from_arrow(name, a))
        elif array.num_chunks == 0:
            pys = PySeries.from_arrow(name, pa.nulls(0, type=array.type))
        else:
            pys = PySeries.from_arrow(name, array.chunks[0])

        if rechunk:
            pys.rechunk(in_place=True)

    return pys.cast(dtype, strict=strict) if dtype is not None else pys


def numpy_to_pyseries(
    name: str,
    values: np.ndarray[Any, Any],
    *,
    strict: bool = True,
    nan_to_null: bool = False,
) -> PySeries:
    """Construct a PySeries from a numpy array."""
    values = np.ascontiguousarray(values)

    if values.ndim == 1:
        values, dtype = numpy_values_and_dtype(values)
        constructor = numpy_type_to_constructor(values, dtype)
        return constructor(
            name, values, nan_to_null if dtype in (np.float32, np.float64) else strict
        )
    # TODO: remove this branch on 1.0.
    # This returns a List whereas we should return an Array type
    elif values.ndim == 2:
        # Optimize by ingesting 1D and reshaping in Rust
        original_shape = values.shape
        values = values.reshape(-1)
        py_s = numpy_to_pyseries(
            name,
            values,
            strict=strict,
            nan_to_null=nan_to_null,
        )
        return wrap_s(py_s).reshape(original_shape)._s
    else:
        original_shape = values.shape
        values = values.reshape(-1)
        py_s = numpy_to_pyseries(
            name,
            values,
            strict=strict,
            nan_to_null=nan_to_null,
        )
        return wrap_s(py_s).reshape(original_shape, Array)._s


def series_to_pyseries(
    name: str | None,
    values: Series,
    *,
    dtype: PolarsDataType | None = None,
    strict: bool = True,
) -> PySeries:
    """Construct a new PySeries from a Polars Series."""
    s = values.clone()
    if dtype is not None and dtype != s.dtype:
        s = s.cast(dtype, strict=strict)
    if name is not None:
        s = s.alias(name)
    return s._s


def dataframe_to_pyseries(
    name: str | None,
    values: DataFrame,
    *,
    dtype: PolarsDataType | None = None,
    strict: bool = True,
) -> PySeries:
    """Construct a new PySeries from a Polars DataFrame."""
    if values.width > 1:
        name = name or ""
        s = values.to_struct(name)
    elif values.width == 1:
        s = values.to_series()
        if name is not None:
            s = s.alias(name)
    else:
        msg = "cannot initialize Series from DataFrame without any columns"
        raise TypeError(msg)

    if dtype is not None and dtype != s.dtype:
        s = s.cast(dtype, strict=strict)

    return s._s
