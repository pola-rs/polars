from __future__ import annotations

from contextlib import suppress
from dataclasses import astuple, is_dataclass
from datetime import date, datetime, time, timedelta
from itertools import islice, zip_longest
from sys import version_info
from typing import (
    TYPE_CHECKING,
    Any,
    Generator,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
    get_type_hints,
)

from polars import internals as pli
from polars.datatypes import (
    N_INFER_DEFAULT,
    Categorical,
    Date,
    Datetime,
    Duration,
    Float32,
    List,
    PolarsDataType,
    SchemaDefinition,
    SchemaDict,
    Struct,
    Time,
    Unknown,
    Utf8,
    dtype_to_arrow_type,
    dtype_to_py_type,
    is_polars_dtype,
    py_type_to_arrow_type,
    py_type_to_dtype,
)
from polars.datatypes_constructor import (
    numpy_type_to_constructor,
    numpy_values_and_dtype,
    polars_type_to_constructor,
    py_type_to_constructor,
)
from polars.dependencies import (
    _NUMPY_AVAILABLE,
    _PYARROW_AVAILABLE,
    _check_for_numpy,
    _check_for_pandas,
)
from polars.dependencies import numpy as np
from polars.dependencies import pandas as pd
from polars.dependencies import pyarrow as pa
from polars.exceptions import ShapeError
from polars.utils import _is_generator, arrlen, range_to_series, threadpool_size

if version_info >= (3, 10):

    def dataclass_type_hints(obj: type) -> dict[str, Any]:
        return get_type_hints(obj)

else:

    def dataclass_type_hints(obj: type) -> dict[str, Any]:
        return obj.__annotations__


try:
    from polars.polars import PyDataFrame, PySeries

    _DOCUMENTING = False
except ImportError:
    _DOCUMENTING = True


if TYPE_CHECKING:
    from polars.internals.type_aliases import Orientation


def is_namedtuple(value: Any, annotated: bool = False) -> bool:
    """Infer whether value is a NamedTuple."""
    if all(hasattr(value, attr) for attr in ("_fields", "_field_defaults", "_replace")):
        return len(value.__annotations__) == len(value._fields) if annotated else True
    return False


def include_unknowns(
    schema: SchemaDict, cols: Sequence[str]
) -> MutableMapping[str, PolarsDataType]:
    """Complete partial schema dict by including Unknown type."""
    return {col: schema.get(col, Unknown) for col in cols}


################################
# Series constructor interface #
################################


def series_to_pyseries(name: str, values: pli.Series) -> PySeries:
    """Construct a PySeries from a Polars Series."""
    values.rename(name, in_place=True)
    return values._s


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
        pys = pli.Series(name, [], dtype=Categorical)._s

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
            pys = PySeries.from_arrow(name, array.combine_chunks())

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
        return PySeries.new_from_anyvalues(name, values)
    # raised if we cannot convert to Wrap<AnyValue>
    except RuntimeError:
        return PySeries.new_object(name, values, False)


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

    def to_series_chunk(values: list[Any], dtype: PolarsDataType | None) -> pli.Series:
        return pli.Series(
            name=name,
            values=values,
            dtype=dtype,
            strict=strict,
            dtype_if_empty=dtype_if_empty,
        )

    n_chunks = 0
    series: pli.Series = None  # type: ignore[assignment]
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
    nested_dtype: PolarsDataType | type | None = None

    # empty sequence
    if not values and dtype is None:
        if dtype_if_empty:
            # if dtype for empty sequence could be guessed
            # (e.g comparisons between self and other)
            dtype = dtype_if_empty
        else:
            # default to Float32 type
            dtype = Float32
    # lists defer to subsequent handling; identify nested type
    elif dtype == List:
        nested_dtype = getattr(dtype, "inner", None)
        python_dtype = list

    # infer temporal type handling
    py_temporal_types = {date, datetime, timedelta, time}
    pl_temporal_types = {Date, Datetime, Duration, Time}

    value = _get_first_non_none(values)
    if value is not None:
        if is_dataclass(value) or is_namedtuple(value, annotated=True):
            return pli.DataFrame(values).to_struct(name)._s
        elif isinstance(value, range):
            values = [range_to_series("", v) for v in values]
        else:
            # for temporal dtypes:
            # * if the values are integer, we take the physical branch.
            # * if the values are python types, take the temporal branch.
            # * if the values are ISO-8601 strings, init then convert via strptime.
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
        and dtype not in (Struct, Unknown)
        and is_polars_dtype(dtype)
        and (python_dtype is None)
    ):
        constructor = polars_type_to_constructor(dtype)
        pyseries = constructor(name, values, strict)

        if dtype in (Date, Datetime, Duration, Time, Categorical):
            pyseries = pyseries.cast(dtype, True)
        return pyseries

    elif dtype == Struct:
        struct_schema = dtype.to_schema() if isinstance(dtype, Struct) else None
        empty = {}  # type: ignore[var-annotated]
        return sequence_to_pydf(
            data=[(empty if v is None else v) for v in values],
            columns=struct_schema,
            orient="row",
        ).to_struct(name)
    else:
        if python_dtype is None:
            if value is None:
                # generic default dtype
                python_dtype = float
            else:
                python_dtype = type(value)

        # temporal branch
        if python_dtype in py_temporal_types:
            if dtype is None:
                dtype = py_type_to_dtype(python_dtype)  # construct from integer
            elif dtype in py_temporal_types:
                dtype = py_type_to_dtype(dtype)

            # we use anyvalue builder to create the datetime array
            # we store the values internally as UTC and set the timezone
            if dtype == Datetime and value.tzinfo is not None:
                py_series = PySeries.new_from_anyvalues(name, values)
                tz = str(value.tzinfo)
                return pli.wrap_s(py_series).dt.tz_localize(tz)._s

            # TODO: use anyvalues here (no need to require pyarrow for this).
            arrow_dtype = dtype_to_arrow_type(dtype)
            return arrow_to_pyseries(name, pa.array(values, type=arrow_dtype))

        elif python_dtype in (list, tuple):
            if nested_dtype is None:
                nested_value = _get_first_non_none(value)
                if isinstance(nested_value, range):
                    nested_dtype = list
                else:
                    nested_dtype = (
                        type(nested_value) if nested_value is not None else float
                    )

            # recursively call Series constructor
            if nested_dtype == list:
                return sequence_to_pyseries(
                    name=name,
                    values=[
                        sequence_to_pyseries(name, seq, dtype=None, strict=strict)
                        for seq in values
                    ],
                    dtype=None,
                    strict=strict,
                )

            # logs will show a panic if we infer wrong dtype and it's hard to error
            # from the rust side. to reduce the likelihood of this happening we
            # infer the dtype of first 100 elements; if all() fails, we will hit
            # the PySeries.new_object
            if not _PYARROW_AVAILABLE:
                # check lists for consistent inner types
                if isinstance(value, list):
                    count = 0
                    equal_to_inner = True
                    for lst in values:
                        for vl in lst:
                            equal_to_inner = type(vl) == nested_dtype
                            if not equal_to_inner or count > 50:
                                break
                            count += 1
                    if equal_to_inner:
                        dtype = py_type_to_dtype(nested_dtype)
                        with suppress(BaseException):
                            return PySeries.new_list(name, values, dtype)
                # pass; we create an object if we get here
            else:
                try:
                    if is_polars_dtype(nested_dtype):
                        nested_arrow_dtype = dtype_to_arrow_type(
                            nested_dtype  # type: ignore[arg-type]
                        )
                    else:
                        nested_arrow_dtype = py_type_to_arrow_type(
                            nested_dtype  # type: ignore[arg-type]
                        )
                except ValueError:  # pragma: no cover
                    return sequence_from_anyvalue_or_object(name, values)
                with suppress(pa.lib.ArrowInvalid, pa.lib.ArrowTypeError):
                    arrow_values = pa.array(values, pa.large_list(nested_arrow_dtype))
                    return arrow_to_pyseries(name, arrow_values)

            # Convert mixed sequences like `[[12], "foo", 9]`
            return PySeries.new_object(name, values, strict)

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

        elif python_dtype == pli.Series:
            return PySeries.new_series_list(name, [v._s for v in values], strict)

        elif python_dtype == PySeries:
            return PySeries.new_series_list(name, values, strict)
        else:
            constructor = py_type_to_constructor(python_dtype)
            if constructor == PySeries.new_object:
                try:
                    return PySeries.new_from_anyvalues(name, values)
                # raised if we cannot convert to Wrap<AnyValue>
                except RuntimeError:
                    return sequence_from_anyvalue_or_object(name, values)

            while True:
                try:
                    return constructor(name, values, strict)
                except TypeError as error:
                    str_val = str(error)

                    # from x to float
                    # error message can be:
                    #   - integers: "'float' object cannot be interpreted as an integer"
                    if "'float'" in str_val:
                        constructor = py_type_to_constructor(float)

                    # from x to string
                    # error message can be:
                    #   - integers: "'str' object cannot be interpreted as an integer"
                    #   - floats: "must be real number, not str"
                    elif (
                        "'str'" in str_val or str_val == "must be real number, not str"
                    ):
                        constructor = py_type_to_constructor(str)

                    # from x to int
                    # error message can be:
                    #   - bools: "'int' object cannot be converted to 'PyBool'"
                    elif str_val == "'int' object cannot be converted to 'PyBool'":
                        constructor = py_type_to_constructor(int)
                    else:
                        raise error


def _pandas_series_to_arrow(
    values: pd.Series | pd.DatetimeIndex,
    nan_to_none: bool = True,
    min_len: int | None = None,
) -> pa.Array:
    """
    Convert a pandas Series to an Arrow Array.

    Parameters
    ----------
    values : :class:`pandas.Series` or :class:`pandas.DatetimeIndex`
        Series to convert to arrow
    nan_to_none : bool, default = True
        Interpret `NaN` as missing values
    min_len : int, optional
        in case of null values, this length will be used to create a dummy f64 array
        (with all values set to null)

    Returns
    -------
    :class:`pyarrow.Array`

    """
    dtype = getattr(values, "dtype", None)
    if dtype == "object":
        first_non_none = _get_first_non_none(values.values)  # type: ignore[arg-type]

        if isinstance(first_non_none, str):
            return pa.array(values, pa.large_utf8(), from_pandas=nan_to_none)
        if first_non_none is None:
            return pa.nulls(min_len, pa.large_utf8())

        return pa.array(values, from_pandas=nan_to_none)
    elif dtype:
        return pa.array(values, from_pandas=nan_to_none)
    else:
        # Pandas Series is actually a Pandas DataFrame when the original dataframe
        # contains duplicated columns and a duplicated column is requested with df["a"].
        raise ValueError(
            "Duplicate column names found: "
            + f"{str(values.columns.tolist())}"  # type: ignore[union-attr]
        )


def pandas_to_pyseries(
    name: str, values: pd.Series | pd.DatetimeIndex, nan_to_none: bool = True
) -> PySeries:
    """Construct a PySeries from a pandas Series or DatetimeIndex."""
    # TODO: Change `if not name` to `if name is not None` once name is Optional[str]
    if not name and values.name is not None:
        name = str(values.name)
    return arrow_to_pyseries(
        name, _pandas_series_to_arrow(values, nan_to_none=nan_to_none)
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
            return [pli.Series(c, None)._s for c in columns]
        elif len(data) == len(columns):
            if from_dict:
                series_map = {s.name(): s for s in data}
                if all((col in series_map) for col in columns):
                    return [series_map[col] for col in columns]

            for i, c in enumerate(columns):
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
    columns, dtypes = _unpack_columns(
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
            column_casts.append(pli.col(col).cast(Categorical)._pyexpr)
        elif structs and col in structs and structs[col] != pydf_dtypes[i]:
            column_casts.append(pli.col(col).cast(structs[col])._pyexpr)
        elif col in dtypes and dtypes[col] != pydf_dtypes[i]:
            column_casts.append(pli.col(col).cast(dtypes[col])._pyexpr)

    if column_casts or column_subset:
        pydf = pydf.lazy()
        if column_casts:
            pydf = pydf.with_columns(column_casts)
        if column_subset:
            pydf = pydf.select([pli.col(col)._pyexpr for col in column_subset])
        pydf = pydf.collect()

    return pydf


def _unpack_columns(
    columns: SchemaDefinition | None,
    schema_overrides: SchemaDict | None = None,
    n_expected: int | None = None,
    lookup_names: Iterable[str] | None = None,
) -> tuple[list[str], SchemaDict]:
    """
    Unpack column names and create dtype lookup.

    Works for any (name, dtype) pairs or schema dict input,
    overriding any inferred dtypes with explicit dtypes if supplied.
    """
    if isinstance(columns, dict):
        columns = list(columns.items())
    column_names = [
        (col or f"column_{i}") if isinstance(col, str) else col[0]
        for i, col in enumerate(columns or [])
    ]
    if not column_names and n_expected:
        column_names = [f"column_{i}" for i in range(n_expected)]
    lookup = {
        col: name for col, name in zip_longest(column_names, lookup_names or []) if name
    }
    column_dtypes = {
        lookup.get(col[0], col[0]): col[1]
        for col in (columns or [])
        if not isinstance(col, str) and col[1]
    }
    if schema_overrides:
        column_dtypes.update(schema_overrides)

    return (
        column_names or None,  # type: ignore[return-value]
        column_dtypes,
    )


def _expand_dict_scalars(
    data: Mapping[str, Sequence[object] | Mapping[str, Sequence[object]] | pli.Series],
    schema_overrides: SchemaDict | None = None,
    order: Sequence[str] | None = None,
) -> dict[str, pli.Series]:
    """Expand scalar values in dict data (propagate literal as array)."""
    updated_data = {}
    if data:
        dtypes = schema_overrides or {}
        array_len = max((arrlen(val) or 0) for val in data.values())
        if array_len > 0:
            for name, val in data.items():
                dtype = dtypes.get(name)
                if isinstance(val, dict) and dtype != Struct:
                    updated_data[name] = pli.DataFrame(val).to_struct(name)

                elif arrlen(val) is not None or _is_generator(val):
                    updated_data[name] = pli.Series(name=name, values=val, dtype=dtype)

                elif val is None or isinstance(  # type: ignore[redundant-expr]
                    val, (int, float, str, bool)
                ):
                    updated_data[name] = pli.Series(
                        name=name, values=[val], dtype=dtype
                    ).extend_constant(val, array_len - 1)
                else:
                    updated_data[name] = pli.Series(
                        name=name, values=[val] * array_len, dtype=dtype
                    )

        elif all((arrlen(val) == 0) for val in data.values()):
            for name, val in data.items():
                updated_data[name] = pli.Series(
                    name, values=val, dtype=dtypes.get(name)
                )

        elif all((arrlen(val) is None) for val in data.values()):
            for name, val in data.items():
                updated_data[name] = pli.Series(
                    name,
                    values=(val if _is_generator(val) else [val]),
                    dtype=dtypes.get(name),
                )
    if order and list(updated_data) != order:
        return {col: updated_data.pop(col) for col in order}
    return updated_data


def dict_to_pydf(
    data: Mapping[str, Sequence[object] | Mapping[str, Sequence[object]] | pli.Series],
    columns: SchemaDefinition | None = None,
    schema_overrides: SchemaDict | None = None,
) -> PyDataFrame:
    """Construct a PyDataFrame from a dictionary of sequences."""
    if not columns:
        columns = list(data)
    if columns:
        # the columns arg may also set the dtype/column order of the series
        if isinstance(columns, dict) and data:
            if not all((col in columns) for col in data):
                raise ValueError(
                    "The given column-schema names do not match the data dictionary"
                )
            data = {col: data[col] for col in columns}

        columns, schema_overrides = _unpack_columns(
            columns, lookup_names=data.keys(), schema_overrides=schema_overrides
        )
        if not data and schema_overrides:
            data_series = [
                pli.Series(name, [], schema_overrides.get(name))._s for name in columns
            ]
        else:
            data_series = [
                s._s for s in _expand_dict_scalars(data, schema_overrides).values()
            ]

        data_series = _handle_columns_arg(data_series, columns=columns, from_dict=True)
        return PyDataFrame(data_series)

    if _NUMPY_AVAILABLE:
        count_numpy = 0
        for val in data.values():
            # only start a thread pool from a reasonable size.
            count_numpy += int(
                _check_for_numpy(val)
                and isinstance(val, np.ndarray)
                and len(val) > 1000
            )

        # if we have more than 3 numpy arrays we multi-thread
        if count_numpy > 2:
            # yes, multi-threading was easier in python here
            # we cannot run multiple threads that run python code
            # and release the gil in pyo3
            # it will deadlock.

            # dummy is threaded
            import multiprocessing.dummy

            pool_size = threadpool_size()
            with multiprocessing.dummy.Pool(pool_size) as pool:
                data_series = pool.map(
                    lambda t: pli.Series(t[0], t[1])._s,
                    [(k, v) for k, v in data.items()],
                )
            return PyDataFrame(data_series)

    data = _expand_dict_scalars(data)
    return PyDataFrame.read_dict(data)


def sequence_to_pydf(
    data: Sequence[Any],
    columns: SchemaDefinition | None = None,
    schema_overrides: SchemaDict | None = None,
    orient: Orientation | None = None,
    infer_schema_length: int | None = N_INFER_DEFAULT,
) -> PyDataFrame:
    """Construct a PyDataFrame from a sequence."""
    data_series: list[PySeries]
    if len(data) == 0:
        return dict_to_pydf({}, columns=columns, schema_overrides=schema_overrides)

    if isinstance(data[0], Generator):
        data = [list(row) for row in data]

    if isinstance(data[0], pli.Series):
        series_names = [s.name for s in data]
        columns, schema_overrides = _unpack_columns(
            columns or series_names,
            schema_overrides=schema_overrides,
            n_expected=len(data),
        )
        data_series = []
        for i, s in enumerate(data):
            if not s.name:  # TODO: Replace by `if s.name is None` once allowed
                s.rename(columns[i], in_place=True)
            new_dtype = schema_overrides.get(columns[i])
            if new_dtype and new_dtype != s.dtype:
                s = s.cast(new_dtype)
            data_series.append(s._s)

    elif isinstance(data[0], dict):
        column_names, schema_overrides = _unpack_columns(
            columns, schema_overrides=schema_overrides
        )
        dtypes = (
            include_unknowns(schema_overrides, column_names)
            if schema_overrides
            else None
        )
        pydf = PyDataFrame.read_dicts(data, infer_schema_length, dtypes)
        if column_names:
            pydf = _post_apply_columns(pydf, column_names)
        return pydf

    elif isinstance(data[0], (list, tuple, Sequence)) and not isinstance(data[0], str):
        if is_namedtuple(data[0]):
            if columns is None:
                columns = data[0]._fields  # type: ignore[attr-defined]
                if len(data[0].__annotations__) == len(columns):
                    columns = [
                        (name, py_type_to_dtype(tp, raise_unmatched=False))
                        for name, tp in data[0].__annotations__.items()
                    ]
            elif orient is None:
                orient = "row"

        if orient is None and columns is not None:
            orient = "col" if len(columns) == len(data) else "row"

        if orient == "row":
            column_names, schema_overrides = _unpack_columns(
                columns, schema_overrides=schema_overrides, n_expected=len(data[0])
            )
            schema_override = (
                include_unknowns(schema_overrides, column_names)
                if schema_overrides
                else {}
            )
            if column_names and data and len(data[0]) != len(column_names):
                raise ShapeError("The row data does not match the number of columns")

            for col, tp in schema_override.items():
                if tp == Categorical:
                    schema_override[col] = Utf8

            pydf = PyDataFrame.read_rows(
                data,
                infer_schema_length,
                schema_override or None,
            )
            if column_names or schema_overrides:
                pydf = _post_apply_columns(
                    pydf, column_names, schema_overrides=schema_overrides
                )
            return pydf

        elif orient == "col" or orient is None:
            columns, schema_overrides = _unpack_columns(
                columns, schema_overrides=schema_overrides, n_expected=len(data)
            )
            data_series = [
                pli.Series(columns[i], data[i], schema_overrides.get(columns[i]))._s
                for i in range(len(data))
            ]
        else:
            raise ValueError(
                f"orient must be one of {{'col', 'row', None}}, got {orient} instead."
            )

    elif is_dataclass(data[0]):
        if columns:
            columns, schema_overrides = _unpack_columns(
                columns, schema_overrides=schema_overrides
            )
            schema_override = {
                col: schema_overrides.get(col, Unknown) for col in columns
            }
        else:
            columns = None
            schema_override = {
                col: (py_type_to_dtype(tp, raise_unmatched=False) or Unknown)
                for col, tp in dataclass_type_hints(data[0].__class__).items()
            }
            schema_override.update(schema_overrides or {})

        for col, tp in schema_override.items():
            if tp == Categorical:
                schema_override[col] = Utf8

        pydf = PyDataFrame.read_rows(
            [astuple(dc) for dc in data], infer_schema_length, schema_override or None
        )
        if schema_override:
            structs = {
                col: tp for col, tp in schema_override.items() if isinstance(tp, Struct)
            }
            pydf = _post_apply_columns(
                pydf, columns, structs, schema_overrides=schema_overrides
            )
        return pydf

    elif _check_for_pandas(data[0]) and isinstance(
        data[0], (pd.Series, pd.DatetimeIndex)
    ):
        if columns is not None:
            columns, schema_overrides = _unpack_columns(
                columns, schema_overrides=schema_overrides, n_expected=1
            )

        schema_overrides = schema_overrides or {}
        data_series = []
        for i, s in enumerate(data):
            name = columns[i] if columns else s.name
            dtype = schema_overrides.get(name, None)
            pyseries = pandas_to_pyseries(name=name, values=s)
            if dtype is not None and dtype != pyseries.dtype():
                pyseries = pyseries.cast(dtype, strict=True)
            data_series.append(pyseries)

        columns = None
    else:
        columns, schema_overrides = _unpack_columns(
            columns, schema_overrides=schema_overrides, n_expected=1
        )
        data_series = [
            pli.Series(columns[0], data, schema_overrides.get(columns[0]))._s
        ]

    data_series = _handle_columns_arg(data_series, columns=columns)
    return PyDataFrame(data_series)


def numpy_to_pydf(
    data: np.ndarray[Any, Any],
    columns: SchemaDefinition | None = None,
    schema_overrides: SchemaDict | None = None,
    orient: Orientation | None = None,
) -> PyDataFrame:
    """Construct a PyDataFrame from a numpy ndarray."""
    shape = data.shape

    # Unpack columns
    if shape == (0,):
        n_columns = 0

    elif len(shape) == 1:
        n_columns = 1

    elif len(shape) == 2:
        # default convention
        # first axis is rows, second axis is columns
        if orient is None and columns is None:
            n_columns = shape[1]
            orient = "row"

        # Infer orientation if columns argument is given
        elif orient is None and columns is not None:
            if len(columns) == shape[0]:
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
                f"orient must be one of {{'col', 'row', None}}, got {orient} instead."
            )
    else:
        raise ValueError(
            "Cannot create DataFrame from numpy array with more than two dimensions."
        )

    if columns is not None and len(columns) != n_columns:
        raise ValueError("Dimensions of columns arg must match data dimensions.")

    columns, schema_overrides = _unpack_columns(
        columns, schema_overrides=schema_overrides, n_expected=n_columns
    )

    # Convert data to series
    if shape == (0,):
        data_series = []

    elif len(shape) == 1:
        data_series = [
            pli.Series(columns[0], data, schema_overrides.get(columns[0]))._s
        ]

    else:
        if orient == "row":
            data_series = [
                pli.Series(columns[i], data[:, i], schema_overrides.get(columns[i]))._s
                for i in range(n_columns)
            ]
        else:
            data_series = [
                pli.Series(columns[i], data[i], schema_overrides.get(columns[i]))._s
                for i in range(n_columns)
            ]

    data_series = _handle_columns_arg(data_series, columns=columns)
    return PyDataFrame(data_series)


def arrow_to_pydf(
    data: pa.Table,
    columns: SchemaDefinition | None = None,
    schema_overrides: SchemaDict | None = None,
    rechunk: bool = True,
) -> PyDataFrame:
    """Construct a PyDataFrame from an Arrow Table."""
    original_columns = columns
    columns, schema_overrides = _unpack_columns(
        (columns or data.column_names), schema_overrides=schema_overrides
    )
    try:
        if columns and columns != data.column_names:
            data = data.rename_columns(columns)
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
        if column._name is None:
            name = f"column_{i}"
        else:
            name = column._name
        names.append(name)

        column = coerce_arrow(column)
        if pa.types.is_dictionary(column.type):
            ps = arrow_to_pyseries(name, column, rechunk)
            dictionary_cols[i] = pli.wrap_s(ps)
        elif isinstance(column.type, pa.StructType) and column.num_chunks > 1:
            ps = arrow_to_pyseries(name, column, rechunk)
            struct_cols[i] = pli.wrap_s(ps)
        else:
            data_dict[name] = column

    if len(data_dict) > 0:
        tbl = pa.table(data_dict)

        # path for table without rows that keeps datatype
        if tbl.shape[0] == 0:
            pydf = pli.DataFrame(
                [
                    pli.Series(name, c)
                    for (name, c) in zip(tbl.column_names, tbl.columns)
                ]
            )._df
        else:
            pydf = PyDataFrame.from_arrow_record_batches(tbl.to_batches())
    else:
        pydf = pli.DataFrame([])._df
    if rechunk:
        pydf = pydf.rechunk()

    reset_order = False
    if len(dictionary_cols) > 0:
        df = pli.wrap_df(pydf)
        df = df.with_columns(
            [pli.lit(s).alias(s.name) for s in dictionary_cols.values()]
        )
        reset_order = True

    if len(struct_cols) > 0:
        df = pli.wrap_df(pydf)
        df = df.with_columns([pli.lit(s).alias(s.name) for s in struct_cols.values()])
        reset_order = True

    if reset_order:
        df = df[names]
        pydf = df._df

    if columns != original_columns and (schema_overrides or original_columns):
        pydf = _post_apply_columns(
            pydf, original_columns, schema_overrides=schema_overrides
        )
    elif schema_overrides:
        for col, dtype in zip(pydf.columns(), pydf.dtypes()):
            override_dtype = schema_overrides.get(col)
            if override_dtype is not None and dtype != override_dtype:
                pydf = _post_apply_columns(
                    pydf, original_columns, schema_overrides=schema_overrides
                )
                break

    return pydf


def series_to_pydf(
    data: pli.Series,
    columns: SchemaDefinition | None = None,
    schema_overrides: SchemaDict | None = None,
) -> PyDataFrame:
    """Construct a PyDataFrame from a Polars Series."""
    data_series = [data._s]
    series_name = [s.name() for s in data_series]
    columns, schema_overrides = _unpack_columns(
        columns or series_name, schema_overrides=schema_overrides, n_expected=1
    )
    if schema_overrides:
        new_dtype = list(schema_overrides.values())[0]
        if new_dtype != data.dtype:
            data_series[0] = data_series[0].cast(new_dtype, True)

    data_series = _handle_columns_arg(data_series, columns=columns)
    return PyDataFrame(data_series)


def iterable_to_pydf(
    data: Iterable[Any],
    columns: SchemaDefinition | None = None,
    schema_overrides: SchemaDict | None = None,
    orient: Orientation | None = None,
    chunk_size: int | None = None,
    infer_schema_length: int | None = N_INFER_DEFAULT,
) -> PyDataFrame:
    """Construct a PyDataFrame from an iterable/generator."""
    original_columns = columns
    dtypes_by_idx: dict[int, PolarsDataType] = {}
    if columns is not None:
        columns, schema_overrides = _unpack_columns(
            columns, schema_overrides=schema_overrides
        )
    elif schema_overrides:
        _columns, schema_overrides = _unpack_columns(
            columns, schema_overrides=schema_overrides
        )

    if not isinstance(data, Generator):
        data = iter(data)

    if orient == "col":
        if columns and schema_overrides:
            dtypes_by_idx = {
                idx: schema_overrides.get(col, Unknown)
                for idx, col in enumerate(columns)
            }

        return pli.DataFrame(
            {
                (f"column_{idx}" if columns is None else columns[idx]): pli.Series(
                    coldata, dtype=dtypes_by_idx.get(idx)
                )
                for idx, coldata in enumerate(data)
            }
        )._df

    def to_frame_chunk(
        values: list[Any], columns: SchemaDefinition | None
    ) -> pli.DataFrame:
        return pli.DataFrame(
            data=values,
            columns=columns,
            orient="row",
            infer_schema_length=infer_schema_length,
        )

    n_chunks = 0
    n_chunk_elems = 1_000_000

    if chunk_size:
        adaptive_chunk_size = chunk_size
    elif columns:
        adaptive_chunk_size = n_chunk_elems // len(columns)
    else:
        adaptive_chunk_size = None

    df: pli.DataFrame = None  # type: ignore[assignment]
    while True:
        values = list(islice(data, adaptive_chunk_size or 1000))
        if not values:
            break
        frame_chunk = to_frame_chunk(values, original_columns)
        if df is None:
            df = frame_chunk
            if not original_columns:
                original_columns = list(df.schema.items())
            if not adaptive_chunk_size:
                adaptive_chunk_size = n_chunk_elems // len(df.columns)
        else:
            df.vstack(frame_chunk, in_place=True)
            n_chunks += 1

    if df is None:
        df = to_frame_chunk([], original_columns)

    return (df.rechunk() if n_chunks > 0 else df)._df


def pandas_to_pydf(
    data: pd.DataFrame,
    columns: SchemaDefinition | None = None,
    schema_overrides: SchemaDict | None = None,
    rechunk: bool = True,
    nan_to_none: bool = True,
) -> PyDataFrame:
    """Construct a PyDataFrame from a pandas DataFrame."""
    length = data.shape[0]
    arrow_dict = {
        str(col): _pandas_series_to_arrow(
            data[col], nan_to_none=nan_to_none, min_len=length
        )
        for col in data.columns
    }
    arrow_table = pa.table(arrow_dict)
    return arrow_to_pydf(
        arrow_table, columns=columns, schema_overrides=schema_overrides, rechunk=rechunk
    )


def coerce_arrow(array: pa.Array, rechunk: bool = True) -> pa.Array:
    # note: Decimal256 could not be cast to float
    if isinstance(array.type, pa.Decimal128Type):
        array = pa.compute.cast(array, pa.float64())

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
            array = pa.compute.cast(
                array, pa.dictionary(pa.uint32(), pa.large_string())
            ).combine_chunks()
    return array
