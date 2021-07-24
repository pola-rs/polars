import warnings
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Type

import numpy as np
import pyarrow as pa

import polars as pl
from polars.datatypes import (
    DataType,
    Date32,
    Date64,
    numpy_type_to_constructor,
    polars_type_to_constructor,
    py_type_to_arrow_type,
    py_type_to_constructor,
)
from polars.utils import coerce_arrow

try:
    from polars.polars import PyDataFrame, PySeries
except ImportError:
    warnings.warn("binary files missing")

if TYPE_CHECKING:
    import pandas as pd


###################################
# DataFrame constructor interface #
###################################


def _handle_columns_arg(
    data: List["PySeries"],
    columns: Optional[Sequence[str]] = None,
    nullable: bool = True,
) -> List["PySeries"]:
    """
    Rename data according to columns argument.
    """
    if columns is None:
        return data
    else:
        if not data:
            return [pl.Series(c, None, nullable=nullable).inner() for c in columns]
        elif len(data) == len(columns):
            for i, c in enumerate(columns):
                data[i].rename(c)
            return data
        else:
            raise ValueError("Dimensions of columns arg must match data dimensions.")


def dict_to_pydf(
    data: Dict[str, Sequence[Any]],
    columns: Optional[Sequence[str]] = None,
    nullable: bool = True,
) -> "PyDataFrame":
    """
    Construct a PyDataFrame from a dictionary of sequences.
    """
    data_series = [
        pl.Series(name, values, nullable=nullable).inner()
        for name, values in data.items()
    ]
    data_series = _handle_columns_arg(data_series, columns=columns, nullable=nullable)
    return PyDataFrame(data_series)


def numpy_to_pydf(
    data: np.ndarray,
    columns: Optional[Sequence[str]] = None,
    orient: Optional[str] = None,
    nullable: bool = True,
) -> "PyDataFrame":
    """
    Construct a PyDataFrame from a numpy ndarray.
    """
    shape = data.shape

    if shape == (0,):
        data_series = []

    elif len(shape) == 1:
        s = pl.Series("column_0", data, nullable=False).inner()
        data_series = [s]

    elif len(shape) == 2:
        # Infer orientation
        if orient is None:
            warnings.warn(
                "Default orientation for constructing DataFrame from numpy "
                'array will change from "row" to "column" in a future version. '
                "Specify orientation explicitly to silence this warning.",
                DeprecationWarning,
                stacklevel=2,
            )
            orient = "row"
        # Exchange if-block above for block below when removing warning
        # if orientation is None and columns is not None:
        #     orientation = "col" if len(columns) == shape[0] else "row"

        if orient == "row":
            data_series = [
                pl.Series(f"column_{i}", data[:, i], nullable=False).inner()
                for i in range(shape[1])
            ]
        else:
            data_series = [
                pl.Series(f"column_{i}", data[i], nullable=False).inner()
                for i in range(shape[0])
            ]
    else:
        raise ValueError("A numpy array should not have more than two dimensions.")

    data_series = _handle_columns_arg(data_series, columns=columns, nullable=nullable)

    return PyDataFrame(data_series)


def sequence_to_pydf(
    data: Sequence[Any],
    columns: Optional[Sequence[str]] = None,
    orient: Optional[str] = None,
    nullable: bool = True,
) -> "PyDataFrame":
    """
    Construct a PyDataFrame from a sequence.
    """
    data_series: List["PySeries"]
    if len(data) == 0:
        data_series = []

    elif isinstance(data[0], pl.Series):
        data_series = []
        for i, s in enumerate(data):
            if not s.name:  # TODO: Replace by `if s.name is None` once allowed
                s.rename(f"column_{i}", in_place=True)
            data_series.append(s.inner())

    elif isinstance(data[0], Sequence) and not isinstance(data[0], str):
        # Infer orientation
        if orient is None and columns is not None:
            orient = "col" if len(columns) == len(data) else "row"

        if orient == "row":
            pydf = PyDataFrame.read_rows(data)
            if columns is not None:
                pydf.set_column_names(columns)
            return pydf
        else:
            data_series = [
                pl.Series(f"column_{i}", data[i], nullable=nullable).inner()
                for i in range(len(data))
            ]

    else:
        s = pl.Series("column_0", data, nullable=nullable).inner()
        data_series = [s]

    data_series = _handle_columns_arg(data_series, columns=columns, nullable=nullable)
    return PyDataFrame(data_series)


def pandas_to_pydf(
    data: "pd.DataFrame",
    columns: Optional[Sequence[str]] = None,
    nullable: bool = True,
) -> "PyDataFrame":
    """
    Construct a PyDataFrame from a pandas DataFrame.
    """
    if nullable:
        data_series = [
            pl.Series(str(col), data[col].to_list(), nullable=nullable).inner()
            for col in data.columns
        ]
    else:
        data_series = [
            pl.Series(str(col), data[col].values, nullable=nullable).inner()
            for col in data.columns
        ]
    data_series = _handle_columns_arg(data_series, columns=columns, nullable=nullable)
    return PyDataFrame(data_series)


def series_to_pydf(
    data: "pl.Series",
    columns: Optional[Sequence[str]] = None,
) -> "PyDataFrame":
    """
    Construct a PyDataFrame from a Polars Series.
    """
    data_series = [data.inner()]
    data_series = _handle_columns_arg(data_series, columns=columns)
    return PyDataFrame(data_series)


################################
# Series constructor interface #
################################


def series_to_pyseries(
    name: Optional[str],
    values: "pl.Series",
) -> "PySeries":
    """
    Construct a PySeries from a Polars Series.
    """
    values.rename(name, in_place=True)
    return values.inner()


def arrow_to_pyseries(name: Optional[str], values: pa.Array) -> "PySeries":
    """
    Construct a PySeries from an Arrow array.
    """
    array = coerce_arrow(values)
    return PySeries.from_arrow(name, array)


def numpy_to_pyseries(
    name: Optional[str],
    values: np.ndarray,
    nullable: bool = True,
) -> "PySeries":
    """
    Construct a PySeries from a numpy array.
    """
    if not values.data.contiguous:
        values = np.array(values)

    if len(values.shape) == 1:
        dtype = values.dtype.type
        constructor = numpy_type_to_constructor(dtype)
        if dtype == np.float32 or dtype == np.float64:
            return constructor(name, values, nullable)
        else:
            return constructor(name, values)
    else:
        return PySeries.new_object(name, values)


def _get_first_non_none(values: Sequence[Optional[Any]]) -> Any:
    """
    Return the first value from a sequence that isn't None.

    If sequence doesn't contain non-None values, return None.
    """
    return next((v for v in values if v is not None), None)


def sequence_to_pyseries(
    name: Optional[str],
    values: Sequence[Any],
    dtype: Optional[Type[DataType]] = None,
) -> "PySeries":
    """
    Construct a PySeries from a sequence.
    """
    if dtype is not None:
        constructor = polars_type_to_constructor(dtype)
        pyseries = constructor(name, values)
        if dtype == Date32:
            pyseries = pyseries.cast_date32()
        elif dtype == Date64:
            pyseries = pyseries.cast_date64()
        return pyseries

    else:
        value = _get_first_non_none(values)
        dtype_ = type(value) if value is not None else float

        if dtype_ == date or dtype_ == datetime:
            return arrow_to_pyseries(name, pa.array(values))

        elif dtype_ == list or dtype_ == tuple:
            nested_value = _get_first_non_none(value)
            nested_dtype = type(nested_value) if value is not None else float

            try:
                nested_arrow_dtype = py_type_to_arrow_type(nested_dtype)
            except ValueError as e:
                raise ValueError(
                    f"Cannot construct Series from sequence of {nested_dtype}."
                ) from e

            try:
                arrow_values = pa.array(values, pa.large_list(nested_arrow_dtype))
                return arrow_to_pyseries(name, arrow_values)
            # failure expected for mixed sequences like `[[12], "foo", 9]`
            except pa.lib.ArrowInvalid:
                return PySeries.new_object(name, values)

        else:
            constructor = py_type_to_constructor(dtype_)
            return constructor(name, values)
