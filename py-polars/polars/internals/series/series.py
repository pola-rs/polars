from __future__ import annotations

import math
import warnings
from collections.abc import Sized
from datetime import date, datetime, time, timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Iterable,
    NoReturn,
    Sequence,
    Union,
    overload,
)
from warnings import warn

from polars import internals as pli
from polars.datatypes import (
    Boolean,
    DataType,
    Date,
    Datetime,
    Duration,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    List,
    Object,
    PolarsDataType,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Utf8,
    dtype_to_ctype,
    get_idx_type,
    maybe_cast,
    numpy_char_code_to_dtype,
    py_type_to_dtype,
    supported_numpy_char_code,
)
from polars.dependencies import (
    _NUMPY_TYPE,
    _PANDAS_TYPE,
    _PYARROW_AVAILABLE,
    _PYARROW_TYPE,
)
from polars.dependencies import numpy as np
from polars.dependencies import pandas as pd
from polars.dependencies import pyarrow as pa
from polars.exceptions import ShapeError
from polars.internals.construction import (
    arrow_to_pyseries,
    iterable_to_pyseries,
    numpy_to_pyseries,
    pandas_to_pyseries,
    sequence_to_pyseries,
    series_to_pyseries,
)
from polars.internals.series.categorical import CatNameSpace
from polars.internals.series.datetime import DateTimeNameSpace
from polars.internals.series.list import ListNameSpace
from polars.internals.series.string import StringNameSpace
from polars.internals.series.struct import StructNameSpace
from polars.internals.series.utils import expr_dispatch, get_ffi_func
from polars.internals.slice import PolarsSlice
from polars.utils import (
    _date_to_pl_date,
    _datetime_to_pl_timestamp,
    _time_to_pl_time,
    accessor,
    is_bool_sequence,
    is_int_sequence,
    range_to_slice,
    scale_bytes,
)

try:
    from polars.polars import PyDataFrame, PySeries

    _DOCUMENTING = False
except ImportError:
    _DOCUMENTING = True


if TYPE_CHECKING:
    from polars.internals.series._numpy import SeriesView
    from polars.internals.type_aliases import (
        ComparisonOperator,
        FillNullStrategy,
        InterpolationMethod,
        NullBehavior,
        RankMethod,
        RollingInterpolationMethod,
        SizeUnit,
        TimeUnit,
    )


def wrap_s(s: PySeries) -> Series:
    return Series._from_pyseries(s)


ArrayLike = Union[
    Sequence[Any],
    "Series",
    "pa.Array",
    "pa.ChunkedArray",
    "np.ndarray",
    "pd.Series",
    "pd.DatetimeIndex",
]


@expr_dispatch
class Series:
    """
    A Series represents a single column in a polars DataFrame.

    Parameters
    ----------
    name : str, default None
        Name of the series. Will be used as a column name when used in a DataFrame.
        When not specified, name is set to an empty string.
    values : ArrayLike, default None
        One-dimensional data in various forms. Supported are: Sequence, Series,
        pyarrow Array, and numpy ndarray.
    dtype : DataType, default None
        Polars dtype of the Series data. If not specified, the dtype is inferred.
    strict
        Throw error on numeric overflow.
    nan_to_null
        In case a numpy array is used to create this Series, indicate how to deal
        with np.nan values.
    dtype_if_empty=dtype_if_empty : DataType, default None
        If no dtype is specified and values contains None or an empty list,
        set the Polars dtype of the Series data. If not specified, Float32 is used.

    Examples
    --------
    Constructing a Series by specifying name and values positionally:

    >>> s = pl.Series("a", [1, 2, 3])
    >>> s
    shape: (3,)
    Series: 'a' [i64]
    [
            1
            2
            3
    ]

    Notice that the dtype is automatically inferred as a polars Int64:

    >>> s.dtype
    Int64

    Constructing a Series with a specific dtype:

    >>> s2 = pl.Series("a", [1, 2, 3], dtype=pl.Float32)
    >>> s2
    shape: (3,)
    Series: 'a' [f32]
    [
        1.0
        2.0
        3.0
    ]

    It is possible to construct a Series with values as the first positional argument.
    This syntax considered an anti-pattern, but it can be useful in certain
    scenarios. You must specify any other arguments through keywords.

    >>> s3 = pl.Series([1, 2, 3])
    >>> s3
    shape: (3,)
    Series: '' [i64]
    [
            1
            2
            3
    ]

    """

    _s: PySeries = None
    _accessors: set[str] = {"arr", "cat", "dt", "str", "struct"}

    def __init__(
        self,
        name: str | ArrayLike | None = None,
        values: ArrayLike | None = None,
        dtype: PolarsDataType | None = None,
        strict: bool = True,
        nan_to_null: bool = False,
        dtype_if_empty: PolarsDataType | None = None,
    ):

        # Handle case where values are passed as the first argument
        if name is not None and not isinstance(name, str):
            if values is None:
                values = name
                name = None
            else:
                raise ValueError("Series name must be a string.")

        if name is None:
            name = ""

        if values is None:
            self._s = sequence_to_pyseries(
                name, [], dtype=dtype, dtype_if_empty=dtype_if_empty
            )
        elif isinstance(values, Series):
            self._s = series_to_pyseries(name, values)

        elif _PYARROW_TYPE(values) and isinstance(values, (pa.Array, pa.ChunkedArray)):
            self._s = arrow_to_pyseries(name, values)

        elif _NUMPY_TYPE(values) and isinstance(values, np.ndarray):
            self._s = numpy_to_pyseries(name, values, strict, nan_to_null)
            if values.dtype.type == np.datetime64:
                # cast to appropriate dtype, handling NaT values
                dtype = _resolve_datetime_dtype(dtype, values.dtype)
                if dtype is not None:
                    self._s = (
                        self.cast(dtype)
                        .set_at_idx(np.argwhere(np.isnat(values)).flatten(), None)
                        ._s
                    )
                    return

            if dtype is not None:
                self._s = self.cast(dtype, strict=True)._s

        elif isinstance(values, range):
            self._s = (
                pli.arange(
                    low=values.start,
                    high=values.stop,
                    step=values.step,
                    eager=True,
                    dtype=dtype,
                )
                .rename(name, in_place=True)
                ._s
            )
        elif isinstance(values, Sequence):
            self._s = sequence_to_pyseries(
                name, values, dtype=dtype, strict=strict, dtype_if_empty=dtype_if_empty
            )
        elif _PANDAS_TYPE(values) and isinstance(values, (pd.Series, pd.DatetimeIndex)):
            self._s = pandas_to_pyseries(name, values)

        elif isinstance(values, (Generator, Iterable)) and not isinstance(
            values, Sized
        ):
            self._s = iterable_to_pyseries(
                name,
                values,
                dtype=dtype,
                strict=strict,
                dtype_if_empty=dtype_if_empty,
            )
        else:
            raise ValueError(
                f"Series constructor called with unsupported type; got {type(values)}"
            )

    @classmethod
    def _from_pyseries(cls, pyseries: PySeries) -> Series:
        series = cls.__new__(cls)
        series._s = pyseries
        return series

    @classmethod
    def _repeat(
        cls, name: str, val: int | float | str | bool, n: int, dtype: type[DataType]
    ) -> Series:
        return cls._from_pyseries(PySeries.repeat(name, val, n, dtype))

    @classmethod
    def _from_arrow(cls, name: str, values: pa.Array, rechunk: bool = True) -> Series:
        """Construct a Series from an Arrow Array."""
        return cls._from_pyseries(arrow_to_pyseries(name, values, rechunk))

    @classmethod
    def _from_pandas(
        cls, name: str, values: pd.Series | pd.DatetimeIndex, nan_to_none: bool = True
    ) -> Series:
        """Construct a Series from a pandas Series or DatetimeIndex."""
        return cls._from_pyseries(
            pandas_to_pyseries(name, values, nan_to_none=nan_to_none)
        )

    @property
    def dtype(self) -> type[DataType]:
        """
        Get the data type of this Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.dtype
        Int64

        """
        return self._s.dtype()

    @property
    def flags(self) -> dict[str, bool]:
        """
        Get flags that are set on the Series.

        Returns
        -------
        Dictionary containing the flag name and the value

        """
        out = {
            "SORTED_ASC": self._s.is_sorted_flag(),
            "SORTED_DESC": self._s.is_sorted_reverse_flag(),
        }
        if self.dtype == List:
            out["FAST_EXPLODE"] = self._s.can_fast_explode_flag()
        return out

    @property
    def inner_dtype(self) -> type[DataType] | None:
        """
        Get the inner dtype in of a List typed Series.

        Returns
        -------
        DataType

        """
        return self._s.inner_dtype()

    @property
    def name(self) -> str:
        """Get the name of this Series."""
        return self._s.name()

    @property
    def shape(self) -> tuple[int]:
        """Shape of this Series."""
        return (self._s.len(),)

    @property
    def time_unit(self) -> TimeUnit | None:
        """Get the time unit of underlying Datetime Series as {"ns", "us", "ms"}."""
        return self._s.time_unit()

    def __bool__(self) -> NoReturn:
        raise ValueError(
            "The truth value of a Series is ambiguous. Hint: use '&' or '|' to chain "
            "Series boolean results together, not and/or; to check if a Series "
            "contains any values, use 'is_empty()'"
        )

    def __getstate__(self) -> Any:
        return self._s.__getstate__()

    def __setstate__(self, state: Any) -> None:
        self._s = sequence_to_pyseries("", [], Float32)
        self._s.__setstate__(state)

    def __str__(self) -> str:
        s_repr: str = self._s.as_str()
        return s_repr.replace("Series", f"{self.__class__.__name__}", 1)

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return self.len()

    def __and__(self, other: Series) -> Series:
        if not isinstance(other, Series):
            other = Series([other])
        return wrap_s(self._s.bitand(other._s))

    def __rand__(self, other: Series) -> Series:
        return self.__and__(other)

    def __or__(self, other: Series) -> Series:
        if not isinstance(other, Series):
            other = Series([other])
        return wrap_s(self._s.bitor(other._s))

    def __ror__(self, other: Series) -> Series:
        return self.__or__(other)

    def __xor__(self, other: Series) -> Series:
        if not isinstance(other, Series):
            other = Series([other])
        return wrap_s(self._s.bitxor(other._s))

    def __rxor__(self, other: Series) -> Series:
        return self.__xor__(other)

    def _comp(self, other: Any, op: ComparisonOperator) -> Series:
        if isinstance(other, datetime) and self.dtype == Datetime:
            ts = _datetime_to_pl_timestamp(other, self.time_unit)
            f = get_ffi_func(op + "_<>", Int64, self._s)
            assert f is not None
            return wrap_s(f(ts))
        if isinstance(other, time) and self.dtype == Time:
            d = _time_to_pl_time(other)
            f = get_ffi_func(op + "_<>", Int64, self._s)
            assert f is not None
            return wrap_s(f(d))
        if isinstance(other, date) and self.dtype == Date:
            d = _date_to_pl_date(other)
            f = get_ffi_func(op + "_<>", Int32, self._s)
            assert f is not None
            return wrap_s(f(d))

        if isinstance(other, Sequence) and not isinstance(other, str):
            other = Series("", other, dtype_if_empty=self.dtype)
        if isinstance(other, Series):
            return wrap_s(getattr(self._s, op)(other._s))

        if other is not None:
            other = maybe_cast(other, self.dtype, self.time_unit)
        f = get_ffi_func(op + "_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented

        return wrap_s(f(other))

    def __eq__(self, other: Any) -> Series:  # type: ignore[override]
        return self._comp(other, "eq")

    def __ne__(self, other: Any) -> Series:  # type: ignore[override]
        return self._comp(other, "neq")

    def __gt__(self, other: Any) -> Series:
        return self._comp(other, "gt")

    def __lt__(self, other: Any) -> Series:
        return self._comp(other, "lt")

    def __ge__(self, other: Any) -> Series:
        return self._comp(other, "gt_eq")

    def __le__(self, other: Any) -> Series:
        return self._comp(other, "lt_eq")

    def _arithmetic(self, other: Any, op_s: str, op_ffi: str) -> Series:
        if isinstance(other, pli.Expr):
            # expand pl.lit, pl.datetime, pl.duration Exprs to compatible Series
            other = self.to_frame().select(other).to_series()
        if isinstance(other, Series):
            return wrap_s(getattr(self._s, op_s)(other._s))

        # recurse; the 'if' statement above will ensure we return early
        if isinstance(other, (date, datetime, timedelta, str)):
            other = Series("", [other])
            return self._arithmetic(other, op_s, op_ffi)
        if isinstance(other, float) and not self.is_float():
            _s = sequence_to_pyseries("", [other])
            if "rhs" in op_ffi:
                return wrap_s(getattr(_s, op_s)(self._s))
            else:
                return wrap_s(getattr(self._s, op_s)(_s))
        else:
            other = maybe_cast(other, self.dtype, self.time_unit)
            f = get_ffi_func(op_ffi, self.dtype, self._s)
        if f is None:
            raise ValueError(
                f"cannot do arithmetic with series of dtype: {self.dtype} and argument"
                f" of type: {type(other)}"
            )
        return wrap_s(f(other))

    @overload
    def __add__(self, other: pli.DataFrame) -> pli.DataFrame:  # type: ignore[misc]
        ...

    @overload
    def __add__(self, other: Any) -> Series:
        ...

    def __add__(self, other: Any) -> Series | pli.DataFrame:
        if isinstance(other, str):
            other = Series("", [other])
        elif isinstance(other, pli.DataFrame):
            return other + self
        return self._arithmetic(other, "add", "add_<>")

    def __sub__(self, other: Any) -> Series:
        return self._arithmetic(other, "sub", "sub_<>")

    def __truediv__(self, other: Any) -> Series:
        if self.is_datelike():
            raise ValueError("first cast to integer before dividing datelike dtypes")

        # this branch is exactly the floordiv function without rounding the floats
        if self.is_float():
            return self._arithmetic(other, "div", "div_<>")

        return self.cast(Float64) / other

    def __floordiv__(self, other: Any) -> Series:
        if self.is_datelike():
            raise ValueError("first cast to integer before dividing datelike dtypes")
        result = self._arithmetic(other, "div", "div_<>")
        # todo! in place, saves allocation
        if self.is_float() or isinstance(other, float):
            result = result.floor()
        return result

    def __invert__(self) -> Series:
        if self.dtype == Boolean:
            return wrap_s(self._s._not())
        return NotImplemented

    @overload
    def __mul__(self, other: pli.DataFrame) -> pli.DataFrame:  # type: ignore[misc]
        ...

    @overload
    def __mul__(self, other: Any) -> Series:
        ...

    def __mul__(self, other: Any) -> Series | pli.DataFrame:
        if self.is_datelike():
            raise ValueError("first cast to integer before multiplying datelike dtypes")
        elif isinstance(other, pli.DataFrame):
            return other * self
        else:
            return self._arithmetic(other, "mul", "mul_<>")

    def __mod__(self, other: Any) -> Series:
        if self.is_datelike():
            raise ValueError(
                "first cast to integer before applying modulo on datelike dtypes"
            )
        return self._arithmetic(other, "rem", "rem_<>")

    def __rmod__(self, other: Any) -> Series:
        if self.is_datelike():
            raise ValueError(
                "first cast to integer before applying modulo on datelike dtypes"
            )
        return self._arithmetic(other, "rem", "rem_<>_rhs")

    def __radd__(self, other: Any) -> Series:
        if isinstance(other, str):
            return (other + self.to_frame()).to_series()
        return self._arithmetic(other, "add", "add_<>_rhs")

    def __rsub__(self, other: Any) -> Series:
        return self._arithmetic(other, "sub", "sub_<>_rhs")

    def __rtruediv__(self, other: Any) -> Series:
        if self.is_datelike():
            raise ValueError("first cast to integer before dividing datelike dtypes")
        if self.is_float():
            self.__rfloordiv__(other)

        if isinstance(other, int):
            other = float(other)
        return self.cast(Float64).__rfloordiv__(other)

    def __rfloordiv__(self, other: Any) -> Series:
        if self.is_datelike():
            raise ValueError("first cast to integer before dividing datelike dtypes")
        return self._arithmetic(other, "div", "div_<>_rhs")

    def __rmul__(self, other: Any) -> Series:
        if self.is_datelike():
            raise ValueError("first cast to integer before multiplying datelike dtypes")
        return self._arithmetic(other, "mul", "mul_<>")

    def __pow__(self, power: int | float | Series) -> Series:
        if self.is_datelike():
            raise ValueError(
                "first cast to integer before raising datelike dtypes to a power"
            )
        return self.to_frame().select(pli.col(self.name).pow(power)).to_series()

    def __rpow__(self, other: Any) -> Series:
        if self.is_datelike():
            raise ValueError(
                "first cast to integer before raising datelike dtypes to a power"
            )
        return self.to_frame().select(other ** pli.col(self.name)).to_series()

    def __matmul__(self, other: Any) -> float | Series | None:
        if isinstance(other, Sequence) or (
            _NUMPY_TYPE(other) and isinstance(other, np.ndarray)
        ):
            other = Series(other)
        # elif isinstance(other, pli.DataFrame):
        #     return other.__rmatmul__(self)  # type: ignore[return-value]
        return self.dot(other)

    def __rmatmul__(self, other: Any) -> float | Series | None:
        if isinstance(other, Sequence) or (
            _NUMPY_TYPE(other) and isinstance(other, np.ndarray)
        ):
            other = Series(other)
        return other.dot(self)

    def __neg__(self) -> Series:
        return 0 - self

    def __abs__(self) -> Series:
        return self.abs()

    def __copy__(self) -> Series:
        return self.clone()

    def __deepcopy__(self, memo: None = None) -> Series:
        return self.clone()

    def __iter__(self) -> SeriesIter:
        return SeriesIter(self.len(), self)

    def _pos_idxs(self, idxs: np.ndarray[Any, Any] | Series) -> Series:
        # pl.UInt32 (polars) or pl.UInt64 (polars_u64_idx).
        idx_type = get_idx_type()

        if isinstance(idxs, Series):
            if idxs.dtype == idx_type:
                return idxs
            if idxs.dtype in {
                UInt8,
                UInt16,
                UInt64 if idx_type == UInt32 else UInt32,
                Int8,
                Int16,
                Int32,
                Int64,
            }:
                if idx_type == UInt32:
                    if idxs.dtype in {Int64, UInt64}:
                        if idxs.max() >= 2**32:  # type: ignore[operator]
                            raise ValueError(
                                "Index positions should be smaller than 2^32."
                            )
                    if idxs.dtype == Int64:
                        if idxs.min() < -(2**32):  # type: ignore[operator]
                            raise ValueError(
                                "Index positions should be bigger than -2^32 + 1."
                            )
                if idxs.dtype in {Int8, Int16, Int32, Int64}:
                    if idxs.min() < 0:  # type: ignore[operator]
                        if idx_type == UInt32:
                            if idxs.dtype in {Int8, Int16}:
                                idxs = idxs.cast(Int32)
                        else:
                            if idxs.dtype in {Int8, Int16, Int32}:
                                idxs = idxs.cast(Int64)

                        # Update negative indexes to absolute indexes.
                        return (
                            idxs.to_frame()
                            .select(
                                pli.when(pli.col(idxs.name) < 0)
                                .then(self.len() + pli.col(idxs.name))
                                .otherwise(pli.col(idxs.name))
                                .cast(idx_type)
                            )
                            .to_series(0)
                        )

                return idxs.cast(idx_type)

        elif _NUMPY_TYPE(idxs) and isinstance(idxs, np.ndarray):
            if idxs.ndim != 1:
                raise ValueError("Only 1D numpy array is supported as index.")
            if idxs.dtype.kind in ("i", "u"):
                # Numpy array with signed or unsigned integers.

                if idx_type == UInt32:
                    if idxs.dtype in {np.int64, np.uint64} and idxs.max() >= 2**32:
                        raise ValueError("Index positions should be smaller than 2^32.")
                    if idxs.dtype == np.int64 and idxs.min() < -(2**32):
                        raise ValueError(
                            "Index positions should be bigger than -2^32 + 1."
                        )
                if idxs.dtype.kind == "i":
                    if idxs.min() < 0:
                        if idx_type == UInt32:
                            if idxs.dtype in (np.int8, np.int16):
                                idxs = idxs.astype(np.int32)
                        else:
                            if idxs.dtype in (np.int8, np.int16, np.int32):
                                idxs = idxs.astype(np.int64)

                        # Update negative indexes to absolute indexes.
                        idxs = np.where(idxs < 0, self.len() + idxs, idxs)

                    # Cast signed numpy array to unsigned numpy array as all indexes
                    # are positive and casting signed Polars Series to unsigned
                    # Polars series is much slower.
                    if isinstance(idxs, np.ndarray):
                        idxs = idxs.astype(
                            np.uint32 if idx_type == UInt32 else np.uint64
                        )

                return Series("", idxs, dtype=idx_type)

        raise NotImplementedError("Unsupported idxs datatype.")

    @overload
    def __getitem__(self, item: int) -> Any:
        ...

    @overload
    def __getitem__(
        self,
        item: Series | range | slice | np.ndarray[Any, Any] | list[int] | list[bool],
    ) -> Series:
        ...

    def __getitem__(
        self,
        item: int
        | Series
        | range
        | slice
        | np.ndarray[Any, Any]
        | list[int]
        | list[bool],
    ) -> Any:

        if isinstance(item, Series) and item.dtype in {
            UInt8,
            UInt16,
            UInt32,
            UInt64,
            Int8,
            Int16,
            Int32,
            Int64,
        }:
            # Unsigned or signed Series (ordered from fastest to slowest).
            #   - pl.UInt32 (polars) or pl.UInt64 (polars_u64_idx) Series indexes.
            #   - Other unsigned Series indexes are converted to pl.UInt32 (polars)
            #     or pl.UInt64 (polars_u64_idx).
            #   - Signed Series indexes are converted pl.UInt32 (polars) or
            #     pl.UInt64 (polars_u64_idx) after negative indexes are converted
            #     to absolute indexes.
            return wrap_s(self._s.take_with_series(self._pos_idxs(item)._s))

        elif (
            _NUMPY_TYPE(item)
            and isinstance(item, np.ndarray)
            and item.dtype.kind in ("i", "u")
        ):
            if item.ndim != 1:
                raise ValueError("Only a 1D-Numpy array is supported as index.")

            # Unsigned or signed Numpy array (ordered from fastest to slowest).
            #   - np.uint32 (polars) or np.uint64 (polars_u64_idx) numpy array
            #     indexes.
            #   - Other unsigned numpy array indexes are converted to pl.UInt32
            #     (polars) or pl.UInt64 (polars_u64_idx).
            #   - Signed numpy array indexes are converted pl.UInt32 (polars) or
            #     pl.UInt64 (polars_u64_idx) after negative indexes are converted
            #     to absolute indexes.
            return wrap_s(self._s.take_with_series(self._pos_idxs(item)._s))

        # Integer.
        elif isinstance(item, int):
            if item < 0:
                item = self.len() + item
            if self.dtype in (List, Object):
                f = get_ffi_func("get_<>", self.dtype, self._s)
                if f is None:
                    return NotImplemented
                out = f(item)
                if self.dtype == List:
                    if out is None:
                        return None
                    return wrap_s(out)
                return out

            return self._s.get_idx(item)

        # Slice.
        elif isinstance(item, slice):
            return PolarsSlice(self).apply(item)

        # Range.
        elif isinstance(item, range):
            return self[range_to_slice(item)]

        # Sequence of integers (slow to check if sequence contains all integers).
        elif is_int_sequence(item):
            return wrap_s(self._s.take_with_series(self._pos_idxs(Series("", item))._s))

        # Check for boolean masks last as this is deprecated functionality and
        # thus can be a slow path.

        elif isinstance(item, Series) and item.dtype == Boolean:
            warnings.warn(
                "passing a boolean mask to Series.__getitem__ is being deprecated; "
                "instead use Series.filter",
                category=DeprecationWarning,
            )
            return wrap_s(self._s.filter(item._s))

        elif (
            _NUMPY_TYPE(item)
            and isinstance(item, np.ndarray)
            and item.dtype.kind == "b"
        ):
            if item.ndim != 1:
                raise ValueError("Only a 1D-Numpy array is supported as index.")

            warnings.warn(
                "passing a boolean mask to Series.__getitem__ is being deprecated; "
                "instead use Series.filter",
                category=DeprecationWarning,
            )
            return wrap_s(self._s.filter(Series("", item)._s))

        # Sequence of boolean masks (slow to check if sequence contains all booleans).
        elif is_bool_sequence(item):
            warnings.warn(
                "passing a boolean mask to Series.__getitem__ is being deprecated; "
                "instead use Series.filter",
                category=DeprecationWarning,
            )
            return wrap_s(self._s.filter(Series("", item)._s))

        raise ValueError(
            f"Cannot __getitem__ on Series of dtype: '{self.dtype}' "
            f"with argument: '{item}' of type: '{type(item)}'."
        )

    def __setitem__(
        self,
        key: int | Series | np.ndarray[Any, Any] | Sequence[object] | tuple[object],
        value: Any,
    ) -> None:
        if isinstance(value, Sequence) and not isinstance(value, str):
            if self.is_numeric() or self.is_datelike():
                self.set_at_idx(key, value)  # type: ignore[arg-type]
                return None
            raise ValueError(
                f"cannot set Series of dtype: {self.dtype} with list/tuple as value;"
                " use a scalar value"
            )
        if isinstance(key, Series):
            if key.dtype == Boolean:
                self._s = self.set(key, value)._s
            elif key.dtype == UInt64:
                self._s = self.set_at_idx(key.cast(UInt32), value)._s
            elif key.dtype == UInt32:
                self._s = self.set_at_idx(key, value)._s

        # TODO: implement for these types without casting to series
        elif _NUMPY_TYPE(key) and isinstance(key, np.ndarray):
            if key.dtype == np.bool_:
                # boolean numpy mask
                self._s = self.set_at_idx(np.argwhere(key)[:, 0], value)._s
            else:
                s = wrap_s(PySeries.new_u32("", np.array(key, np.uint32), True))
                self.__setitem__(s, value)
        elif isinstance(key, (list, tuple)):
            s = wrap_s(sequence_to_pyseries("", key, dtype=UInt32))
            self.__setitem__(s, value)
        elif isinstance(key, int) and not isinstance(key, bool):
            self.__setitem__([key], value)
        else:
            raise ValueError(f'cannot use "{key}" for indexing')

    def __array__(self, dtype: Any = None) -> np.ndarray[Any, Any]:
        if dtype:
            return self.to_numpy().__array__(dtype)
        else:
            return self.to_numpy().__array__()

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any
    ) -> Series:
        """Numpy universal functions."""
        if self._s.n_chunks() > 1:
            self._s.rechunk(in_place=True)

        s = self._s

        if method == "__call__":
            if not ufunc.nout == 1:
                raise NotImplementedError(
                    "Only ufuncs that return one 1D array, are supported."
                )

            args: list[int | float | np.ndarray[Any, Any]] = []

            for arg in inputs:
                if isinstance(arg, (int, float, np.ndarray)):
                    args.append(arg)
                elif isinstance(arg, Series):
                    args.append(arg.view(ignore_nulls=True))
                else:
                    raise ValueError(f"Unsupported type {type(arg)} for {arg}.")

            # Get minimum dtype needed to be able to cast all input arguments to the
            # same dtype.
            dtype_char_minimum = np.result_type(*args).char

            # Get all possible output dtypes for ufunc.
            # Input dtypes and output dtypes seem to always match for ufunc.types,
            # so pick all the different output dtypes.
            dtypes_ufunc = [
                input_output_type[-1]
                for input_output_type in ufunc.types
                if supported_numpy_char_code(input_output_type[-1])
            ]

            # Get the first ufunc dtype from all possible ufunc dtypes for which
            # the input arguments can be safely cast to that ufunc dtype.
            for dtype_ufunc in dtypes_ufunc:
                if np.can_cast(dtype_char_minimum, dtype_ufunc):
                    dtype_char_minimum = dtype_ufunc
                    break

            # Override minimum dtype if requested.
            dtype_char = (
                np.dtype(kwargs.pop("dtype")).char
                if "dtype" in kwargs
                else dtype_char_minimum
            )

            f = get_ffi_func("apply_ufunc_<>", numpy_char_code_to_dtype(dtype_char), s)

            if f is None:
                raise NotImplementedError(
                    "Could not find "
                    f"`apply_ufunc_{numpy_char_code_to_dtype(dtype_char)}`."
                )

            series = f(lambda out: ufunc(*args, out=out, dtype=dtype_char, **kwargs))
            return wrap_s(series)
        else:
            raise NotImplementedError(
                "Only `__call__` is implemented for numpy ufuncs on a Series, got"
                f" `{method}`."
            )

    def _repr_html_(self) -> str:
        """Format output data in HTML for display in Jupyter Notebooks."""
        return self.to_frame()._repr_html_(from_series=True)

    def estimated_size(self, unit: SizeUnit = "b") -> int | float:
        """
        Return an estimation of the total (heap) allocated size of the Series.

        Estimated size is given in the specified unit (bytes by default).

        This estimation is the sum of the size of its buffers, validity, including
        nested arrays. Multiple arrays may share buffers and bitmaps. Therefore, the
        size of 2 arrays is not the sum of the sizes computed from this function. In
        particular, [`StructArray`]'s size is an upper bound.

        When an array is sliced, its allocated size remains constant because the buffer
        unchanged. However, this function will yield a smaller number. This is because
        this function returns the visible size of the buffer, not its total capacity.

        FFI buffers are included in this estimation.

        Parameters
        ----------
        unit : {'b', 'kb', 'mb', 'gb', 'tb'}
            Scale the returned size to the given unit.

        Examples
        --------
        >>> s = pl.Series("values", list(range(1_000_000)), dtype=pl.UInt32)
        >>> s.estimated_size()
        4000000
        >>> s.estimated_size("mb")
        3.814697265625

        """
        sz = self._s.estimated_size()
        return scale_bytes(sz, to=unit)

    def sqrt(self) -> Series:
        """
        Compute the square root of the elements.

        Syntactic sugar for

        >>> pl.Series([1, 2]) ** 0.5
        shape: (2,)
        Series: '' [f64]
        [
            1.0
            1.414214
        ]

        """

    def any(self) -> bool:
        """
        Check if any boolean value in the column is `True`.

        Returns
        -------
        Boolean literal

        """
        return self.to_frame().select(pli.col(self.name).any()).to_series()[0]

    def all(self) -> bool:
        """
        Check if all boolean values in the column are `True`.

        Returns
        -------
        Boolean literal

        """
        return self.to_frame().select(pli.col(self.name).all()).to_series()[0]

    def log(self, base: float = math.e) -> Series:
        """Compute the logarithm to a given base."""

    def log10(self) -> Series:
        """Compute the base 10 logarithm of the input array, element-wise."""

    def exp(self) -> Series:
        """Compute the exponential, element-wise."""

    def drop_nulls(self) -> Series:
        """Create a new Series that copies data from this Series without null values."""

    def drop_nans(self) -> Series:
        """Drop NaN values."""

    def to_frame(self, name: str | None = None) -> pli.DataFrame:
        """
        Cast this Series to a DataFrame.

        Parameters
        ----------
        name
            optionally name/rename the Series column in the new DataFrame.

        Examples
        --------
        >>> s = pl.Series("a", [123, 456])
        >>> df = s.to_frame()
        >>> df
        shape: (2, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 123 │
        ├╌╌╌╌╌┤
        │ 456 │
        └─────┘

        >>> df = s.to_frame("xyz")
        >>> df
        shape: (2, 1)
        ┌─────┐
        │ xyz │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 123 │
        ├╌╌╌╌╌┤
        │ 456 │
        └─────┘

        """
        if isinstance(name, str):
            return pli.wrap_df(PyDataFrame([self.rename(name)._s]))
        return pli.wrap_df(PyDataFrame([self._s]))

    def describe(self) -> pli.DataFrame:
        """
        Quick summary statistics of a series.

        Series with mixed datatypes will return summary statistics for the datatype of
        the first value.

        Returns
        -------
        Dictionary with summary statistics of a Series.

        Examples
        --------
        >>> series_num = pl.Series([1, 2, 3, 4, 5])
        >>> series_num.describe()
        shape: (6, 2)
        ┌────────────┬──────────┐
        │ statistic  ┆ value    │
        │ ---        ┆ ---      │
        │ str        ┆ f64      │
        ╞════════════╪══════════╡
        │ min        ┆ 1.0      │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ max        ┆ 5.0      │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ null_count ┆ 0.0      │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ mean       ┆ 3.0      │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ std        ┆ 1.581139 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ count      ┆ 5.0      │
        └────────────┴──────────┘

        >>> series_str = pl.Series(["a", "a", None, "b", "c"])
        >>> series_str.describe()
        shape: (3, 2)
        ┌────────────┬───────┐
        │ statistic  ┆ value │
        │ ---        ┆ ---   │
        │ str        ┆ i64   │
        ╞════════════╪═══════╡
        │ unique     ┆ 4     │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ null_count ┆ 1     │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ count      ┆ 5     │
        └────────────┴───────┘

        """
        stats: dict[str, float | None | int | str | date | datetime | timedelta]

        if self.len() == 0:
            raise ValueError("Series must contain at least one value")
        elif self.is_numeric():
            s = self.cast(Float64)
            stats = {
                "min": s.min(),
                "max": s.max(),
                "null_count": s.null_count(),
                "mean": s.mean(),
                "std": s.std(),
                "count": s.len(),
            }
        elif self.is_boolean():
            stats = {
                "sum": self.sum(),
                "null_count": self.null_count(),
                "count": self.len(),
            }
        elif self.is_utf8():
            stats = {
                "unique": len(self.unique()),
                "null_count": self.null_count(),
                "count": self.len(),
            }
        elif self.is_datelike():
            # we coerce all to string, because a polars column
            # only has a single dtype and dates: datetime and count: int don't match
            stats = {
                "min": str(self.dt.min()),
                "max": str(self.dt.max()),
                "null_count": str(self.null_count()),
                "count": str(self.len()),
            }
        else:
            raise TypeError("This type is not supported")

        return pli.DataFrame(
            {"statistic": list(stats.keys()), "value": list(stats.values())}
        )

    def sum(self) -> int | float:
        """
        Reduce this Series to the sum value.

        Notes
        -----
        Dtypes in {Int8, UInt8, Int16, UInt16} are cast to
        Int64 before summing to prevent overflow issues.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.sum()
        6

        """
        return self._s.sum()

    def mean(self) -> int | float:
        """
        Reduce this Series to the mean value.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.mean()
        2.0

        """
        return self._s.mean()

    def product(self) -> int | float:
        """Reduce this Series to the product value."""
        return self.to_frame().select(pli.col(self.name).product()).to_series()[0]

    def min(self) -> int | float | date | datetime | timedelta | str:
        """
        Get the minimal value in this Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.min()
        1

        """
        return self._s.min()

    def max(self) -> int | float | date | datetime | timedelta | str:
        """
        Get the maximum value in this Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.max()
        3

        """
        return self._s.max()

    def nan_max(self) -> int | float | date | datetime | timedelta | str:
        """
        Get maximum value, but propagate/poison encountered NaN values.

        This differs from numpy's `nanmax` as numpy defaults to propagating NaN values,
        whereas polars defaults to ignoring them.

        """
        return self.to_frame().select(pli.col(self.name).nan_max())[0, 0]

    def nan_min(self) -> int | float | date | datetime | timedelta | str:
        """
        Get minimum value, but propagate/poison encountered NaN values.

        This differs from numpy's `nanmax` as numpy defaults to propagating NaN values,
        whereas polars defaults to ignoring them.

        """
        return self.to_frame().select(pli.col(self.name).nan_min())[0, 0]

    def std(self, ddof: int = 1) -> float | None:
        """
        Get the standard deviation of this Series.

        Parameters
        ----------
        ddof
            “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
            where N represents the number of elements.
            By default ddof is 1.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.std()
        1.0

        """
        if not self.is_numeric():
            return None
        return self.to_frame().select(pli.col(self.name).std(ddof)).to_series()[0]

    def var(self, ddof: int = 1) -> float | None:
        """
        Get variance of this Series.

        Parameters
        ----------
        ddof
            “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
            where N represents the number of elements.
            By default ddof is 1.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.var()
        1.0

        """
        if not self.is_numeric():
            return None
        return self.to_frame().select(pli.col(self.name).var(ddof)).to_series()[0]

    def median(self) -> float:
        """
        Get the median of this Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.median()
        2.0

        """
        return self._s.median()

    def quantile(
        self, quantile: float, interpolation: RollingInterpolationMethod = "nearest"
    ) -> float:
        """
        Get the quantile value of this Series.

        Parameters
        ----------
        quantile
            Quantile between 0.0 and 1.0.
        interpolation : {'nearest', 'higher', 'lower', 'midpoint', 'linear'}
            Interpolation method.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.quantile(0.5)
        2.0

        """
        return self._s.quantile(quantile, interpolation)

    def to_dummies(self) -> pli.DataFrame:
        """
        Get dummy variables.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.to_dummies()
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ a_1 ┆ a_2 ┆ a_3 │
        │ --- ┆ --- ┆ --- │
        │ u8  ┆ u8  ┆ u8  │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 0   ┆ 0   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 0   ┆ 1   ┆ 0   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 0   ┆ 0   ┆ 1   │
        └─────┴─────┴─────┘

        """
        return pli.wrap_df(self._s.to_dummies())

    def value_counts(self, sort: bool = False) -> pli.DataFrame:
        """
        Count the unique values in a Series.

        Parameters
        ----------
        sort
            Ensure the output is sorted from most values to least.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 2, 3])
        >>> s.value_counts().sort(by="a")
        shape: (3, 2)
        ┌─────┬────────┐
        │ a   ┆ counts │
        │ --- ┆ ---    │
        │ i64 ┆ u32    │
        ╞═════╪════════╡
        │ 1   ┆ 1      │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
        │ 2   ┆ 2      │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
        │ 3   ┆ 1      │
        └─────┴────────┘

        """
        return pli.wrap_df(self._s.value_counts(sort))

    def unique_counts(self) -> Series:
        """
        Return a count of the unique values in the order of appearance.

        Examples
        --------
        >>> s = pl.Series("id", ["a", "b", "b", "c", "c", "c"])
        >>> s.unique_counts()
        shape: (3,)
        Series: 'id' [u32]
        [
            1
            2
            3
        ]

        """

    def entropy(self, base: float = math.e, normalize: bool = False) -> float | None:
        """
        Computes the entropy.

        Uses the formula ``-sum(pk * log(pk)`` where ``pk`` are discrete probabilities.

        Parameters
        ----------
        base
            Given base, defaults to `e`
        normalize
            Normalize pk if it doesn't sum to 1.

        Examples
        --------
        >>> a = pl.Series([0.99, 0.005, 0.005])
        >>> a.entropy(normalize=True)
        0.06293300616044681
        >>> b = pl.Series([0.65, 0.10, 0.25])
        >>> b.entropy(normalize=True)
        0.8568409950394724

        """
        return pli.select(pli.lit(self).entropy(base, normalize)).to_series()[0]

    def cumulative_eval(
        self, expr: pli.Expr, min_periods: int = 1, parallel: bool = False
    ) -> Series:
        """
        Run an expression over a sliding window that increases `1` slot every iteration.

        Parameters
        ----------
        expr
            Expression to evaluate
        min_periods
            Number of valid values there should be in the window before the expression
            is evaluated. valid values = `length - null_count`
        parallel
            Run in parallel. Don't do this in a groupby or another operation that
            already has much parallelization.

        Warnings
        --------
        This functionality is experimental and may change without it being considered a
        breaking change.

        This can be really slow as it can have `O(n^2)` complexity. Don't use this
        for operations that visit all elements.

        Examples
        --------
        >>> s = pl.Series("values", [1, 2, 3, 4, 5])
        >>> s.cumulative_eval(pl.element().first() - pl.element().last() ** 2)
        shape: (5,)
        Series: 'values' [f64]
        [
            0.0
            -3.0
            -8.0
            -15.0
            -24.0
        ]

        """

    def alias(self, name: str) -> Series:
        """
        Return a copy of the Series with a new alias/name.

        Parameters
        ----------
        name
            New name.

        Examples
        --------
        >>> srs = pl.Series("x", [1, 2, 3])
        >>> new_aliased_srs = srs.alias("y")

        """
        s = self.clone()
        s._s.rename(name)
        return s

    def rename(self, name: str, in_place: bool = False) -> Series:
        """
        Rename this Series.

        Parameters
        ----------
        name
            New name.
        in_place
            Modify the Series in-place.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.rename("b")
        shape: (3,)
        Series: 'b' [i64]
        [
                1
                2
                3
        ]

        """
        if in_place:
            self._s.rename(name)
            return self
        else:
            return self.alias(name)

    def chunk_lengths(self) -> list[int]:
        """
        Get the length of each individual chunk.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s2 = pl.Series("a", [4, 5, 6])

        Concatenate Series with rechunk = True

        >>> pl.concat([s, s2]).chunk_lengths()
        [6]

        Concatenate Series with rechunk = False

        >>> pl.concat([s, s2], rechunk=False).chunk_lengths()
        [3, 3]

        """
        return self._s.chunk_lengths()

    def n_chunks(self) -> int:
        """
        Get the number of chunks that this Series contains.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.n_chunks()
        1
        >>> s2 = pl.Series("a", [4, 5, 6])

        Concatenate Series with rechunk = True

        >>> pl.concat([s, s2]).n_chunks()
        1

        Concatenate Series with rechunk = False

        >>> pl.concat([s, s2], rechunk=False).n_chunks()
        2

        """
        return self._s.n_chunks()

    def cumsum(self, reverse: bool = False) -> Series:
        """
        Get an array with the cumulative sum computed at every element.

        Parameters
        ----------
        reverse
            reverse the operation.

        Notes
        -----
        Dtypes in {Int8, UInt8, Int16, UInt16} are cast to
        Int64 before summing to prevent overflow issues.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.cumsum()
        shape: (3,)
        Series: 'a' [i64]
        [
            1
            3
            6
        ]

        """

    def cummin(self, reverse: bool = False) -> Series:
        """
        Get an array with the cumulative min computed at every element.

        Parameters
        ----------
        reverse
            reverse the operation.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.cummin()
        shape: (3,)
        Series: 'a' [i64]
        [
            1
            1
            1
        ]

        """

    def cummax(self, reverse: bool = False) -> Series:
        """
        Get an array with the cumulative max computed at every element.

        Parameters
        ----------
        reverse
            reverse the operation.

        Examples
        --------
        >>> s = pl.Series("a", [3, 5, 1])
        >>> s.cummax()
        shape: (3,)
        Series: 'a' [i64]
        [
            3
            5
            5
        ]

        """

    def cumprod(self, reverse: bool = False) -> Series:
        """
        Get an array with the cumulative product computed at every element.

        Parameters
        ----------
        reverse
            reverse the operation.

        Notes
        -----
        Dtypes in {Int8, UInt8, Int16, UInt16} are cast to
        Int64 before summing to prevent overflow issues.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.cumprod()
        shape: (3,)
        Series: 'a' [i64]
        [
            1
            2
            6
        ]

        """

    def limit(self, n: int = 10) -> Series:
        """
        Get the first `n` rows.

        Alias for :func:`Series.head`.

        Parameters
        ----------
        n
            Number of rows to return.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.head(2)
        shape: (2,)
        Series: 'a' [i64]
        [
                1
                2
        ]

        """
        return self.to_frame().select(pli.col(self.name).limit(n)).to_series()

    def slice(self, offset: int, length: int | None = None) -> Series:
        """
        Get a slice of this Series.

        Parameters
        ----------
        offset
            Start index. Negative indexing is supported.
        length
            Length of the slice. If set to ``None``, all rows starting at the offset
            will be selected.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4])
        >>> s.slice(1, 2)
        shape: (2,)
        Series: 'a' [i64]
        [
                2
                3
        ]

        """

    def append(self, other: Series, append_chunks: bool = True) -> Series:
        """
        Append a Series to this one.

        Parameters
        ----------
        other
            Series to append.
        append_chunks
            If set to `True` the append operation will add the chunks from `other` to
            self. This is super cheap.

            If set to `False` the append operation will do the same as
            `DataFrame.extend` which extends the memory backed by this `Series` with
            the values from `other`.

            Different from `append chunks`, `extend` appends the data from `other` to
            the underlying memory locations and thus may cause a reallocation (which are
            expensive).

            If this does not cause a reallocation, the resulting data structure will not
            have any extra chunks and thus will yield faster queries.

            Prefer `extend` over `append_chunks` when you want to do a query after a
            single append. For instance during online operations where you add `n` rows
            and rerun a query.

            Prefer `append_chunks` over `extend` when you want to append many times
            before doing a query. For instance when you read in multiple files and when
            to store them in a single `Series`. In the latter case, finish the sequence
            of `append_chunks` operations with a `rechunk`.


        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s2 = pl.Series("b", [4, 5, 6])
        >>> s.append(s2)
        shape: (6,)
        Series: 'a' [i64]
        [
            1
            2
            3
            4
            5
            6
        ]

        """
        try:
            if append_chunks:
                self._s.append(other._s)
            else:
                self._s.extend(other._s)
        except RuntimeError as e:
            if str(e) == "Already mutably borrowed":
                self.append(other.clone(), append_chunks)
            else:
                raise e
        return self

    def filter(self, predicate: Series | list[bool]) -> Series:
        """
        Filter elements by a boolean mask.

        Parameters
        ----------
        predicate
            Boolean mask.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> mask = pl.Series("", [True, False, True])
        >>> s.filter(mask)
        shape: (2,)
        Series: 'a' [i64]
        [
                1
                3
        ]

        """
        if isinstance(predicate, list):
            predicate = Series("", predicate)
        return wrap_s(self._s.filter(predicate._s))

    def head(self, n: int = 10) -> Series:
        """
        Get the first `n` rows.

        Parameters
        ----------
        n
            Number of rows to return.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.head(2)
        shape: (2,)
        Series: 'a' [i64]
        [
                1
                2
        ]

        """
        return self.to_frame().select(pli.col(self.name).head(n)).to_series()

    def tail(self, n: int = 10) -> Series:
        """
        Get the last `n` rows.

        Parameters
        ----------
        n
            Number of rows to return.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.tail(2)
        shape: (2,)
        Series: 'a' [i64]
        [
                2
                3
        ]

        """
        return self.to_frame().select(pli.col(self.name).tail(n)).to_series()

    def take_every(self, n: int) -> Series:
        """
        Take every nth value in the Series and return as new Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4])
        >>> s.take_every(2)
        shape: (2,)
        Series: 'a' [i64]
        [
            1
            3
        ]

        """

    def sort(self, reverse: bool = False, *, in_place: bool = False) -> Series:
        """
        Sort this Series.

        Parameters
        ----------
        reverse
            Reverse sort.
        in_place
            Sort in place.

        Examples
        --------
        >>> s = pl.Series("a", [1, 3, 4, 2])
        >>> s.sort()
        shape: (4,)
        Series: 'a' [i64]
        [
                1
                2
                3
                4
        ]
        >>> s.sort(reverse=True)
        shape: (4,)
        Series: 'a' [i64]
        [
                4
                3
                2
                1
        ]

        """
        if in_place:
            self._s = self._s.sort(reverse)
            return self
        else:
            return wrap_s(self._s.sort(reverse))

    def top_k(self, k: int = 5, reverse: bool = False) -> Series:
        r"""
        Return the `k` largest elements.

        If 'reverse=True` the smallest elements will be given.

        This has time complexity:

        .. math:: O(n + k \\log{}n - \frac{k}{2})

        Parameters
        ----------
        k
            Number of elements to return.
        reverse
            Return the smallest elements.

        """

    def arg_sort(self, reverse: bool = False, nulls_last: bool = False) -> Series:
        """
        Get the index values that would sort this Series.

        Parameters
        ----------
        reverse
            Sort in reverse (descending) order.
        nulls_last
            Place null values last instead of first.

        Examples
        --------
        >>> s = pl.Series("a", [5, 3, 4, 1, 2])
        >>> s.arg_sort()
        shape: (5,)
        Series: 'a' [u32]
        [
            3
            4
            1
            2
            0
        ]

        """

    def argsort(self, reverse: bool = False, nulls_last: bool = False) -> Series:
        """
        Get the index values that would sort this Series.

        Alias for :func:`Series.arg_sort`.

        Parameters
        ----------
        reverse
            Sort in reverse (descending) order.
        nulls_last
            Place null values last instead of first.

        """

    def arg_unique(self) -> Series:
        """
        Get unique index as Series.

        Returns
        -------
        Series

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 2, 3])
        >>> s.arg_unique()
        shape: (3,)
        Series: 'a' [u32]
        [
                0
                1
                3
        ]

        """

    def arg_min(self) -> int | None:
        """
        Get the index of the minimal value.

        Returns
        -------
        Integer

        Examples
        --------
        >>> s = pl.Series("a", [3, 2, 1])
        >>> s.arg_min()
        2

        """
        return self._s.arg_min()

    def arg_max(self) -> int | None:
        """
        Get the index of the maximal value.

        Returns
        -------
        Integer

        Examples
        --------
        >>> s = pl.Series("a", [3, 2, 1])
        >>> s.arg_max()
        0

        """
        return self._s.arg_max()

    def search_sorted(self, element: int | float) -> int:
        """
        Find indices where elements should be inserted to maintain order.

        .. math:: a[i-1] < v <= a[i]

        Parameters
        ----------
        element
            Expression or scalar value.

        """
        return pli.select(pli.lit(self).search_sorted(element))[0, 0]

    def unique(self, maintain_order: bool = False) -> Series:
        """
        Get unique elements in series.

        Parameters
        ----------
        maintain_order
            Maintain order of data. This requires more work.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 2, 3])
        >>> s.unique().sort()
        shape: (3,)
        Series: 'a' [i64]
        [
            1
            2
            3
        ]

        """

    def take(
        self, indices: int | list[int] | pli.Expr | Series | np.ndarray[Any, Any]
    ) -> Series:
        """
        Take values by index.

        Parameters
        ----------
        indices
            Index location used for selection.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4])
        >>> s.take([1, 3])
        shape: (2,)
        Series: 'a' [i64]
        [
                2
                4
        ]

        """
        return self.to_frame().select(pli.col(self.name).take(indices)).to_series()

    def null_count(self) -> int:
        """Count the null values in this Series."""
        return self._s.null_count()

    def has_validity(self) -> bool:
        """
        Return True if the Series has a validity bitmask.

        If there is none, it means that there are no null values.
        Use this to swiftly assert a Series does not have null values.

        """
        return self._s.has_validity()

    def is_empty(self) -> bool:
        """
        Check if the Series is empty.

        Examples
        --------
        >>> s = pl.Series("a", [], dtype=pl.Float32)
        >>> s.is_empty()
        True

        """
        return self.len() == 0

    def is_null(self) -> Series:
        """
        Returns a boolean Series indicating which values are null.

        Returns
        -------
        Boolean Series

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, None])
        >>> s.is_null()
        shape: (4,)
        Series: 'a' [bool]
        [
            false
            false
            false
            true
        ]

        """

    def is_not_null(self) -> Series:
        """
        Returns a boolean Series indicating which values are not null.

        Returns
        -------
        Boolean Series

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, None])
        >>> s.is_not_null()
        shape: (4,)
        Series: 'a' [bool]
        [
            true
            true
            true
            false
        ]

        """

    def is_finite(self) -> Series:
        """
        Returns a boolean Series indicating which values are finite.

        Returns
        -------
        Boolean Series

        Examples
        --------
        >>> import numpy as np
        >>> s = pl.Series("a", [1.0, 2.0, np.inf])
        >>> s.is_finite()
        shape: (3,)
        Series: 'a' [bool]
        [
                true
                true
                false
        ]

        """

    def is_infinite(self) -> Series:
        """
        Returns a boolean Series indicating which values are infinite.

        Returns
        -------
        Boolean Series

        Examples
        --------
        >>> import numpy as np
        >>> s = pl.Series("a", [1.0, 2.0, np.inf])
        >>> s.is_infinite()
        shape: (3,)
        Series: 'a' [bool]
        [
                false
                false
                true
        ]

        """

    def is_nan(self) -> Series:
        """
        Returns a boolean Series indicating which values are not NaN.

        Returns
        -------
        Boolean Series

        Examples
        --------
        >>> import numpy as np
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, np.NaN])
        >>> s.is_nan()
        shape: (4,)
        Series: 'a' [bool]
        [
                false
                false
                false
                true
        ]

        """

    def is_not_nan(self) -> Series:
        """
        Returns a boolean Series indicating which values are not NaN.

        Returns
        -------
        Boolean Series

        Examples
        --------
        >>> import numpy as np
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, np.NaN])
        >>> s.is_not_nan()
        shape: (4,)
        Series: 'a' [bool]
        [
                true
                true
                true
                false
        ]

        """

    def is_in(self, other: Series | Sequence[Any]) -> Series:
        """
        Check if elements of this Series are in the other Series.

        Returns
        -------
        Boolean Series

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s2 = pl.Series("b", [2, 4])
        >>> s2.is_in(s)
        shape: (2,)
        Series: 'b' [bool]
        [
                true
                false
        ]

        >>> # check if some values are a member of sublists
        >>> sets = pl.Series("sets", [[1, 2, 3], [1, 2], [9, 10]])
        >>> optional_members = pl.Series("optional_members", [1, 2, 3])
        >>> print(sets)
        shape: (3,)
        Series: 'sets' [list]
        [
            [1, 2, 3]
            [1, 2]
            [9, 10]
        ]
        >>> print(optional_members)
        shape: (3,)
        Series: 'optional_members' [i64]
        [
            1
            2
            3
        ]
        >>> optional_members.is_in(sets)
        shape: (3,)
        Series: 'optional_members' [bool]
        [
            true
            true
            false
        ]

        """

    def arg_true(self) -> Series:
        """
        Get index values where Boolean Series evaluate True.

        Returns
        -------
        UInt32 Series

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> (s == 2).arg_true()
        shape: (1,)
        Series: 'a' [u32]
        [
                1
        ]

        """
        return pli.arg_where(self, eager=True)

    def is_unique(self) -> Series:
        """
        Get mask of all unique values.

        Returns
        -------
        Boolean Series

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 2, 3])
        >>> s.is_unique()
        shape: (4,)
        Series: 'a' [bool]
        [
                true
                false
                false
                true
        ]

        """

    def is_first(self) -> Series:
        """
        Get a mask of the first unique value.

        Returns
        -------
        Boolean Series

        """

    def is_duplicated(self) -> Series:
        """
        Get mask of all duplicated values.

        Returns
        -------
        Boolean Series

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 2, 3])
        >>> s.is_duplicated()
        shape: (4,)
        Series: 'a' [bool]
        [
                false
                true
                true
                false
        ]

        """

    def explode(self) -> Series:
        """
        Explode a list or utf8 Series.

        This means that every item is expanded to a new row.

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2], [3, 4], [9, 10]])
        >>> s.explode()
        shape: (6,)
        Series: 'a' [i64]
        [
                1
                2
                3
                4
                9
                10
        ]

        Returns
        -------
        Exploded Series of same dtype

        """

    def series_equal(
        self, other: Series, null_equal: bool = True, strict: bool = False
    ) -> bool:
        """
        Check if series is equal with another Series.

        Parameters
        ----------
        other
            Series to compare with.
        null_equal
            Consider null values as equal.
        strict
            Don't allow different numerical dtypes, e.g. comparing `pl.UInt32` with a
            `pl.Int64` will return `False`.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s2 = pl.Series("b", [4, 5, 6])
        >>> s.series_equal(s)
        True
        >>> s.series_equal(s2)
        False

        """
        return self._s.series_equal(other._s, null_equal, strict)

    def len(self) -> int:
        """
        Length of this Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.len()
        3

        """
        return self._s.len()

    def cast(
        self,
        dtype: (PolarsDataType | type[int] | type[float] | type[str] | type[bool]),
        strict: bool = True,
    ) -> Series:
        """
        Cast between data types.

        Parameters
        ----------
        dtype
            DataType to cast to
        strict
            Throw an error if a cast could not be done for instance due to an overflow

        Examples
        --------
        >>> s = pl.Series("a", [True, False, True])
        >>> s
        shape: (3,)
        Series: 'a' [bool]
        [
            true
            false
            true
        ]

        >>> s.cast(pl.UInt32)
        shape: (3,)
        Series: 'a' [u32]
        [
            1
            0
            1
        ]

        """
        # Do not dispatch cast as it is expensive and used in other functions.
        dtype = py_type_to_dtype(dtype)
        return wrap_s(self._s.cast(dtype, strict))

    def to_physical(self) -> Series:
        """
        Cast to physical representation of the logical dtype.

        - :func:`polars.datatypes.Date` -> :func:`polars.datatypes.Int32`
        - :func:`polars.datatypes.Datetime` -> :func:`polars.datatypes.Int64`
        - :func:`polars.datatypes.Time` -> :func:`polars.datatypes.Int64`
        - :func:`polars.datatypes.Duration` -> :func:`polars.datatypes.Int64`
        - :func:`polars.datatypes.Categorical` -> :func:`polars.datatypes.UInt32`
        - Other data types will be left unchanged.

        Examples
        --------
        Replicating the pandas
        `pd.Series.factorize
        <https://pandas.pydata.org/docs/reference/api/pandas.Series.factorize.html>`_
        method.

        >>> s = pl.Series("values", ["a", None, "x", "a"])
        >>> s.cast(pl.Categorical).to_physical()
        shape: (4,)
        Series: 'values' [u32]
        [
            0
            null
            1
            0
        ]

        """

    def to_list(self, use_pyarrow: bool = False) -> list[Any | None]:
        """
        Convert this Series to a Python List. This operation clones data.

        Parameters
        ----------
        use_pyarrow
            Use pyarrow for the conversion.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.to_list()
        [1, 2, 3]
        >>> type(s.to_list())
        <class 'list'>

        """
        if use_pyarrow:
            return self.to_arrow().to_pylist()
        return self._s.to_list()

    def rechunk(self, in_place: bool = False) -> Series:
        """
        Create a single chunk of memory for this Series.

        Parameters
        ----------
        in_place
            In place or not.

        """
        opt_s = self._s.rechunk(in_place)
        return self if in_place else wrap_s(opt_s)

    def reverse(self) -> Series:
        """
        Return Series in reverse order.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3], dtype=pl.Int8)
        >>> s.reverse()
        shape: (3,)
        Series: 'a' [i8]
        [
            3
            2
            1
        ]

        """

    def is_numeric(self) -> bool:
        """
        Check if this Series datatype is numeric.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.is_numeric()
        True

        """
        return self.dtype in (
            Int8,
            Int16,
            Int32,
            Int64,
            UInt8,
            UInt16,
            UInt32,
            UInt64,
            Float32,
            Float64,
        )

    def is_datelike(self) -> bool:
        """
        Check if this Series datatype is datelike.

        Examples
        --------
        >>> from datetime import date
        >>> s = pl.Series([date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3)])
        >>> s.is_datelike()
        True

        """
        return self.dtype in (Date, Datetime, Duration, Time)

    def is_float(self) -> bool:
        """
        Check if this Series has floating point numbers.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 2.0, 3.0])
        >>> s.is_float()
        True

        """
        return self.dtype in (Float32, Float64)

    def is_boolean(self) -> bool:
        """
        Check if this Series is a Boolean.

        Examples
        --------
        >>> s = pl.Series("a", [True, False, True])
        >>> s.is_boolean()
        True

        """
        return self.dtype is Boolean

    def is_utf8(self) -> bool:
        """
        Check if this Series datatype is a Utf8.

        Examples
        --------
        >>> s = pl.Series("x", ["a", "b", "c"])
        >>> s.is_utf8()
        True

        """
        return self.dtype is Utf8

    def view(self, ignore_nulls: bool = False) -> SeriesView:
        """
        Get a view into this Series data with a numpy array.

        This operation doesn't clone data, but does not include missing values.
        Don't use this unless you know what you are doing.

        Parameters
        ----------
        ignore_nulls
            If True then nulls are converted to 0.
            If False then an Exception is raised if nulls are present.

        Examples
        --------
        >>> s = pl.Series("a", [1, None])
        >>> s.view(ignore_nulls=True)
        SeriesView([1, 0])

        """
        if not ignore_nulls:
            assert not self.has_validity()

        from polars.internals.series._numpy import SeriesView, _ptr_to_numpy

        ptr_type = dtype_to_ctype(self.dtype)
        ptr = self._s.as_single_ptr()
        array = _ptr_to_numpy(ptr, self.len(), ptr_type)
        array.setflags(write=False)
        return SeriesView(array, self)

    def to_numpy(
        self, *args: Any, zero_copy_only: bool = False, writable: bool = False
    ) -> np.ndarray[Any, Any]:
        """
        Convert this Series to numpy. This operation clones data but is completely safe.

        If you want a zero-copy view and know what you are doing, use `.view()`.

        Notes
        -----
        If you are attempting to convert Utf8 to an array you'll need to install
        `pyarrow`.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> arr = s.to_numpy()
        >>> arr  # doctest: +IGNORE_RESULT
        array([1, 2, 3], dtype=int64)
        >>> type(arr)
        <class 'numpy.ndarray'>

        Parameters
        ----------
        args
            args will be sent to pyarrow.Array.to_numpy.
        zero_copy_only
            If True, an exception will be raised if the conversion to a numpy
            array would require copying the underlying data (e.g. in presence
            of nulls, or for non-primitive types).
        writable
            For numpy arrays created with zero copy (view on the Arrow data),
            the resulting array is not writable (Arrow data is immutable).
            By setting this to True, a copy of the array is made to ensure
            it is writable.
        kwargs
            kwargs will be sent to pyarrow.Array.to_numpy

        """

        def convert_to_date(arr: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            if self.dtype == Date:
                tp = "datetime64[D]"
            elif self.dtype == Duration:
                tp = f"timedelta64[{self.time_unit}]"
            else:
                tp = f"datetime64[{self.time_unit}]"
            return arr.astype(tp)

        if _PYARROW_AVAILABLE and not self.is_datelike():
            return self.to_arrow().to_numpy(
                *args, zero_copy_only=zero_copy_only, writable=writable
            )
        else:
            if not self.has_validity():
                if self.is_datelike():
                    np_array = convert_to_date(self.view(ignore_nulls=True))
                else:
                    np_array = self.view(ignore_nulls=True)
            elif self.is_datelike():
                np_array = convert_to_date(self._s.to_numpy())
            else:
                np_array = self._s.to_numpy()

            if writable and not np_array.flags.writeable:
                return np_array.copy()
            else:
                return np_array

    def to_arrow(self) -> pa.Array:
        """
        Get the underlying Arrow Array.

        If the Series contains only a single chunk this operation is zero copy.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s = s.to_arrow()
        >>> s  # doctest: +ELLIPSIS
        <pyarrow.lib.Int64Array object at ...>
        [
          1,
          2,
          3
        ]

        """
        return self._s.to_arrow()

    def to_pandas(self) -> pd.Series:
        """
        Convert this Series to a pandas Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.to_pandas()
        0    1
        1    2
        2    3
        Name: a, dtype: int64

        """
        pd_series = self.to_arrow().to_pandas()
        pd_series.name = self.name
        return pd_series

    def set(self, filter: Series, value: int | float | str) -> Series:
        """
        Set masked values.

        Parameters
        ----------
        filter
            Boolean mask.
        value
            Value with which to replace the masked values.

        Notes
        -----
        Use of this function is frequently an anti-pattern, as it can
        block optimisation (predicate pushdown, etc). Consider using
        `pl.when(predicate).then(value).otherwise(self)` instead.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.set(s == 2, 10)
        shape: (3,)
        Series: 'a' [i64]
        [
                1
                10
                3
        ]

        It is better to implement this as follows:

        >>> s.to_frame().select(
        ...     pl.when(pl.col("a") == 2).then(10).otherwise(pl.col("a"))
        ... )
        shape: (3, 1)
        ┌─────────┐
        │ literal │
        │ ---     │
        │ i64     │
        ╞═════════╡
        │ 1       │
        ├╌╌╌╌╌╌╌╌╌┤
        │ 10      │
        ├╌╌╌╌╌╌╌╌╌┤
        │ 3       │
        └─────────┘

        """
        f = get_ffi_func("set_with_mask_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(filter._s, value))

    def set_at_idx(
        self,
        idx: Series | np.ndarray[Any, Any] | Sequence[int] | int,
        value: int
        | float
        | str
        | bool
        | Sequence[int]
        | Sequence[float]
        | Sequence[bool]
        | Sequence[str]
        | Sequence[date]
        | Sequence[datetime]
        | date
        | datetime
        | Series
        | None,
    ) -> Series:
        """
        Set values at the index locations.

        Parameters
        ----------
        idx
            Integers representing the index locations.
        value
            replacement values.

        Returns
        -------
        the series mutated

        Notes
        -----
        Use of this function is frequently an anti-pattern, as it can
        block optimisation (predicate pushdown, etc). Consider using
        `pl.when(predicate).then(value).otherwise(self)` instead.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.set_at_idx(1, 10)
        shape: (3,)
        Series: 'a' [i64]
        [
                1
                10
                3
        ]

        It is better to implement this as follows:

        >>> s.to_frame().with_row_count("row_nr").select(
        ...     pl.when(pl.col("row_nr") == 1).then(10).otherwise(pl.col("a"))
        ... )
        shape: (3, 1)
        ┌─────────┐
        │ literal │
        │ ---     │
        │ i64     │
        ╞═════════╡
        │ 1       │
        ├╌╌╌╌╌╌╌╌╌┤
        │ 10      │
        ├╌╌╌╌╌╌╌╌╌┤
        │ 3       │
        └─────────┘

        """
        if isinstance(idx, int):
            idx = [idx]
        if len(idx) == 0:
            return self

        idx = Series("", idx)
        if isinstance(value, (int, float, bool, str)) or (value is None):
            value = Series("", [value])

            # if we need to set more than a single value, we extend it
            if len(idx) > 0:
                value = value.extend_constant(value[0], len(idx) - 1)
        elif not isinstance(value, Series):
            value = Series("", value)
        self._s.set_at_idx(idx._s, value._s)
        return self

    def cleared(self) -> Series:
        """
        Create an empty copy of the current Series.

        The copy has identical name/dtype but no data.

        See Also
        --------
        clone : Cheap deepcopy/clone.

        Examples
        --------
        >>> s = pl.Series("a", [None, True, False])
        >>> s.cleared()
        shape: (0,)
        Series: 'a' [bool]
        [
        ]

        """
        return self.limit(0) if len(self) > 0 else self.clone()

    def clone(self) -> Series:
        """
        Very cheap deepcopy/clone.

        See Also
        --------
        cleared : Create an empty copy of the current Series, with identical
            schema but no data.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.clone()
        shape: (3,)
        Series: 'a' [i64]
        [
                1
                2
                3
        ]

        """
        return wrap_s(self._s.clone())

    def fill_nan(self, fill_value: int | float | pli.Expr | None) -> Series:
        """
        Fill floating point NaN value with a fill value.

        Parameters
        ----------
        fill_value
            Value used to fill nan values.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, float("nan")])
        >>> s.fill_nan(0)
        shape: (4,)
        Series: 'a' [f64]
        [
                1.0
                2.0
                3.0
                0.0
        ]

        """

    def fill_null(
        self,
        value: Any | None = None,
        strategy: FillNullStrategy | None = None,
        limit: int | None = None,
    ) -> Series:
        """
        Fill null values using the specified value or strategy.

        Parameters
        ----------
        value
            Value used to fill null values.
        strategy : {None, 'forward', 'backward', 'min', 'max', 'mean', 'zero', 'one'}
            Strategy used to fill null values.
        limit
            Number of consecutive null values to fill when using the 'forward' or
            'backward' strategy.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, None])
        >>> s.fill_null(strategy="forward")
        shape: (4,)
        Series: 'a' [i64]
        [
            1
            2
            3
            3
        ]
        >>> s.fill_null(strategy="min")
        shape: (4,)
        Series: 'a' [i64]
        [
            1
            2
            3
            1
        ]
        >>> s = pl.Series("b", ["x", None, "z"])
        >>> s.fill_null(pl.lit(""))
        shape: (3,)
        Series: 'b' [str]
        [
            "x"
            ""
            "z"
        ]

        """

    def floor(self) -> Series:
        """
        Rounds down to the nearest integer value.

        Only works on floating point Series.

        Examples
        --------
        >>> s = pl.Series("a", [1.12345, 2.56789, 3.901234])
        >>> s.floor()
        shape: (3,)
        Series: 'a' [f64]
        [
                1.0
                2.0
                3.0
        ]

        """

    def ceil(self) -> Series:
        """
        Rounds up to the nearest integer value.

        Only works on floating point Series.

        Examples
        --------
        >>> s = pl.Series("a", [1.12345, 2.56789, 3.901234])
        >>> s.ceil()
        shape: (3,)
        Series: 'a' [f64]
        [
                2.0
                3.0
                4.0
        ]

        """

    def round(self, decimals: int) -> Series:
        """
        Round underlying floating point data by `decimals` digits.

        Examples
        --------
        >>> s = pl.Series("a", [1.12345, 2.56789, 3.901234])
        >>> s.round(2)
        shape: (3,)
        Series: 'a' [f64]
        [
                1.12
                2.57
                3.9
        ]

        Parameters
        ----------
        decimals
            number of decimals to round by.

        """

    def dot(self, other: Series | ArrayLike) -> float | None:
        """
        Compute the dot/inner product between two Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s2 = pl.Series("b", [4.0, 5.0, 6.0])
        >>> s.dot(s2)
        32.0

        Parameters
        ----------
        other
            Series (or array) to compute dot product with.

        """
        if not isinstance(other, Series):
            other = Series(other)
        if len(self) != len(other):
            n, m = len(self), len(other)
            raise ShapeError(f"Series length mismatch: expected {n}, found {m}")
        return self._s.dot(other._s)

    def mode(self) -> Series:
        """
        Compute the most occurring value(s).

        Can return multiple Values.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 2, 3])
        >>> s.mode()
        shape: (1,)
        Series: 'a' [i64]
        [
                2
        ]

        """

    def sign(self) -> Series:
        """
        Compute the element-wise indication of the sign.

        Examples
        --------
        >>> s = pl.Series("a", [-9.0, -0.0, 0.0, 4.0, None])
        >>> s.sign()
        shape: (5,)
        Series: 'a' [i64]
        [
                -1
                0
                0
                1
                null
        ]

        """

    def sin(self) -> Series:
        """
        Compute the element-wise value for the sine.

        Examples
        --------
        >>> import math
        >>> s = pl.Series("a", [0.0, math.pi / 2.0, math.pi])
        >>> s.sin()
        shape: (3,)
        Series: 'a' [f64]
        [
            0.0
            1.0
            1.2246e-16
        ]

        """

    def cos(self) -> Series:
        """
        Compute the element-wise value for the cosine.

        Examples
        --------
        >>> import math
        >>> s = pl.Series("a", [0.0, math.pi / 2.0, math.pi])
        >>> s.cos()
        shape: (3,)
        Series: 'a' [f64]
        [
            1.0
            6.1232e-17
            -1.0
        ]

        """

    def tan(self) -> Series:
        """
        Compute the element-wise value for the tangent.

        Examples
        --------
        >>> import math
        >>> s = pl.Series("a", [0.0, math.pi / 2.0, math.pi])
        >>> s.tan()
        shape: (3,)
        Series: 'a' [f64]
        [
            0.0
            1.6331e16
            -1.2246e-16
        ]

        """

    def arcsin(self) -> Series:
        """
        Compute the element-wise value for the inverse sine.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 0.0, -1.0])
        >>> s.arcsin()
        shape: (3,)
        Series: 'a' [f64]
        [
            1.570796
            0.0
            -1.570796
        ]

        """

    def arccos(self) -> Series:
        """
        Compute the element-wise value for the inverse cosine.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 0.0, -1.0])
        >>> s.arccos()
        shape: (3,)
        Series: 'a' [f64]
        [
            0.0
            1.570796
            3.141593
        ]

        """

    def arctan(self) -> Series:
        """
        Compute the element-wise value for the inverse tangent.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 0.0, -1.0])
        >>> s.arctan()
        shape: (3,)
        Series: 'a' [f64]
        [
            0.785398
            0.0
            -0.785398
        ]

        """

    def arcsinh(self) -> Series:
        """
        Compute the element-wise value for the inverse hyperbolic sine.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 0.0, -1.0])
        >>> s.arcsinh()
        shape: (3,)
        Series: 'a' [f64]
        [
            0.881374
            0.0
            -0.881374
        ]

        """

    def arccosh(self) -> Series:
        """
        Compute the element-wise value for the inverse hyperbolic cosine.

        Examples
        --------
        >>> s = pl.Series("a", [5.0, 1.0, 0.0, -1.0])
        >>> s.arccosh()
        shape: (4,)
        Series: 'a' [f64]
        [
            2.292432
            0.0
            NaN
            NaN
        ]

        """

    def arctanh(self) -> Series:
        """
        Compute the element-wise value for the inverse hyperbolic tangent.

        Examples
        --------
        >>> s = pl.Series("a", [2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.1])
        >>> s.arctanh()
        shape: (7,)
        Series: 'a' [f64]
        [
            NaN
            inf
            0.549306
            0.0
            -0.549306
            -inf
            NaN
        ]

        """

    def sinh(self) -> Series:
        """
        Compute the element-wise value for the hyperbolic sine.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 0.0, -1.0])
        >>> s.sinh()
        shape: (3,)
        Series: 'a' [f64]
        [
            1.175201
            0.0
            -1.175201
        ]

        """

    def cosh(self) -> Series:
        """
        Compute the element-wise value for the hyperbolic cosine.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 0.0, -1.0])
        >>> s.cosh()
        shape: (3,)
        Series: 'a' [f64]
        [
            1.543081
            1.0
            1.543081
        ]

        """

    def tanh(self) -> Series:
        """
        Compute the element-wise value for the hyperbolic tangent.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 0.0, -1.0])
        >>> s.tanh()
        shape: (3,)
        Series: 'a' [f64]
        [
            0.761594
            0.0
            -0.761594
        ]

        """

    def apply(
        self,
        func: Callable[[Any], Any],
        return_dtype: PolarsDataType | None = None,
        skip_nulls: bool = True,
    ) -> Series:
        """
        Apply a custom/user-defined function (UDF) over elements in this Series and
        return a new Series.

        If the function returns another datatype, the return_dtype arg should be set,
        otherwise the method will fail.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.apply(lambda x: x + 10)
        shape: (3,)
        Series: 'a' [i64]
        [
                11
                12
                13
        ]

        Parameters
        ----------
        func
            function or lambda.
        return_dtype
            Output datatype. If none is given, the same datatype as this Series will be
            used.
        skip_nulls
            Nulls will be skipped and not passed to the python function.
            This is faster because python can be skipped and because we call
            more specialized functions.

        Returns
        -------
        Series

        """  # noqa: D400,D205
        if return_dtype is None:
            pl_return_dtype = None
        else:
            pl_return_dtype = py_type_to_dtype(return_dtype)
        return wrap_s(self._s.apply_lambda(func, pl_return_dtype, skip_nulls))

    def shift(self, periods: int = 1) -> Series:
        """
        Shift the values by a given period.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.shift(periods=1)
        shape: (3,)
        Series: 'a' [i64]
        [
                null
                1
                2
        ]
        >>> s.shift(periods=-1)
        shape: (3,)
        Series: 'a' [i64]
        [
                2
                3
                null
        ]

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).

        """

    def shift_and_fill(self, periods: int, fill_value: int | pli.Expr) -> Series:
        """
        Shift the values by a given period and fill the resulting null values.

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).
        fill_value
            Fill None values with the result of this expression.

        """

    def zip_with(self, mask: Series, other: Series) -> Series:
        """
        Take values from self or other based on the given mask.

        Where mask evaluates true, take values from self. Where mask evaluates false,
        take values from other.

        Parameters
        ----------
        mask
            Boolean Series.
        other
            Series of same type.

        Returns
        -------
        New Series

        Examples
        --------
        >>> s1 = pl.Series([1, 2, 3, 4, 5])
        >>> s2 = pl.Series([5, 4, 3, 2, 1])
        >>> s1.zip_with(s1 < s2, s2)
        shape: (5,)
        Series: '' [i64]
        [
                1
                2
                3
                2
                1
        ]
        >>> mask = pl.Series([True, False, True, False, True])
        >>> s1.zip_with(mask, s2)
        shape: (5,)
        Series: '' [i64]
        [
                1
                4
                3
                2
                5
        ]

        """
        return wrap_s(self._s.zip_with(mask._s, other._s))

    def rolling_min(
        self,
        window_size: int,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        center: bool = False,
    ) -> Series:
        """
        Apply a rolling min (moving min) over the values in this array.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window

        Examples
        --------
        >>> s = pl.Series("a", [100, 200, 300, 400, 500])
        >>> s.rolling_min(window_size=3)
        shape: (5,)
        Series: 'a' [i64]
        [
            null
            null
            100
            200
            300
        ]

        """
        return (
            self.to_frame()
            .select(
                pli.col(self.name).rolling_min(
                    window_size, weights, min_periods, center
                )
            )
            .to_series()
        )

    def rolling_max(
        self,
        window_size: int,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        center: bool = False,
    ) -> Series:
        """
        Apply a rolling max (moving max) over the values in this array.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window

        Examples
        --------
        >>> s = pl.Series("a", [100, 200, 300, 400, 500])
        >>> s.rolling_max(window_size=2)
        shape: (5,)
        Series: 'a' [i64]
        [
            null
            200
            300
            400
            500
        ]

        """
        return (
            self.to_frame()
            .select(
                pli.col(self.name).rolling_max(
                    window_size, weights, min_periods, center
                )
            )
            .to_series()
        )

    def rolling_mean(
        self,
        window_size: int,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        center: bool = False,
    ) -> Series:
        """
        Apply a rolling mean (moving mean) over the values in this array.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window

        Examples
        --------
        >>> s = pl.Series("a", [100, 200, 300, 400, 500])
        >>> s.rolling_mean(window_size=2)
        shape: (5,)
        Series: 'a' [f64]
        [
            null
            150.0
            250.0
            350.0
            450.0
        ]

        """
        return (
            self.to_frame()
            .select(
                pli.col(self.name).rolling_mean(
                    window_size, weights, min_periods, center
                )
            )
            .to_series()
        )

    def rolling_sum(
        self,
        window_size: int,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        center: bool = False,
    ) -> Series:
        """
        Apply a rolling sum (moving sum) over the values in this array.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length of the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4, 5])
        >>> s.rolling_sum(window_size=2)
        shape: (5,)
        Series: 'a' [i64]
        [
                null
                3
                5
                7
                9
        ]

        """
        return (
            self.to_frame()
            .select(
                pli.col(self.name).rolling_sum(
                    window_size, weights, min_periods, center
                )
            )
            .to_series()
        )

    def rolling_std(
        self,
        window_size: int,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        center: bool = False,
    ) -> Series:
        """
        Compute a rolling std dev.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
        >>> s.rolling_std(window_size=3)
        shape: (6,)
        Series: 'a' [f64]
        [
                null
                null
                1.0
                1.0
                1.527525
                2.0
        ]

        """
        return (
            self.to_frame()
            .select(
                pli.col(self.name).rolling_std(
                    window_size, weights, min_periods, center
                )
            )
            .to_series()
        )

    def rolling_var(
        self,
        window_size: int,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        center: bool = False,
    ) -> Series:
        """
        Compute a rolling variance.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
        >>> s.rolling_var(window_size=3)
        shape: (6,)
        Series: 'a' [f64]
        [
                null
                null
                1.0
                1.0
                2.333333
                4.0
        ]

        """
        return (
            self.to_frame()
            .select(
                pli.col(self.name).rolling_var(
                    window_size, weights, min_periods, center
                )
            )
            .to_series()
        )

    def rolling_apply(
        self,
        function: Callable[[pli.Series], Any],
        window_size: int,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        center: bool = False,
    ) -> pli.Series:
        """
        Apply a custom rolling window function.

        Prefer the specific rolling window functions over this one, as they are faster:

            * rolling_min
            * rolling_max
            * rolling_mean
            * rolling_sum

        Parameters
        ----------
        function
            Aggregation function
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window

        Examples
        --------
        >>> import numpy as np
        >>> s = pl.Series("A", [11.0, 2.0, 9.0, float("nan"), 8.0])
        >>> print(s.rolling_apply(function=np.nanstd, window_size=3))
        shape: (5,)
        Series: 'A' [f64]
        [
            null
            null
            3.858612
            3.5
            0.5
        ]

        """

    def rolling_median(
        self,
        window_size: int,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        center: bool = False,
    ) -> Series:
        """
        Compute a rolling median.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
        >>> s.rolling_median(window_size=3)
        shape: (6,)
        Series: 'a' [f64]
        [
                null
                null
                2.0
                3.0
                4.0
                6.0
        ]

        """
        if min_periods is None:
            min_periods = window_size

        return (
            self.to_frame()
            .select(
                pli.col(self.name).rolling_median(
                    window_size, weights, min_periods, center
                )
            )
            .to_series()
        )

    def rolling_quantile(
        self,
        quantile: float,
        interpolation: RollingInterpolationMethod = "nearest",
        window_size: int = 2,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        center: bool = False,
    ) -> Series:
        """
        Compute a rolling quantile.

        Parameters
        ----------
        quantile
            Quantile between 0.0 and 1.0.
        interpolation : {'nearest', 'higher', 'lower', 'midpoint', 'linear'}
            Interpolation method.
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
        >>> s.rolling_quantile(quantile=0.33, window_size=3)
        shape: (6,)
        Series: 'a' [f64]
        [
                null
                null
                1.0
                2.0
                3.0
                4.0
        ]
        >>> s.rolling_quantile(quantile=0.33, interpolation="linear", window_size=3)
        shape: (6,)
        Series: 'a' [f64]
        [
                null
                null
                1.66
                2.66
                3.66
                5.32
        ]

        """
        if min_periods is None:
            min_periods = window_size

        return (
            self.to_frame()
            .select(
                pli.col(self.name).rolling_quantile(
                    quantile, interpolation, window_size, weights, min_periods, center
                )
            )
            .to_series()
        )

    def rolling_skew(self, window_size: int, bias: bool = True) -> Series:
        """
        Compute a rolling skew.

        Parameters
        ----------
        window_size
            Integer size of the rolling window.
        bias
            If False, the calculations are corrected for statistical bias.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
        >>> s.rolling_skew(window_size=3)
        shape: (6,)
        Series: 'a' [f64]
        [
                null
                null
                0.0
                0.0
                0.381802
                0.0
        ]

        """

    def sample(
        self,
        n: int | None = None,
        frac: float | None = None,
        with_replacement: bool = False,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> Series:
        """
        Sample from this Series.

        Parameters
        ----------
        n
            Number of items to return. Cannot be used with `frac`. Defaults to 1 if
            `frac` is None.
        frac
            Fraction of items to return. Cannot be used with `n`.
        with_replacement
            Allow values to be sampled more than once.
        shuffle
            Shuffle the order of sampled data points.
        seed
            Seed for the random number generator. If set to None (default), a random
            seed is generated using the ``random`` module.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4, 5])
        >>> s.sample(2, seed=0)  # doctest: +IGNORE_RESULT
        shape: (2,)
        Series: 'a' [i64]
        [
            1
            5
        ]

        """

    def peak_max(self) -> Series:
        """
        Get a boolean mask of the local maximum peaks.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4, 5])
        >>> s.peak_max()
        shape: (5,)
        Series: '' [bool]
        [
                false
                false
                false
                false
                true
        ]

        """
        return wrap_s(self._s.peak_max())

    def peak_min(self) -> Series:
        """
        Get a boolean mask of the local minimum peaks.

        Examples
        --------
        >>> s = pl.Series("a", [4, 1, 3, 2, 5])
        >>> s.peak_min()
        shape: (5,)
        Series: '' [bool]
        [
            false
            true
            false
            true
            false
        ]

        """
        return wrap_s(self._s.peak_min())

    def n_unique(self) -> int:
        """
        Count the number of unique values in this Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 2, 3])
        >>> s.n_unique()
        3

        """
        return self._s.n_unique()

    def shrink_to_fit(self, in_place: bool = False) -> Series:
        """
        Shrink Series memory usage.

        Shrinks the underlying array capacity to exactly fit the actual data.
        (Note that this function does not change the Series data type).

        """
        if in_place:
            self._s.shrink_to_fit()
            return self
        else:
            series = self.clone()
            series._s.shrink_to_fit()
            return series

    def hash(
        self,
        seed: int = 0,
        seed_1: int | None = None,
        seed_2: int | None = None,
        seed_3: int | None = None,
    ) -> pli.Series:
        """
        Hash the Series.

        The hash value is of type `UInt64`.

        Parameters
        ----------
        seed
            Random seed parameter. Defaults to 0.
        seed_1
            Random seed parameter. Defaults to `seed` if not set.
        seed_2
            Random seed parameter. Defaults to `seed` if not set.
        seed_3
            Random seed parameter. Defaults to `seed` if not set.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.hash(seed=42)
        shape: (3,)
        Series: 'a' [u64]
        [
            10734580197236529959
            3022416320763508302
            13756996518000038261
        ]

        """

    def reinterpret(self, signed: bool = True) -> Series:
        """
        Reinterpret the underlying bits as a signed/unsigned integer.

        This operation is only allowed for 64bit integers. For lower bits integers,
        you can safely use that cast operation.

        Parameters
        ----------
        signed
            If True, reinterpret as `pl.Int64`. Otherwise, reinterpret as `pl.UInt64`.

        """

    def interpolate(self, method: InterpolationMethod = "linear") -> Series:
        """
        Interpolate intermediate values. The interpolation method is linear.

        Parameters
        ----------
        method : {'linear', 'linear'}
            Interpolation method

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, None, None, 5])
        >>> s.interpolate()
        shape: (5,)
        Series: 'a' [i64]
        [
            1
            2
            3
            4
            5
        ]

        """

    def abs(self) -> Series:
        """
        Compute absolute values.

        Same as `abs(series)`.
        """

    def rank(self, method: RankMethod = "average", reverse: bool = False) -> Series:
        """
        Assign ranks to data, dealing with ties appropriately.

        Parameters
        ----------
        method : {'average', 'min', 'max', 'dense', 'ordinal', 'random'}
            The method used to assign ranks to tied elements.
            The following methods are available (default is 'average'):

            - 'average' : The average of the ranks that would have been assigned to
              all the tied values is assigned to each value.
            - 'min' : The minimum of the ranks that would have been assigned to all
              the tied values is assigned to each value. (This is also referred to
              as "competition" ranking.)
            - 'max' : The maximum of the ranks that would have been assigned to all
              the tied values is assigned to each value.
            - 'dense' : Like 'min', but the rank of the next highest element is
              assigned the rank immediately after those assigned to the tied
              elements.
            - 'ordinal' : All values are given a distinct rank, corresponding to
              the order that the values occur in the Series.
            - 'random' : Like 'ordinal', but the rank for ties is not dependent
              on the order that the values occur in the Series.
        reverse
            Reverse the operation.

        Examples
        --------
        The 'average' method:

        >>> s = pl.Series("a", [3, 6, 1, 1, 6])
        >>> s.rank()
        shape: (5,)
        Series: 'a' [f32]
        [
            3.0
            4.5
            1.5
            1.5
            4.5
        ]

        The 'ordinal' method:

        >>> s = pl.Series("a", [3, 6, 1, 1, 6])
        >>> s.rank("ordinal")
        shape: (5,)
        Series: 'a' [u32]
        [
            3
            4
            1
            2
            5
        ]

        """

    def diff(self, n: int = 1, null_behavior: NullBehavior = "ignore") -> Series:
        """
        Calculate the n-th discrete difference.

        Parameters
        ----------
        n
            Number of slots to shift.
        null_behavior : {'ignore', 'drop'}
            How to handle null values.

        """

    def pct_change(self, n: int = 1) -> Series:
        """
        Computes percentage change between values.

        Percentage change (as fraction) between current element and most-recent
        non-null element at least ``n`` period(s) before the current element.

        Computes the change from the previous row by default.

        Parameters
        ----------
        n
            periods to shift for forming percent change.

        Examples
        --------
        >>> pl.Series(range(10)).pct_change()
        shape: (10,)
        Series: '' [f64]
        [
            null
            inf
            1.0
            0.5
            0.333333
            0.25
            0.2
            0.166667
            0.142857
            0.125
        ]

        >>> pl.Series([1, 2, 4, 8, 16, 32, 64, 128, 256, 512]).pct_change(2)
        shape: (10,)
        Series: '' [f64]
        [
            null
            null
            3.0
            3.0
            3.0
            3.0
            3.0
            3.0
            3.0
            3.0
        ]

        """

    def skew(self, bias: bool = True) -> float | None:
        r"""
        Compute the sample skewness of a data set.

        For normally distributed data, the skewness should be about zero. For
        unimodal continuous distributions, a skewness value greater than zero means
        that there is more weight in the right tail of the distribution. The
        function `skewtest` can be used to determine if the skewness value
        is close enough to zero, statistically speaking.


        See scipy.stats for more information.

        Parameters
        ----------
        bias : bool, optional
            If False, the calculations are corrected for statistical bias.

        Notes
        -----
        The sample skewness is computed as the Fisher-Pearson coefficient
        of skewness, i.e.

        .. math:: g_1=\frac{m_3}{m_2^{3/2}}

        where

        .. math:: m_i=\frac{1}{N}\sum_{n=1}^N(x[n]-\bar{x})^i

        is the biased sample :math:`i\texttt{th}` central moment, and
        :math:`\bar{x}` is
        the sample mean.  If ``bias`` is False, the calculations are
        corrected for bias and the value computed is the adjusted
        Fisher-Pearson standardized moment coefficient, i.e.

        .. math::
            G_1 = \frac{k_3}{k_2^{3/2}} = \frac{\sqrt{N(N-1)}}{N-2}\frac{m_3}{m_2^{3/2}}

        """
        return self._s.skew(bias)

    def kurtosis(self, fisher: bool = True, bias: bool = True) -> float | None:
        """
        Compute the kurtosis (Fisher or Pearson) of a dataset.

        Kurtosis is the fourth central moment divided by the square of the
        variance. If Fisher's definition is used, then 3.0 is subtracted from
        the result to give 0.0 for a normal distribution.
        If bias is False then the kurtosis is calculated using k statistics to
        eliminate bias coming from biased moment estimators

        See scipy.stats for more information

        Parameters
        ----------
        fisher : bool, optional
            If True, Fisher's definition is used (normal ==> 0.0). If False,
            Pearson's definition is used (normal ==> 3.0).
        bias : bool, optional
            If False, the calculations are corrected for statistical bias.

        """
        return self._s.kurtosis(fisher, bias)

    def clip(self, min_val: int | float, max_val: int | float) -> Series:
        """
        Clip (limit) the values in an array to a `min` and `max` boundary.

        Only works for numerical types.

        If you want to clip other dtypes, consider writing a "when, then, otherwise"
        expression. See :func:`when` for more information.

        Parameters
        ----------
        min_val
            Minimum value.
        max_val
            Maximum value.

        Examples
        --------
        >>> s = pl.Series("foo", [-50, 5, None, 50])
        >>> s.clip(1, 10)
        shape: (4,)
        Series: 'foo' [i64]
        [
            1
            5
            null
            10
        ]

        """

    def clip_min(self, min_val: int | float) -> Series:
        """
        Clip (limit) the values in an array to a `min` boundary.

        Only works for numerical types.

        If you want to clip other dtypes, consider writing a "when, then, otherwise"
        expression. See :func:`when` for more information.

        Parameters
        ----------
        min_val
            Minimum value.

        """

    def clip_max(self, max_val: int | float) -> Series:
        """
        Clip (limit) the values in an array to a `max` boundary.

        Only works for numerical types.

        If you want to clip other dtypes, consider writing a "when, then, otherwise"
        expression. See :func:`when` for more information.

        Parameters
        ----------
        max_val
            Maximum value.

        """

    def reshape(self, dims: tuple[int, ...]) -> Series:
        """
        Reshape this Series to a flat Series or a Series of Lists.

        Parameters
        ----------
        dims
            Tuple of the dimension sizes. If a -1 is used in any of the dimensions, that
            dimension is inferred.

        Returns
        -------
        Series
            If a single dimension is given, results in a flat Series of shape (len,).
            If a multiple dimensions are given, results in a Series of Lists with shape
            (rows, cols).

        """

    def shuffle(self, seed: int | None = None) -> Series:
        """
        Shuffle the contents of this Series.

        Parameters
        ----------
        seed
            Seed for the random number generator.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.shuffle(seed=1)
        shape: (3,)
        Series: 'a' [i64]
        [
                2
                1
                3
        ]

        """
        if seed is None:
            warn(
                "Series.shuffle will default to a random seed in a future version."
                " Provide a value for the seed argument to silence this warning.",
                DeprecationWarning,
                stacklevel=2,
            )
            seed = 0
        return self.to_frame().select(pli.col(self.name).shuffle(seed)).to_series()

    def ewm_mean(
        self,
        com: float | None = None,
        span: float | None = None,
        half_life: float | None = None,
        alpha: float | None = None,
        adjust: bool = True,
        min_periods: int = 1,
    ) -> Series:
        r"""
        Exponentially-weighted moving average.

        Parameters
        ----------
        com
            Specify decay in terms of center of mass, :math:`\gamma`, with

                .. math::
                    \alpha = \frac{1}{1 + \gamma} \; \forall \; \gamma \geq 0
        span
            Specify decay in terms of span, :math:`\theta`, with

                .. math::
                    \alpha = \frac{2}{\theta + 1} \; \forall \; \theta \geq 1
        half_life
            Specify decay in terms of half-life, :math:`\lambda`, with

                .. math::
                    \alpha = 1 - \exp \left\{ \frac{ -\ln(2) }{ \lambda } \right\} \;
                    \forall \; \lambda > 0
        alpha
            Specify smoothing factor alpha directly, :math:`0 < \alpha \leq 1`.
        adjust
            Divide by decaying adjustment factor in beginning periods to account for
            imbalance in relative weightings

                - When ``adjust=True`` the EW function is calculated
                  using weights :math:`w_i = (1 - \alpha)^i`
                - When ``adjust=False`` the EW function is calculated
                  recursively by

                  .. math::
                    y_0 &= x_0 \\
                    y_t &= (1 - \alpha)y_{t - 1} + \alpha x_t
        min_periods
            Minimum number of observations in window required to have a value
            (otherwise result is null).

        """

    def ewm_std(
        self,
        com: float | None = None,
        span: float | None = None,
        half_life: float | None = None,
        alpha: float | None = None,
        adjust: bool = True,
        bias: bool = False,
        min_periods: int = 1,
    ) -> Series:
        r"""
        Exponentially-weighted moving standard deviation.

        Parameters
        ----------
        com
            Specify decay in terms of center of mass, :math:`\gamma`, with

                .. math::
                    \alpha = \frac{1}{1 + \gamma} \; \forall \; \gamma \geq 0
        span
            Specify decay in terms of span, :math:`\theta`, with

                .. math::
                    \alpha = \frac{2}{\theta + 1} \; \forall \; \theta \geq 1
        half_life
            Specify decay in terms of half-life, :math:`\lambda`, with

                .. math::
                    \alpha = 1 - \exp \left\{ \frac{ -\ln(2) }{ \lambda } \right\} \;
                    \forall \; \lambda > 0
        alpha
            Specify smoothing factor alpha directly, :math:`0 < \alpha \leq 1`.
        adjust
            Divide by decaying adjustment factor in beginning periods to account for
            imbalance in relative weightings

                - When ``adjust=True`` the EW function is calculated
                  using weights :math:`w_i = (1 - \alpha)^i`
                - When ``adjust=False`` the EW function is calculated
                  recursively by

                  .. math::
                    y_0 &= x_0 \\
                    y_t &= (1 - \alpha)y_{t - 1} + \alpha x_t
        bias
            When ``bias=False``, apply a correction to make the estimate statistically
            unbiased.
        min_periods
            Minimum number of observations in window required to have a value
            (otherwise result is null).

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.ewm_std(com=1)
        shape: (3,)
        Series: 'a' [f64]
        [
            0.0
            0.707107
            0.963624
        ]

        """

    def ewm_var(
        self,
        com: float | None = None,
        span: float | None = None,
        half_life: float | None = None,
        alpha: float | None = None,
        adjust: bool = True,
        bias: bool = False,
        min_periods: int = 1,
    ) -> Series:
        r"""
        Exponentially-weighted moving variance.

        Parameters
        ----------
        com
            Specify decay in terms of center of mass, :math:`\gamma`, with

                .. math::
                    \alpha = \frac{1}{1 + \gamma} \; \forall \; \gamma \geq 0
        span
            Specify decay in terms of span, :math:`\theta`, with

                .. math::
                    \alpha = \frac{2}{\theta + 1} \; \forall \; \theta \geq 1
        half_life
            Specify decay in terms of half-life, :math:`\lambda`, with

                .. math::
                    \alpha = 1 - \exp \left\{ \frac{ -\ln(2) }{ \lambda } \right\} \;
                    \forall \; \lambda > 0
        alpha
            Specify smoothing factor alpha directly, :math:`0 < \alpha \leq 1`.
        adjust
            Divide by decaying adjustment factor in beginning periods to account for
            imbalance in relative weightings

                - When ``adjust=True`` the EW function is calculated
                  using weights :math:`w_i = (1 - \alpha)^i`
                - When ``adjust=False`` the EW function is calculated
                  recursively by

                  .. math::
                    y_0 &= x_0 \\
                    y_t &= (1 - \alpha)y_{t - 1} + \alpha x_t
        bias
            When ``bias=False``, apply a correction to make the estimate statistically
            unbiased.
        min_periods
            Minimum number of observations in window required to have a value
            (otherwise result is null).

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.ewm_var(com=1)
        shape: (3,)
        Series: 'a' [f64]
        [
            0.0
            0.5
            0.928571
        ]

        """

    def extend_constant(self, value: int | float | str | bool | None, n: int) -> Series:
        """
        Extend the Series with given number of values.

        Parameters
        ----------
        value
            The value to extend the Series with. This value may be None to fill with
            nulls.
        n
            The number of values to extend.

        Examples
        --------
        >>> s = pl.Series([1, 2, 3])
        >>> s.extend_constant(99, n=2)
        shape: (5,)
        Series: '' [i64]
        [
                1
                2
                3
                99
                99
        ]

        """

    def set_sorted(self, reverse: bool = False) -> Series:
        """
        Flags the Series as 'sorted'.

        Enables downstream code to user fast paths for sorted arrays.

        Parameters
        ----------
        reverse
            If the `Series` order is reversed, e.g. descending.

        Warnings
        --------
        This can lead to incorrect results if this `Series` is not sorted!!
        Use with care!

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.set_sorted().max()
        3

        """
        return wrap_s(self._s.set_sorted(reverse))

    def new_from_index(self, index: int, length: int) -> pli.Series:
        """Create a new Series filled with values from the given index."""
        return wrap_s(self._s.new_from_index(index, length))

    def shrink_dtype(self) -> Series:
        """
        Shrink numeric columns to the minimal required datatype.

        Shrink to the dtype needed to fit the extrema of this [`Series`].
        This can be used to reduce memory pressure.
        """

    def get_chunks(self) -> list[Series]:
        """Get the chunks of this Series as a list of Series."""
        return self._s.get_chunks()

    # Below are the namespaces defined. Do not move these up in the definition of
    # Series, as it confuses mypy between the type annotation `str` and the
    # namespace `str`
    @accessor
    def arr(self) -> ListNameSpace:
        """Create an object namespace of all list related methods."""
        return ListNameSpace(self)

    @accessor
    def cat(self) -> CatNameSpace:
        """Create an object namespace of all categorical related methods."""
        return CatNameSpace(self)

    @accessor
    def dt(self) -> DateTimeNameSpace:
        """Create an object namespace of all datetime related methods."""
        return DateTimeNameSpace(self)

    @accessor
    def str(self) -> StringNameSpace:
        """Create an object namespace of all string related methods."""
        return StringNameSpace(self)

    @accessor
    def struct(self) -> StructNameSpace:
        """Create an object namespace of all struct related methods."""
        return StructNameSpace(self)


class SeriesIter:
    """Utility class that allows slow iteration over a `Series`."""

    def __init__(self, length: int, s: Series):
        self.len = length
        self.i = 0
        self.s = s

    def __iter__(self) -> SeriesIter:
        return self

    def __next__(self) -> Any:
        if self.i < self.len:
            i = self.i
            self.i += 1
            return self.s[i]
        else:
            raise StopIteration


def _resolve_datetime_dtype(
    dtype: PolarsDataType | None, ndtype: np.datetime64
) -> PolarsDataType | None:
    """Given polars/numpy datetime dtypes, resolve to an explicit unit."""
    if dtype is None or (dtype == Datetime and not getattr(dtype, "tu", None)):
        tu = getattr(dtype, "tu", None) or np.datetime_data(ndtype)[0]
        # explicit formulation is verbose, but keeps mypy happy
        # (and avoids unsupported timeunits such as "s")
        if tu == "ns":
            dtype = Datetime("ns")
        elif tu == "us":
            dtype = Datetime("us")
        elif tu == "ms":
            dtype = Datetime("ms")
    return dtype
