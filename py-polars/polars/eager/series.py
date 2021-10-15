import typing as tp
from datetime import date, datetime
from numbers import Number
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, Union

import numpy as np

try:
    import pyarrow as pa

    _PYARROW_AVAILABLE = True
except ImportError:
    _PYARROW_AVAILABLE = False

import polars as pl
from polars.internals.construction import (
    arrow_to_pyseries,
    numpy_to_pyseries,
    pandas_to_pyseries,
    sequence_to_pyseries,
    series_to_pyseries,
)

try:
    from polars.polars import PyDataFrame, PySeries

    _DOCUMENTING = False
except ImportError:
    _DOCUMENTING = True

from ..datatypes import (
    DTYPE_TO_FFINAME,
    DTYPES,
    Boolean,
    DataType,
    Date,
    Datetime,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    List,
    Object,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Utf8,
    date_like_to_physical,
    dtype_to_ctype,
)
from ..utils import _ptr_to_numpy

try:
    import pandas as pd

    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

__all__ = [
    "Series",
]


def match_dtype(value: Any, dtype: "Type[DataType]") -> Any:
    """
    In right hand side operation, make sure that the operand is coerced to the Series dtype
    """
    if dtype == Float32 or dtype == Float64:
        return float(value)
    else:
        return int(value)


def get_ffi_func(
    name: str,
    dtype: Type["DataType"],
    obj: Optional["Series"] = None,
    default: Optional[Callable[[Any], Any]] = None,
) -> Callable[..., Any]:
    """
    Dynamically obtain the proper ffi function/ method.

    Parameters
    ----------
    name
        function or method name where dtype is replaced by <>
        for example
            "call_foo_<>"
    dtype
        polars dtype.
    obj
        Optional object to find the method for. If none provided globals are used.
    default
        default function to use if not found.

    Returns
    -------
    ffi function
    """
    ffi_name = DTYPE_TO_FFINAME[dtype]
    fname = name.replace("<>", ffi_name)
    if obj:
        return getattr(obj, fname, default)
    else:
        return globals().get(fname, default)


def wrap_s(s: "PySeries") -> "Series":
    return Series._from_pyseries(s)


ArrayLike = Union[
    Sequence[Any], "Series", "pa.Array", np.ndarray, "pd.Series", "pd.DatetimeIndex"
]


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
        Throw error on numeric overflow
    nan_to_null
        In case a numpy arrow is used to create this Series, indicate how to deal with np.nan

    Examples
    --------
    Constructing a Series by specifying name and values positionally:

    >>> s = pl.Series('a', [1, 2, 3])
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
    <class 'polars.datatypes.Int64'>

    Constructing a Series with a specific dtype:

    >>> s2 = pl.Series('a', [1, 2, 3], dtype=pl.Float32)
    >>> s2
    shape: (3,)
    Series: 'a' [f32]
    [
            1
            2
            3
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

    def __init__(
        self,
        name: Optional[Union[str, ArrayLike]] = None,
        values: Optional[ArrayLike] = None,
        dtype: Optional[Type[DataType]] = None,
        strict: bool = True,
        nan_to_null: bool = False,
    ):

        # Handle case where values are passed as the first argument
        if name is not None and not isinstance(name, str):
            if values is None:
                values = name
                name = None
            else:
                raise ValueError("Series name must be a string.")

        # TODO: Remove if-statement below once Series name is allowed to be None
        if name is None:
            name = ""

        if values is None:
            self._s = sequence_to_pyseries(name, [], dtype=dtype)
        elif isinstance(values, Series):
            self._s = series_to_pyseries(name, values)
        elif _PYARROW_AVAILABLE and isinstance(values, pa.Array):
            self._s = arrow_to_pyseries(name, values)
        elif isinstance(values, np.ndarray):
            self._s = numpy_to_pyseries(name, values, strict, nan_to_null)
        elif isinstance(values, Sequence):
            self._s = sequence_to_pyseries(name, values, dtype=dtype, strict=strict)
        elif _PANDAS_AVAILABLE and isinstance(values, (pd.Series, pd.DatetimeIndex)):
            self._s = pandas_to_pyseries(name, values)
        else:
            raise ValueError("Series constructor not called properly.")

    @classmethod
    def _from_pyseries(cls, pyseries: "PySeries") -> "Series":
        series = cls.__new__(cls)
        series._s = pyseries
        return series

    @classmethod
    def _repeat(
        cls, name: str, val: Union[int, float, str, bool], n: int, dtype: Type[DataType]
    ) -> "Series":
        return cls._from_pyseries(PySeries.repeat(name, val, n, dtype))

    @classmethod
    def _from_arrow(cls, name: str, values: "pa.Array") -> "Series":
        """
        Construct a Series from an Arrow Array.
        """
        return cls._from_pyseries(arrow_to_pyseries(name, values))

    @classmethod
    def _from_pandas(
        cls,
        name: str,
        values: Union["pd.Series", "pd.DatetimeIndex"],
        nan_to_none: bool = True,
    ) -> "Series":
        """
        Construct a Series from a pandas Series or DatetimeIndex.
        """
        return cls._from_pyseries(
            pandas_to_pyseries(name, values, nan_to_none=nan_to_none)
        )

    def inner(self) -> "PySeries":
        return self._s

    def __getstate__(self):  # type: ignore
        return self._s.__getstate__()

    def __setstate__(self, state):  # type: ignore
        self._s = sequence_to_pyseries("", [], Float32)
        self._s.__setstate__(state)

    def __str__(self) -> str:
        return self._s.as_str()

    def __repr__(self) -> str:
        return self.__str__()

    def __and__(self, other: "Series") -> "Series":
        if not isinstance(other, Series):
            other = Series([other])
        return wrap_s(self._s.bitand(other._s))

    def __rand__(self, other: "Series") -> "Series":
        return self.__and__(other)

    def __or__(self, other: "Series") -> "Series":
        if not isinstance(other, Series):
            other = Series([other])
        return wrap_s(self._s.bitor(other._s))

    def __ror__(self, other: "Series") -> "Series":
        return self.__or__(other)

    def __xor__(self, other: "Series") -> "Series":
        if not isinstance(other, Series):
            other = Series([other])
        return wrap_s(self._s.bitxor(other._s))

    def __rxor__(self, other: "Series") -> "Series":
        return self.__xor__(other)

    def __eq__(self, other: Any) -> "Series":  # type: ignore[override]
        if isinstance(other, Sequence) and not isinstance(other, str):
            other = Series("", other)
        if isinstance(other, Series):
            return Series._from_pyseries(self._s.eq(other._s))
        f = get_ffi_func("eq_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __ne__(self, other: Any) -> "Series":  # type: ignore[override]
        if isinstance(other, Sequence) and not isinstance(other, str):
            other = Series("", other)
        if isinstance(other, Series):
            return Series._from_pyseries(self._s.neq(other._s))
        f = get_ffi_func("neq_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __gt__(self, other: Any) -> "Series":
        if isinstance(other, Sequence) and not isinstance(other, str):
            other = Series("", other)
        if isinstance(other, Series):
            return Series._from_pyseries(self._s.gt(other._s))
        f = get_ffi_func("gt_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __lt__(self, other: Any) -> "Series":
        if isinstance(other, Sequence) and not isinstance(other, str):
            other = Series("", other)
        if isinstance(other, Series):
            return Series._from_pyseries(self._s.lt(other._s))
        f = get_ffi_func("lt_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __ge__(self, other: Any) -> "Series":
        if isinstance(other, Sequence) and not isinstance(other, str):
            other = Series("", other)
        if isinstance(other, Series):
            return Series._from_pyseries(self._s.gt_eq(other._s))
        f = get_ffi_func("gt_eq_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __le__(self, other: Any) -> "Series":
        if isinstance(other, Sequence) and not isinstance(other, str):
            other = Series("", other)
        if isinstance(other, Series):
            return Series._from_pyseries(self._s.lt_eq(other._s))
        f = get_ffi_func("lt_eq_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __add__(self, other: Any) -> "Series":
        if isinstance(other, str):
            other = Series("", [other])
        if isinstance(other, Series):
            return wrap_s(self._s.add(other._s))
        dtype = date_like_to_physical(self.dtype)
        f = get_ffi_func("add_<>", dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __sub__(self, other: Any) -> "Series":
        if isinstance(other, Series):
            return Series._from_pyseries(self._s.sub(other._s))
        dtype = date_like_to_physical(self.dtype)
        f = get_ffi_func("sub_<>", dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __truediv__(self, other: Any) -> "Series":
        physical_type = date_like_to_physical(self.dtype)
        if self.dtype != physical_type or self.is_float():
            return self.__floordiv__(other)
        return self.cast(pl.Float64) / other

    def __floordiv__(self, other: Any) -> "Series":
        if isinstance(other, Series):
            return Series._from_pyseries(self._s.div(other._s))
        dtype = date_like_to_physical(self.dtype)
        f = get_ffi_func("div_<>", dtype, self._s)
        return wrap_s(f(other))

    def __mul__(self, other: Any) -> "Series":
        if isinstance(other, Series):
            return Series._from_pyseries(self._s.mul(other._s))
        dtype = date_like_to_physical(self.dtype)
        f = get_ffi_func("mul_<>", dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __mod__(self, other: Any) -> "Series":
        if isinstance(other, Series):
            return Series._from_pyseries(self._s.rem(other._s))
        dtype = date_like_to_physical(self.dtype)
        f = get_ffi_func("rem_<>", dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __rmod__(self, other: Any) -> "Series":
        if isinstance(other, Series):
            return Series._from_pyseries(other._s.rem(self._s))
        dtype = date_like_to_physical(self.dtype)
        other = match_dtype(other, dtype)
        f = get_ffi_func("rem_<>_rhs", dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __radd__(self, other: Any) -> "Series":
        if isinstance(other, Series):
            return Series._from_pyseries(self._s.add(other._s))
        dtype = date_like_to_physical(self.dtype)
        other = match_dtype(other, dtype)
        f = get_ffi_func("add_<>_rhs", dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __rsub__(self, other: Any) -> "Series":
        if isinstance(other, Series):
            return Series._from_pyseries(other._s.sub(self._s))
        dtype = date_like_to_physical(self.dtype)
        other = match_dtype(other, dtype)
        f = get_ffi_func("sub_<>_rhs", dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __invert__(self) -> "Series":
        if self.dtype == Boolean:
            return wrap_s(self._s._not())
        return NotImplemented

    def __rtruediv__(self, other: Any) -> np.ndarray:
        primitive = date_like_to_physical(self.dtype)
        if self.dtype != primitive or self.is_float():
            self.__rfloordiv__(other)

        if isinstance(other, int):
            other = float(other)

        return self.cast(pl.Float64).__rfloordiv__(other)  # type: ignore

    def __rfloordiv__(self, other: Any) -> "Series":
        if isinstance(other, Series):
            return Series._from_pyseries(other._s.div(self._s))
        dtype = date_like_to_physical(self.dtype)
        other = match_dtype(other, dtype)
        f = get_ffi_func("div_<>_rhs", dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __rmul__(self, other: Any) -> "Series":
        if isinstance(other, Series):
            return Series._from_pyseries(self._s.mul(other._s))
        dtype = date_like_to_physical(self.dtype)
        other = match_dtype(other, dtype)
        f = get_ffi_func("mul_<>", dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __pow__(self, power: float, modulo: None = None) -> "Series":
        return np.power(self, power)  # type: ignore

    def __neg__(self) -> "Series":
        return 0 - self

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, int):
            if item < 0:
                item = self.len() + item
            if self.dtype in (List, Date, Datetime, Object):
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
        # assume it is boolean mask
        if isinstance(item, Series):
            return Series._from_pyseries(self._s.filter(item._s))

        if isinstance(item, range):
            step: Optional[int]
            # maybe we can slice instead of take by indices
            if item.step != 1:
                step = item.step
            else:
                step = None
            slc = slice(item.start, item.stop, step)
            return self[slc]

        # slice
        if type(item) == slice:
            start, stop, stride = item.indices(self.len())
            out = self.slice(start, stop - start)
            if stride != 1:
                return out.take_every(stride)
            else:
                return out
        f = get_ffi_func("get_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        out = f(item)
        if self.dtype == List:
            return wrap_s(out)
        return out

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(value, list):
            raise ValueError("cannot set with a list as value, use a primitive value")
        if isinstance(key, Series):
            if key.dtype == Boolean:
                self._s = self.set(key, value)._s
            elif key.dtype == UInt64:
                self._s = self.set_at_idx(key, value)._s
            elif key.dtype == UInt32:
                self._s = self.set_at_idx(key.cast(UInt64), value)._s
        # TODO: implement for these types without casting to series
        elif isinstance(key, (np.ndarray, list, tuple)):
            s = wrap_s(PySeries.new_u64("", np.array(key, np.uint64)))
            self.__setitem__(s, value)
        elif isinstance(key, int):
            self.__setitem__([key], value)
        else:
            raise ValueError(f'cannot use "{key}" for indexing')

    def sqrt(self) -> "Series":
        """
        Compute the square root of the elements

        Syntactic sugar for

        >>> pl.Series([1, 2]) ** 0.5
        """
        return self ** 0.5

    def drop_nulls(self) -> "Series":
        """
        Create a new Series that copies data from this Series without null values.
        """
        return wrap_s(self._s.drop_nulls())

    def to_frame(self) -> "pl.DataFrame":
        """
        Cast this Series to a DataFrame.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> df = s.to_frame()
        >>> df
        shape: (3, 1)
        ╭─────╮
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 1   │
        ├╌╌╌╌╌┤
        │ 2   │
        ├╌╌╌╌╌┤
        │ 3   │
        ╰─────╯

        >>> type(df)
        <class 'polars.eager.frame.DataFrame'>

        """
        return pl.eager.frame.wrap_df(PyDataFrame([self._s]))

    @property
    def dtype(self) -> Type[DataType]:
        """
        Get the data type of this Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.dtype
        <class 'polars.datatypes.Int64'>

        """
        return DTYPES[self._s.dtype()]

    def describe(self) -> "pl.DataFrame":
        """
        Quick summary statistics of a series. Series with mixed datatypes will return summary statistics for the datatype of the first value.

        Returns
        -------
        Dictionary with summary statistics of a Series.

        Examples
        --------
        >>> series_num = pl.Series([1, 2, 3, 4, 5])
        >>> series_num.describe()
        shape: (6, 2)
        ┌──────────────┬────────────────────┐
        │ statistic    ┆ value              │
        │ ---          ┆ ---                │
        │ str          ┆ f64                │
        ╞══════════════╪════════════════════╡
        │ "min"        ┆ 1                  │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ "max"        ┆ 5                  │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ "null_count" ┆ 0.0                │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ "mean"       ┆ 3                  │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ "std"        ┆ 1.5811388300841898 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ "count"      ┆ 5                  │
        └──────────────┴────────────────────┘

        >>> series_str = pl.Series(["a", "a", None, "b", "c"])
        >>> series_str.describe()
        shape: (3, 2)
        ┌──────────────┬───────┐
        │ statistic    ┆ value │
        │ ---          ┆ ---   │
        │ str          ┆ i64   │
        ╞══════════════╪═══════╡
        │ "unique"     ┆ 4     │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ "null_count" ┆ 1     │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ "count"      ┆ 5     │
        └──────────────┴───────┘

        """
        stats: Dict[str, Union[Optional[float], int, str]]

        if self.len() == 0:
            raise ValueError("Series must contain at least one value")
        elif self.is_numeric():
            self = self.cast(pl.Float64)
            stats = {
                "min": self.min(),
                "max": self.max(),
                "null_count": self.null_count(),
                "mean": self.mean(),
                "std": self.std(),
                "count": self.len(),
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
        elif self.is_datetime():
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

        return pl.DataFrame(
            {"statistic": list(stats.keys()), "value": list(stats.values())}
        )

    def sum(self) -> Union[int, float]:
        """
        Reduce this Series to the sum value.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.sum()
        6

        """
        return self._s.sum()

    def mean(self) -> Union[int, float]:
        """
        Reduce this Series to the mean value.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.mean()
        2.0

        """
        return self._s.mean()

    def min(self) -> Union[int, float]:
        """
        Get the minimal value in this Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.min()
        1

        """
        return self._s.min()

    def max(self) -> Union[int, float]:
        """
        Get the maximum value in this Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.min()
        3

        """
        return self._s.max()

    def std(self, ddof: int = 1) -> Optional[float]:
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
        return np.std(self.drop_nulls().view(), ddof=ddof)

    def var(self, ddof: int = 1) -> Optional[float]:
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
        return np.var(self.drop_nulls().view(), ddof=ddof)

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

    def quantile(self, quantile: float) -> float:
        """
        Get the quantile value of this Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.quantile(0.5)
        2

        """
        return self._s.quantile(quantile)

    def to_dummies(self) -> "pl.DataFrame":
        """
        Get dummy variables.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.to_dummies()
        shape: (3, 3)
        ╭─────┬─────┬─────╮
        │ a_1 ┆ a_2 ┆ a_3 │
        │ --- ┆ --- ┆ --- │
        │ u8  ┆ u8  ┆ u8  │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 0   ┆ 0   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 0   ┆ 1   ┆ 0   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 0   ┆ 0   ┆ 1   │
        ╰─────┴─────┴─────╯

        """
        return pl.eager.frame.wrap_df(self._s.to_dummies())

    def value_counts(self) -> "pl.DataFrame":
        """
        Count the unique values in a Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 2, 3])
        >>> s.value_counts()
        shape: (3, 2)
        ╭─────┬────────╮
        │ a   ┆ counts │
        │ --- ┆ ---    │
        │ i64 ┆ u32    │
        ╞═════╪════════╡
        │ 2   ┆ 2      │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
        │ 1   ┆ 1      │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
        │ 3   ┆ 1      │
        ╰─────┴────────╯

        """
        return pl.eager.frame.wrap_df(self._s.value_counts())

    @property
    def name(self) -> str:
        """
        Get the name of this Series.
        """
        return self._s.name()

    def rename(self, name: str, in_place: bool = False) -> Optional["Series"]:
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
        >>> s.rename('b')
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
            return None
        else:
            s = self.clone()
            s._s.rename(name)
            return s

    def chunk_lengths(self) -> tp.List[int]:
        """
        Get the length of each individual chunk.
        """
        return self._s.chunk_lengths()

    def n_chunks(self) -> int:
        """
        Get the number of chunks that this Series contains.
        """
        return self._s.n_chunks()

    def cumsum(self, reverse: bool = False) -> "Series":
        """
        Get an array with the cumulative sum computed at every element.

        Parameters
        ----------
        reverse
            reverse the operation.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.cumsum()
        shape: (3,)
        Series: 'b' [i64]
        [
                1
                3
                6
        ]

        """
        return wrap_s(self._s.cumsum(reverse))

    def cummin(self, reverse: bool = False) -> "Series":
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
        Series: 'b' [i64]
        [
                1
                1
                1
        ]

        """
        return wrap_s(self._s.cummin(reverse))

    def cummax(self, reverse: bool = False) -> "Series":
        """
        Get an array with the cumulative max computed at every element.

        Parameters
        ----------
        reverse
            reverse the operation.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.cummax()
        shape: (3,)
        Series: 'b' [i64]
        [
                1
                2
                3
        ]

        """
        return wrap_s(self._s.cummax(reverse))

    def limit(self, num_elements: int = 10) -> "Series":
        """
        Take n elements from this Series.

        Parameters
        ----------
        num_elements
            Amount of elements to take.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.limit(2)
        shape: (2,)
        Series: 'a' [i64]
        [
                1
                2
        ]

        """
        return wrap_s(self._s.limit(num_elements))

    def slice(self, offset: int, length: int) -> "Series":
        """
        Get a slice of this Series.

        Parameters
        ----------
        offset
            Offset index.
        length
            Length of the slice.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.slice(1, 2)
        shape: (2,)
        Series: 'a' [i64]
        [
                2
                3
        ]

        """
        return wrap_s(self._s.slice(offset, length))

    def append(self, other: "Series") -> None:
        """
        Append a Series to this one.

        Parameters
        ----------
        other
            Series to append.

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
        self._s.append(other._s)

    def filter(self, predicate: "Series") -> "Series":
        """
        Filter elements by a boolean mask.

        Parameters
        ----------
        predicate
            Boolean mask.
        """
        if isinstance(predicate, list):
            predicate = Series("", predicate)
        return wrap_s(self._s.filter(predicate._s))

    def head(self, length: Optional[int] = None) -> "Series":
        """
        Get first N elements as Series.

        Parameters
        ----------
        length
            Length of the head.

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
        return wrap_s(self._s.head(length))

    def tail(self, length: Optional[int] = None) -> "Series":
        """
        Get last N elements as Series.

        Parameters
        ----------
        length
            Length of the tail.

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
        return wrap_s(self._s.tail(length))

    def take_every(self, n: int) -> "Series":
        """
        Take every nth value in the Series and return as new Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4])
        >>> s.take_every(2))
        shape: (2,)
        Series: '' [i64]
        [
                1
                3
        ]

        """
        return wrap_s(self._s.take_every(n))

    def sort(self, in_place: bool = False, reverse: bool = False) -> Optional["Series"]:
        """
        Sort this Series.

        Parameters
        ----------
        in_place
            Sort in place.
        reverse
            Reverse sort.

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
            self._s.sort_in_place(reverse)
            return None
        else:
            return wrap_s(self._s.sort(reverse))

    def argsort(self, reverse: bool = False) -> "Series":
        """
        Index location of the sorted variant of this Series.

        Returns
        -------
        indexes
            Indexes that can be used to sort this array.
        """
        return wrap_s(self._s.argsort(reverse))

    def arg_sort(self, reverse: bool = False) -> "Series":
        """
        ..deprecate::

        Index location of the sorted variant of this Series.

        Returns
        -------
        indexes
            Indexes that can be used to sort this array.
        """
        return wrap_s(self._s.argsort(reverse))

    def arg_unique(self) -> "Series":
        """
        Get unique index as Series.
        """
        return wrap_s(self._s.arg_unique())

    def arg_min(self) -> Optional[int]:
        """
        Get the index of the minimal value.
        """
        return self._s.arg_min()

    def arg_max(self) -> Optional[int]:
        """
        Get the index of the maximal value.
        """
        return self._s.arg_max()

    def unique(self) -> "Series":
        """
        Get unique elements in series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 2, 3])
        >>> s.unique()
        shape: (3,)
        Series: 'a' [i64]
        [
                1
                2
                3
        ]

        """
        return wrap_s(self._s.unique())

    def take(self, indices: Union[np.ndarray, tp.List[int]]) -> "Series":
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
        if isinstance(indices, list):
            indices = np.array(indices)
        return Series._from_pyseries(self._s.take(indices))

    def null_count(self) -> int:
        """
        Count the null values in this Series.
        """
        return self._s.null_count()

    def is_null(self) -> "Series":
        """
        Get mask of null values.

        Returns
        -------
        Boolean Series

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, None])
        >>> s.is_null()
        shape: (4,)
        Series: 'is_null' [bool]
        [
                false
                false
                false
                true
        ]

        """
        return Series._from_pyseries(self._s.is_null())

    def is_not_null(self) -> "Series":
        """
        Get mask of non null values.

        Returns
        -------
        Boolean Series

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, None])
        >>> s.is_not_null()
        shape: (4,)
        Series: 'is_not_null' [bool]
        [
                true
                true
                true
                false
        ]

        """
        return Series._from_pyseries(self._s.is_not_null())

    def is_finite(self) -> "Series":
        """
        Get mask of finite values if Series dtype is Float.

        Returns
        -------
        Boolean Series

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 2.0, 3.0])
        >>> s.is_finite()
        shape: (3,)
        Series: 'a' [bool]
        [
                true
                true
                true
        ]

        """
        return Series._from_pyseries(self._s.is_finite())

    def is_infinite(self) -> "Series":
        """
        Get mask of infinite values if Series dtype is Float.

        Returns
        -------
        Boolean Series

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 2.0, 3.0])
        >>> s.is_infinite()
        shape: (3,)
        Series: 'a' [bool]
        [
                false
                false
                false
        ]

        """
        return Series._from_pyseries(self._s.is_infinite())

    def is_nan(self) -> "Series":
        """
        Get mask of NaN values if Series dtype is Float.

        Returns
        -------
        Boolean Series

        Examples
        --------
        >>> import numpy as np
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, np.NaN])
        >>> s.take([1, 3])
        shape: (2,)
        Series: 'a' [i64]
        [
                2
                4
        ]

        """
        return Series._from_pyseries(self._s.is_nan())

    def is_not_nan(self) -> "Series":
        """
        Get negated mask of NaN values if Series dtype is_not Float.

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
        return Series._from_pyseries(self._s.is_not_nan())

    def is_in(self, other: "Series") -> "Series":
        """
        Check if elements of this Series are in the right Series, or List values of the right Series.

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

        """
        if type(other) is list:
            other = Series("", other)
        return wrap_s(self._s.is_in(other._s))

    def arg_true(self) -> "Series":
        """
        Get index values where Boolean Series evaluate True.

        Returns
        -------
        UInt32 Series
        """
        return Series._from_pyseries(self._s.arg_true())

    def is_unique(self) -> "Series":
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
        return wrap_s(self._s.is_unique())

    def is_first(self) -> "Series":
        """
        Get a mask of the first unique value.

        Returns
        -------
        Boolean Series
        """
        return wrap_s(self._s.is_first())

    def is_duplicated(self) -> "Series":
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
        return wrap_s(self._s.is_duplicated())

    def explode(self) -> "Series":
        """
        Explode a list or utf8 Series. This means that every item is expanded to a new row.

        Examples
        --------
        >>> s = pl.Series('a', [[1, 2], [3, 4], [9, 10]])
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
        return wrap_s(self._s.explode())

    def series_equal(self, other: "Series", null_equal: bool = False) -> bool:
        """
        Check if series is equal with another Series.

        Parameters
        ----------
        other
            Series to compare with.
        null_equal
            Consider null values as equal.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s2 = pl.Series("b", [4, 5, 6])
        >>> s.series_equal(s))
        True
        >>> s.series_equal(s2))
        False

        """
        return self._s.series_equal(other._s, null_equal)

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

    @property
    def shape(self) -> Tuple[int]:
        """
        Shape of this Series.
        """
        return (self._s.len(),)

    def __len__(self) -> int:
        return self.len()

    def cast(self, dtype: Type[DataType], strict: bool = True) -> "Series":
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
        >>> s = pl.Series("a", ["2020-01-01", "2020-01-02", "2020-01-03"])
        shape: (3,)
        Series: 'a' [str]
        [
            "2020-01-01"
            "2020-01-02"
            "2020-01-03"
        ]
        >>> s.cast(pl.datatypes.Date)
        shape: (3,)
        Series: 'a' [date]
        [
            2020-01-01
            2020-01-02
            2020-01-03
        ]

        """
        if dtype == int:
            dtype = Int64
        elif dtype == str:
            dtype = Utf8
        elif dtype == float:
            dtype = Float64
        return wrap_s(self._s.cast(str(dtype), strict))

    def to_list(self) -> tp.List[Optional[Any]]:
        """
        Convert this Series to a Python List. This operation clones data.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.to_list()
        [1, 2, 3]
        >>> type(s)
        <class 'list'>

        """
        if self.dtype != Object:
            return self.to_arrow().to_pylist()
        return self._s.to_list()

    def __iter__(self) -> "SeriesIter":
        return SeriesIter(self.len(), self)

    def rechunk(self, in_place: bool = False) -> Optional["Series"]:
        """
        Create a single chunk of memory for this Series.

        Parameters
        ----------
        in_place
            In place or not.
        """
        opt_s = self._s.rechunk(in_place)
        if in_place:
            return None
        else:
            return wrap_s(opt_s)

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

    def is_datetime(self) -> bool:
        """
        Check if this Series datatype is a datetime.

        Examples
        --------
        >>> s = pl.Series([date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3)])
        >>> s.is_datetime()
        True

        """
        return self.dtype in (Date, Datetime)

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
        Checks if this Series datatype is a Utf8.

        Examples
        --------
        >>> s = pl.Series("x", ["a", "b", "c"])
        >>> s.is_utf8()
        True
        """
        return self.dtype is Utf8

    def view(self, ignore_nulls: bool = False) -> np.ndarray:
        """
        Get a view into this Series data with a numpy array. This operation doesn't clone data, but does not include
        missing values. Don't use this unless you know what you are doing.

        .. warning::

            This function can lead to undefined behavior in the following cases:

            >>> # returns a view to a piece of memory that is already dropped.
            >>> pl.Series([1, 3, 5]).sort().view()

            >>> # Sums invalid data that is missing.
            >>> pl.Series([1, 2, None]).view().sum()

        """
        if not ignore_nulls:
            assert self.null_count() == 0

        ptr_type = dtype_to_ctype(self.dtype)
        ptr = self._s.as_single_ptr()
        array = _ptr_to_numpy(ptr, self.len(), ptr_type)
        array.setflags(write=False)
        return array

    def __array__(self, dtype=None) -> np.ndarray:  # type: ignore
        return self.to_numpy().__array__(dtype)

    def __array_ufunc__(
        self, ufunc: Callable[..., Any], method: str, *inputs: Any, **kwargs: Any
    ) -> "Series":
        """
        Numpy universal functions.
        """
        if self._s.n_chunks() > 0:
            self._s.rechunk(in_place=True)

        if method == "__call__":
            args: tp.List[Union[Number, np.ndarray]] = []
            for arg in inputs:
                if isinstance(arg, Number):
                    args.append(arg)
                elif isinstance(arg, Series):
                    args.append(arg.view(ignore_nulls=True))
                else:
                    return NotImplemented

            if "dtype" in kwargs:
                dtype = kwargs.pop("dtype")
            else:
                dtype = self.dtype

            try:
                f = get_ffi_func("apply_ufunc_<>", dtype, self._s)
                series = f(lambda out: ufunc(*args, out=out, **kwargs))
                return wrap_s(series)
            except TypeError:
                # some integer to float ufuncs do not work, try on f64
                s = self.cast(pl.Float64)
                args[0] = s.view(ignore_nulls=True)
                f = get_ffi_func("apply_ufunc_<>", pl.Float64, self._s)
                series = f(lambda out: ufunc(*args, out=out, **kwargs))
                return wrap_s(series)

        else:
            return NotImplemented

    def to_numpy(
        self, *args: Any, zero_copy_only: bool = False, **kwargs: Any
    ) -> np.ndarray:
        """
        Convert this Series to numpy. This operation clones data but is completely safe.

        If you want a zero-copy view and know what you are doing, use `.view()`.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.to_numpy()
        >>> s
        [1 2 3]
        >>> type(s)
        <class 'numpy.ndarray'>

        Parameters
        ----------
        args
            args will be sent to pyarrow.Array.to_numpy.
        zero_copy_only
            If True, an exception will be raised if the conversion to a numpy
            array would require copying the underlying data (e.g. in presence
            of nulls, or for non-primitive types).
        kwargs
            kwargs will be sent to pyarrow.Array.to_numpy
        """
        if _PYARROW_AVAILABLE:
            return self.to_arrow().to_numpy(
                *args, zero_copy_only=zero_copy_only, **kwargs
            )
        else:
            if self.null_count() == 0:
                return self.view(ignore_nulls=True)
            return self._s.to_numpy()

    def to_arrow(self) -> "pa.Array":
        """
        Get the underlying Arrow Array. If the Series contains only a single chunk
        this operation is zero copy.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.to_arrow()
        >>> s
        [
        1,
        2,
        3
        ]
        >>> type(s)
        <class 'pyarrow.lib.Int64Array'>

        """
        return self._s.to_arrow()

    def set(self, filter: "Series", value: Union[int, float]) -> "Series":
        """
        Set masked values.

        Parameters
        ----------
        filter
            Boolean mask.
        value
            Value to replace the the masked values with.
        """
        f = get_ffi_func("set_with_mask_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(filter._s, value))

    def set_at_idx(
        self, idx: Union["Series", np.ndarray], value: Union[int, float]
    ) -> "Series":
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
        New allocated Series
        """
        f = get_ffi_func("set_at_idx_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        if isinstance(idx, Series):
            idx_array = idx.view()
        elif isinstance(idx, np.ndarray):
            if not idx.data.c_contiguous:
                idx_array = np.ascontiguousarray(idx, dtype=np.uint64)
            else:
                idx_array = idx
                if idx_array.dtype != np.uint64:
                    idx_array = np.array(idx_array, np.uint64)

        else:
            idx_array = np.array(idx, dtype=np.uint64)

        return wrap_s(f(idx_array, value))

    def clone(self) -> "Series":
        """
        Cheap deep clones.
        """
        return wrap_s(self._s.clone())

    def __copy__(self) -> "Series":  # type: ignore
        return self.clone()

    def __deepcopy__(self, memodict={}) -> "Series":  # type: ignore
        return self.clone()

    def fill_null(self, strategy: Union[str, "pl.Expr"]) -> "Series":
        """
        Fill null values with a filling strategy.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, None])
        >>> s.fill_null('forward'))
        shape: (4,)
        Series: '' [i64]
        [
                1
                2
                3
                3
        ]
        >>> s.fill_null('min'))
        shape: (4,)
        Series: 'a' [i64]
        [
                1
                2
                3
                1
        ]

        Parameters
        ----------
        strategy

        Fill null strategy or a value
               * "backward"
               * "forward"
               * "min"
               * "max"
               * "mean"
               * "one"
               * "zero"
        """
        if not isinstance(strategy, str):
            return self.to_frame().select(pl.col(self.name).fill_null(strategy))[
                self.name
            ]
        return wrap_s(self._s.fill_null(strategy))

    def round(self, decimals: int) -> "Series":
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
        return wrap_s(self._s.round(decimals))

    def dot(self, other: "Series") -> Optional[float]:
        """
        Compute the dot/inner product between two Series

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s2 = pl.Series("b", [4.0, 5.0, 6.0])
        >>> s.dot(s2)
        32.0

        Parameters
        ----------
        other
            Series to compute dot product with
        """
        return self._s.dot(other._s)

    def mode(self) -> "Series":
        """
        Compute the most occurring value(s). Can return multiple Values

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
        return wrap_s(self._s.mode())

    def apply(
        self,
        func: Callable[[Any], Any],
        return_dtype: Optional[Type[DataType]] = None,
    ) -> "Series":
        """
        Apply a function over elements in this Series and return a new Series.

        If the function returns another datatype, the return_dtype arg should be set, otherwise the method will fail.

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
            Output datatype. If none is given, the same datatype as this Series will be used.

        Returns
        -------
        Series
        """
        if return_dtype == str:
            return_dtype = Utf8
        elif return_dtype == int:
            return_dtype = Int64
        elif return_dtype == float:
            return_dtype = Float64
        elif return_dtype == bool:
            return_dtype = Boolean

        return wrap_s(self._s.apply_lambda(func, return_dtype))

    def shift(self, periods: int = 1) -> "Series":
        """
        Shift the values by a given period and fill the parts that will be empty due to this operation
        with `Nones`.

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
        return wrap_s(self._s.shift(periods))

    def shift_and_fill(self, periods: int, fill_value: "pl.Expr") -> "Series":
        """
        Shift the values by a given period and fill the parts that will be empty due to this operation
        with the result of the `fill_value` expression.

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).
        fill_value
            Fill None values with the result of this expression.
        """
        return self.to_frame().select(
            pl.col(self.name).shift_and_fill(periods, fill_value)
        )[self.name]

    def zip_with(self, mask: "Series", other: "Series") -> "Series":
        """
        Where mask evaluates true, take values from self. Where mask evaluates false, take values from other.

        Parameters
        ----------
        mask
            Boolean Series.
        other
            Series of same type.

        Returns
        -------
        New Series
        """
        return wrap_s(self._s.zip_with(mask._s, other._s))

    def rolling_min(
        self,
        window_size: int,
        weights: Optional[tp.List[float]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> "Series":
        """
        apply a rolling min (moving min) over the values in this array.
        A window of length `window_size` will traverse the array. The values that fill this window
        will (optionally) be multiplied with the weights given by the `weight` vector. The resultingParameters
        values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing a result.
            If None, it will be set equal to window size.
        center
            Set the labels at the center of the window

        Examples
        --------
        >>> s = pl.Series("a", [100, 200, 300, 400, 500])
        >>> s.rolling_min(window_size=3)
        shape: (5,)
        Series: '' [i64]
        [
                null
                null
                100
                200
                300
        ]

        """
        if min_periods is None:
            min_periods = window_size
        return wrap_s(self._s.rolling_min(window_size, weights, min_periods, center))

    def rolling_max(
        self,
        window_size: int,
        weights: Optional[tp.List[float]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> "Series":
        """
        Apply a rolling max (moving max) over the values in this array.
        A window of length `window_size` will traverse the array. The values that fill this window
        will (optionally) be multiplied with the weights given by the `weight` vector. The resultingParameters
        values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing a result.
            If None, it will be set equal to window size.
        center
            Set the labels at the center of the window

        Examples
        --------
        >>> s = pl.Series("a", [100, 200, 300, 400, 500])
        >>> s.rolling_max(window_size=2)
        shape: (5,)
        Series: '' [i64]
        [
                null
                null
                300
                400
                500
        ]

        """
        if min_periods is None:
            min_periods = window_size
        return wrap_s(self._s.rolling_max(window_size, weights, min_periods, center))

    def rolling_mean(
        self,
        window_size: int,
        weights: Optional[tp.List[float]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> "Series":
        """
        Apply a rolling mean (moving mean) over the values in this array.
        A window of length `window_size` will traverse the array. The values that fill this window
        will (optionally) be multiplied with the weights given by the `weight` vector. The resultingParameters
        values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing a result.
            If None, it will be set equal to window size.
        center
            Set the labels at the center of the window

        Examples
        --------
        >>> s = pl.Series("a", [100, 200, 300, 400, 500])
        >>> s.rolling_mean(window_size=2)
        shape: (5,)
        Series: '' [i64]
        [
                null
                150
                250
                350
                450
        ]

        """
        if min_periods is None:
            min_periods = window_size
        return wrap_s(self._s.rolling_mean(window_size, weights, min_periods, center))

    def rolling_sum(
        self,
        window_size: int,
        weights: Optional[tp.List[float]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> "Series":
        """
        Apply a rolling sum (moving sum) over the values in this array.
        A window of length `window_size` will traverse the array. The values that fill this window
        will (optionally) be multiplied with the weights given by the `weight` vector. The resultingParameters
        values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window.
        weight
            An optional slice with the same length of the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing a result.
            If None, it will be set equal to window size.
        center
            Set the labels at the center of the window

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4, 5])
        >>> s.rolling_sum(window_size=2)
        shape: (5,)
        Series: '' [i64]
        [
                null
                3
                5
                7
                9
        ]

        """
        if min_periods is None:
            min_periods = window_size
        return wrap_s(self._s.rolling_sum(window_size, weights, min_periods, center))

    def rolling_std(
        self,
        window_size: int,
        weights: Optional[tp.List[float]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> "Series":
        """
        Compute a rolling std dev

        A window of length `window_size` will traverse the array. The values that fill this window
        will (optionally) be multiplied with the weights given by the `weight` vector. The resultingParameters
        values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing a result.
            If None, it will be set equal to window size.
        center
            Set the labels at the center of the window

        """
        if min_periods is None:
            min_periods = window_size
        return wrap_s(self._s.rolling_std(window_size, weights, min_periods, center))

    def rolling_var(
        self,
        window_size: int,
        weights: Optional[tp.List[float]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> "Series":
        """
        Compute a rolling variance.

        A window of length `window_size` will traverse the array. The values that fill this window
        will (optionally) be multiplied with the weights given by the `weight` vector. The resultingParameters
        values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing a result.
            If None, it will be set equal to window size.
        center
            Set the labels at the center of the window

        """
        if min_periods is None:
            min_periods = window_size
        return wrap_s(self._s.rolling_var(window_size, weights, min_periods, center))

    def rolling_apply(
        self, window_size: int, function: Callable[["pl.Series"], Any]
    ) -> "pl.Series":
        """
        Allows a custom rolling window function.
        Prefer the specific rolling window functions over this one, as they are faster.
        Prefer:
            * rolling_min
            * rolling_max
            * rolling_mean
            * rolling_sum
        Parameters
        ----------
        window_size
            Size of the rolling window
        function
            Aggregation function
        Examples
        --------
        >>> s = pl.Series("A", [1.0, 2.0, 9.0, 2.0, 13.0])
        >>> s.rolling_apply(window_size=3, function=lambda s: s.std())
        shape: (5,)
        Series: 'A' [f64]
        [
            null
            null
            4.358898943540674
            4.041451884327381
            5.5677643628300215
        ]
        """
        return self.to_frame().select(
            pl.col(self.name).rolling_apply(window_size, function)  # type: ignore
        )[self.name]

    def rolling_median(self, window_size: int) -> "Series":
        """
        Compute a rolling median

        Parameters
        ----------
        window_size
            Size of the rolling window
        """
        return self.to_frame().select(
            pl.col(self.name).rolling_median(window_size)  # type: ignore
        )[self.name]

    def rolling_quantile(self, window_size: int, quantile: float) -> "Series":
        """
        Compute a rolling quantile

        Parameters
        ----------
        window_size
            Size of the rolling window
        quantile
            quantile to compute
        """
        return self.to_frame().select(
            pl.col(self.name).rolling_quantile(window_size, quantile)  # type: ignore
        )[self.name]

    def rolling_skew(self, window_size: int, bias: bool = True) -> "Series":
        """
        Compute a rolling skew
        window_size
            Size of the rolling window
        bias
            If False, then the calculations are corrected for statistical bias.
        """
        return self.to_frame().select(
            pl.col(self.name).rolling_skew(window_size, bias)  # type: ignore
        )[self.name]

    def sample(
        self,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        with_replacement: bool = False,
    ) -> "Series":
        """
        Sample from this Series by setting either `n` or `frac`.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4, 5])
        >>> s.sample(2)
        shape: (2,)
        Series: 'a' [i64]
        [
                1
                5
        ]

        Parameters
        ----------
        n
            Number of samples < self.len().
        frac
            Fraction between 0.0 and 1.0 .
        with_replacement
            sample with replacement.
        """
        if n is not None:
            return wrap_s(self._s.sample_n(n, with_replacement))
        return wrap_s(self._s.sample_frac(frac, with_replacement))

    def peak_max(self) -> "Series":
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

    def peak_min(self) -> "Series":
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

    def shrink_to_fit(self, in_place: bool = False) -> Optional["Series"]:
        """
        Shrink memory usage of this Series to fit the exact capacity needed to hold the data.
        """
        if in_place:
            self._s.shrink_to_fit()
            return None
        else:
            series = self.clone()
            series._s.shrink_to_fit()
            return series

    @property
    def dt(self) -> "DateTimeNameSpace":
        """
        Create an object namespace of all datetime related methods.
        """
        return DateTimeNameSpace(self)

    @property
    def arr(self) -> "ListNameSpace":
        """
        Create an object namespace of all list related methods.
        """
        return ListNameSpace(self)

    @property
    def str(self) -> "StringNameSpace":
        """
        Create an object namespace of all string related methods.
        """
        return StringNameSpace(self)

    def hash(self, k0: int = 0, k1: int = 1, k2: int = 2, k3: int = 3) -> "pl.Series":
        """
        Hash the Series.

        The hash value is of type `UInt64`

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.hash(k0=42)
        shape: (3,)
        Series: 'a' [u64]
        [
                18040498172617206516
                5352755651785478209
                3939059409923356085
        ]

        Parameters
        ----------
        k0
            seed parameter
        k1
            seed parameter
        k2
            seed parameter
        k3
            seed parameter
        """
        return wrap_s(self._s.hash(k0, k1, k2, k3))

    def reinterpret(self, signed: bool = True) -> "Series":
        """
        Reinterpret the underlying bits as a signed/unsigned integer.
        This operation is only allowed for 64bit integers. For lower bits integers,
        you can safely use that cast operation.

        Parameters
        ----------
        signed
            True -> pl.Int64
            False -> pl.UInt64
        """
        return wrap_s(self._s.reinterpret(signed))

    def interpolate(self) -> "Series":
        """
        Interpolate intermediate values. The interpolation method is linear.

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
        return wrap_s(self._s.interpolate())

    def abs(self) -> "Series":
        """
        Take absolute values
        """
        return np.abs(self)  # type: ignore

    def rank(self, method: str = "average") -> "Series":  # type: ignore
        """
        Assign ranks to data, dealing with ties appropriately.

        Parameters
        ----------
        method
            {'average', 'min', 'max', 'dense', 'ordinal'}, optional
            The method used to assign ranks to tied elements.
            The following methods are available (default is 'average'):
              * 'average': The average of the ranks that would have been assigned to
                all the tied values is assigned to each value.
              * 'min': The minimum of the ranks that would have been assigned to all
                the tied values is assigned to each value.  (This is also
                referred to as "competition" ranking.)
              * 'max': The maximum of the ranks that would have been assigned to all
                the tied values is assigned to each value.
              * 'dense': Like 'min', but the rank of the next highest element is
                assigned the rank immediately after those assigned to the tied
                elements.
              * 'ordinal': All values are given a distinct rank, corresponding to
                the order that the values occur in `a`.
        """
        return wrap_s(self._s.rank(method))

    def diff(self, n: int = 1, null_behavior: str = "ignore") -> "Series":  # type: ignore
        """
        Calculate the n-th discrete difference.

        Parameters
        ----------
        n
            number of slots to shift
        null_behavior
            {'ignore', 'drop'}
        """
        return wrap_s(self._s.diff(n, null_behavior))

    def skew(self, bias: bool = True) -> Optional[float]:
        """Compute the sample skewness of a data set.
        For normally distributed data, the skewness should be about zero. For
        unimodal continuous distributions, a skewness value greater than zero means
        that there is more weight in the right tail of the distribution. The
        function `skewtest` can be used to determine if the skewness value
        is close enough to zero, statistically speaking.


        See scipy.stats for more information.

        Parameters
        ----------
        bias : bool, optional
            If False, then the calculations are corrected for statistical bias.

        Notes
        -----
        The sample skewness is computed as the Fisher-Pearson coefficient
        of skewness, i.e.
        .. math::
            g_1=\frac{m_3}{m_2^{3/2}}
        where
        .. math::
            m_i=\frac{1}{N}\sum_{n=1}^N(x[n]-\bar{x})^i
        is the biased sample :math:`i\texttt{th}` central moment, and
        :math:`\bar{x}` is
        the sample mean.  If ``bias`` is False, the calculations are
        corrected for bias and the value computed is the adjusted
        Fisher-Pearson standardized moment coefficient, i.e.
        .. math::
            G_1=\frac{k_3}{k_2^{3/2}}=
                \frac{\sqrt{N(N-1)}}{N-2}\frac{m_3}{m_2^{3/2}}.
        """
        return self._s.skew(bias)

    def kurtosis(self, fisher: bool = True, bias: bool = True) -> Optional[float]:
        """Compute the kurtosis (Fisher or Pearson) of a dataset.
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
            If False, then the calculations are corrected for statistical bias.
        """
        return self._s.kurtosis(fisher, bias)

    def clip(self, min_val: Union[int, float], max_val: Union[int, float]) -> "Series":
        """
        Clip (limit) the values in an array.

        Parameters
        ----------
        min_val, max_val
            Minimum and maximum value.
        """
        return self.to_frame().select(
            pl.col(self.name).clip(min_val, max_val)  # type: ignore
        )[self.name]

    def str_concat(self, delimiter: str = "-") -> "Series":  # type: ignore
        """
        Vertically concat the values in the Series to a single string value.

        Returns
        -------
        Series of dtype Utf8

        Examples
        >>> assert pl.Series([1, None, 2]).str_concat("-")[0] == "1-null-2"

        """
        return self.to_frame().select(
            pl.col(self.name).delimiter(delimiter)  # type: ignore
        )[self.name]


class StringNameSpace:
    """
    Series.str namespace.
    """

    def __init__(self, series: "Series"):
        self._s = series._s

    def strptime(self, datatype: DataType, fmt: Optional[str] = None) -> Series:
        """
        Parse a Series of dtype Utf8 to a Date/Datetime Series.

        Parameters
        ----------
        datatype
            Date or Datetime.
        fmt
            formatting syntax. [Read more](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html)

        Returns
        -------
        A Date/ Datetime Series
        """
        if datatype == Date:
            return wrap_s(self._s.str_parse_date(fmt))
        if datatype == Datetime:
            return wrap_s(self._s.str_parse_datetime(fmt))
        raise NotImplementedError

    def lengths(self) -> Series:
        """
        Get length of the string values in the Series.

        Returns
        -------
        Series[u32]
        """
        return wrap_s(self._s.str_lengths())

    def contains(self, pattern: str) -> Series:
        """
        Check if strings in Series contain regex pattern.

        Parameters
        ----------
        pattern
            A valid regex pattern.

        Returns
        -------
        Boolean mask
        """
        return wrap_s(self._s.str_contains(pattern))

    def json_path_match(self, json_path: str) -> Series:
        """
        Extract the first match of json string with provided JSONPath expression.
        Throw errors if encounter invalid json strings.
        All return value will be casted to Utf8 regardless of the original value.
        Documentation on JSONPath standard: https://goessner.net/articles/JsonPath/

        Parameters
        ----------
        json_path
            A valid JSON path query string

        Returns
        -------
        Utf8 array. Contain null if original value is null or the json_path return nothing.

        Examples
        --------

        >>> df = pl.DataFrame({
        'json_val':['{"a":"1"}',None,'{"a":2}', '{"a":2.1}', '{"a":true}'
        })
        >>> df.select(pl.col('json_val').str.json_path_match('$.a')
        shape: (5,)
        Series: 'json_val' [str]
        [
            "1"
            null
            "2"
            "2.1"
            "true"
        ]
        """
        return wrap_s(self._s.str_json_path_match(json_path))

    def replace(self, pattern: str, value: str) -> Series:
        """
        Replace first regex match with a string value.

        Parameters
        ----------
        pattern
            A valid regex pattern.
        value
            Substring to replace.
        """
        return wrap_s(self._s.str_replace(pattern, value))

    def replace_all(self, pattern: str, value: str) -> Series:
        """
        Replace all regex matches with a string value.

        Parameters
        ----------
        pattern
            A valid regex pattern.
        value
            Substring to replace.
        """
        return wrap_s(self._s.str_replace_all(pattern, value))

    def to_lowercase(self) -> Series:
        """
        Modify the strings to their lowercase equivalent.
        """
        return wrap_s(self._s.str_to_lowercase())

    def to_uppercase(self) -> Series:
        """
        Modify the strings to their uppercase equivalent.
        """
        return wrap_s(self._s.str_to_uppercase())

    def rstrip(self) -> Series:
        """
        Remove trailing whitespace.
        """
        return self.replace(r"[ \t]+$", "")

    def lstrip(self) -> Series:
        """
        Remove leading whitespace.
        """
        return self.replace(r"^\s*", "")

    def slice(self, start: int, length: Optional[int] = None) -> Series:
        """
        Create subslices of the string values of a Utf8 Series.

        Parameters
        ----------
        start
            Start of the slice (negative indexing may be used).
        length
            Optional length of the slice.

        Returns
        -------
        Series of Utf8 type
        """
        return wrap_s(self._s.str_slice(start, length))


class ListNameSpace:
    """
    Series.dt namespace.
    """

    def __init__(self, series: Series):
        self._s = series._s

    def lengths(self) -> Series:
        """
        Get the length of the arrays as UInt32.
        """
        return wrap_s(self._s.arr_lengths())

    def sum(self) -> Series:
        """
        Sum all the arrays in the list
        """
        s = wrap_s(self._s)
        return s.to_frame().select(pl.col(s.name).arr.sum())  # type: ignore

    def max(self) -> Series:
        """
        Compute the max value of the arrays in the list
        """
        s = wrap_s(self._s)
        return s.to_frame().select(pl.col(s.name).arr.max())  # type: ignore

    def min(self) -> Series:
        """
        Compute the min value of the arrays in the list
        """
        s = wrap_s(self._s)
        return s.to_frame().select(pl.col(s.name).arr.min())  # type: ignore

    def mean(self) -> Series:
        """
        Compute the mean value of the arrays in the list
        """
        s = wrap_s(self._s)
        return s.to_frame().select(pl.col(s.name).arr.min())  # type: ignore

    def sort(self, reverse: bool) -> Series:
        """
        Sort the arrays in the list
        """
        s = wrap_s(self._s)
        return s.to_frame().select(pl.col(s.name).arr.sort(reverse))  # type: ignore

    def reverse(self) -> Series:
        """
        Reverse the arrays in the list
        """
        s = wrap_s(self._s)
        return s.to_frame().select(pl.col(s.name).arr.reverse())  # type: ignore

    def unique(self) -> Series:
        """
        Get the unique/distinct values in the list
        """
        s = wrap_s(self._s)
        return s.to_frame().select(pl.col(s.name).arr.unique())  # type: ignore

    def concat(self, other: Union[tp.List[Series], Series]) -> "Series":
        """
        Concat the arrays in a Series dtype List in linear time.

        Parameters
        ----------
        other
            Columns to concat into a List Series
        """
        if not isinstance(other, list):
            other = [other]
        sthis = wrap_s(self._s)
        names = [s.name for s in other]
        names.insert(0, sthis.name)
        df = pl.DataFrame(other)
        df.insert_at_idx(0, sthis)
        return df.select(pl.concat_list(names))[sthis.name]  # type: ignore


class DateTimeNameSpace:
    """
    Series.dt namespace.
    """

    def __init__(self, series: Series):
        self._s = series._s

    def __getitem__(self, item: int) -> Union[date, datetime]:
        s = wrap_s(self._s)
        out = wrap_s(self._s)[item]
        return _to_python_datetime(out, s.dtype)

    def strftime(self, fmt: str) -> Series:
        """
        Format Date/datetime with a formatting rule: See `chrono strftime/strptime <https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html>`_.

        Returns
        -------
        Utf8 Series
        """
        return wrap_s(self._s.strftime(fmt))

    def year(self) -> Series:
        """
        Extract the year from the underlying date representation.
        Can be performed on Date and Datetime.

        Returns the year number in the calendar date.

        Returns
        -------
        Year as Int32
        """
        return wrap_s(self._s.year())

    def month(self) -> Series:
        """
        Extract the month from the underlying date representation.
        Can be performed on Date and Datetime

        Returns the month number starting from 1.
        The return value ranges from 1 to 12.

        Returns
        -------
        Month as UInt32
        """
        return wrap_s(self._s.month())

    def week(self) -> Series:
        """
        Extract the week from the underlying date representation.
        Can be performed on Date and Datetime

        Returns the ISO week number starting from 1.
        The return value ranges from 1 to 53. (The last week of year differs by years.)

        Returns
        -------
        Week number as UInt32
        """
        return wrap_s(self._s.week())

    def weekday(self) -> Series:
        """
        Extract the week day from the underlying date representation.
        Can be performed on Date and Datetime.

        Returns the weekday number where monday = 0 and sunday = 6

        Returns
        -------
        Week day as UInt32
        """
        return wrap_s(self._s.weekday())

    def day(self) -> Series:
        """
        Extract the day from the underlying date representation.
        Can be performed on Date and Datetime.

        Returns the day of month starting from 1.
        The return value ranges from 1 to 31. (The last day of month differs by months.)

        Returns
        -------
        Day as UInt32
        """
        return wrap_s(self._s.day())

    def ordinal_day(self) -> Series:
        """
        Extract ordinal day from underlying date representation.
        Can be performed on Date and Datetime.

        Returns the day of year starting from 1.
        The return value ranges from 1 to 366. (The last day of year differs by years.)

        Returns
        -------
        Day as UInt32
        """
        return wrap_s(self._s.ordinal_day())

    def hour(self) -> Series:
        """
        Extract the hour from the underlying DateTime representation.
        Can be performed on Datetime.

        Returns the hour number from 0 to 23.

        Returns
        -------
        Hour as UInt32
        """
        return wrap_s(self._s.hour())

    def minute(self) -> Series:
        """
        Extract the minutes from the underlying DateTime representation.
        Can be performed on Datetime.

        Returns the minute number from 0 to 59.

        Returns
        -------
        Minute as UInt32
        """
        return wrap_s(self._s.minute())

    def second(self) -> Series:
        """
        Extract the seconds the from underlying DateTime representation.
        Can be performed on Datetime.

        Returns the second number from 0 to 59.

        Returns
        -------
        Second as UInt32
        """
        return wrap_s(self._s.second())

    def nanosecond(self) -> Series:
        """
        Extract the nanoseconds from the underlying DateTime representation.
        Can be performed on Datetime.

        Returns the number of nanoseconds since the whole non-leap second.
        The range from 1,000,000,000 to 1,999,999,999 represents the leap second.

        Returns
        -------
        Nanosecond as UInt32
        """
        return wrap_s(self._s.nanosecond())

    def timestamp(self) -> Series:
        """
        Return timestamp in ms as Int64 type.
        """
        return wrap_s(self._s.timestamp())

    def to_python_datetime(self) -> Series:
        """
        Go from Date/Datetime to python DateTime objects
        """

        return (self.timestamp() // 1000).apply(
            lambda ts: datetime.utcfromtimestamp(ts), Object
        )

    def min(self) -> Union[date, datetime]:
        """
        Return minimum as python DateTime
        """
        s = wrap_s(self._s)
        out = s.min()
        return _to_python_datetime(out, s.dtype)

    def max(self) -> Union[date, datetime]:
        """
        Return maximum as python DateTime
        """
        s = wrap_s(self._s)
        out = s.max()
        return _to_python_datetime(out, s.dtype)

    def median(self) -> Union[date, datetime]:
        """
        Return median as python DateTime
        """
        s = wrap_s(self._s)
        out = int(s.median())
        return _to_python_datetime(out, s.dtype)

    def mean(self) -> Union[date, datetime]:
        """
        Return mean as python DateTime
        """
        s = wrap_s(self._s)
        out = int(s.mean())
        return _to_python_datetime(out, s.dtype)

    def round(self, rule: str, n: int) -> Series:
        """
        Round the datetime.

        Parameters
        ----------
        rule
            Units of the downscaling operation.

            Any of:
                - "month"
                - "week"
                - "day"
                - "hour"
                - "minute"
                - "second"

        n
            Number of units (e.g. 5 "day", 15 "minute".
        """
        return wrap_s(self._s.round_datetime(rule, n))


def _to_python_datetime(
    value: Union[int, float], dtype: Type[DataType]
) -> Union[date, datetime]:
    if dtype == Date:
        # days to seconds
        # important to create from utc. Not doing this leads
        # to inconsistencies dependent on the timezone you are in.
        return datetime.utcfromtimestamp(value * 3600 * 24).date()
    elif dtype == Datetime:
        # ms to seconds
        return datetime.utcfromtimestamp(value // 1000)
    else:
        raise NotImplementedError


class SeriesIter:
    """
    Utility class that allows slow iteration over a `Series`.
    """

    def __init__(self, length: int, s: Series):
        self.len = length
        self.i = 0
        self.s = s

    def __iter__(self) -> "SeriesIter":
        return self

    def __next__(self) -> Any:
        if self.i < self.len:
            i = self.i
            self.i += 1
            return self.s[i]
        else:
            raise StopIteration


def out_to_dtype(out: Any) -> Union[Type[DataType], Type[np.ndarray]]:
    if isinstance(out, float):
        return Float64
    if isinstance(out, int):
        return Int64
    if isinstance(out, str):
        return Utf8
    if isinstance(out, bool):
        return Boolean
    if isinstance(out, Series):
        return List
    if isinstance(out, np.ndarray):
        return np.ndarray
    raise NotImplementedError
