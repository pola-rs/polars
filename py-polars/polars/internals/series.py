import sys
from datetime import date, datetime, timedelta
from numbers import Number
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    overload,
)

import numpy as np

try:
    import pyarrow as pa

    _PYARROW_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYARROW_AVAILABLE = False

from polars import internals as pli
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
except ImportError:  # pragma: no cover
    _DOCUMENTING = True

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
)
from polars.datatypes import List as PlList
from polars.datatypes import (
    Object,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Utf8,
    dtype_to_ctype,
    dtype_to_ffiname,
    maybe_cast,
    py_type_to_dtype,
)
from polars.utils import (
    _date_to_pl_date,
    _datetime_to_pl_timestamp,
    _ptr_to_numpy,
    _to_python_datetime,
    range_to_slice,
)

try:
    import pandas as pd

    _PANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PANDAS_AVAILABLE = False

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal  # pragma: no cover


def get_ffi_func(
    name: str,
    dtype: Type["DataType"],
    obj: "PySeries",
) -> Optional[Callable[..., Any]]:
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
        Object to find the method for.

    Returns
    -------
    ffi function, or None if not found
    """
    ffi_name = dtype_to_ffiname(dtype)
    fname = name.replace("<>", ffi_name)
    return getattr(obj, fname, None)


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
    <class 'polars.datatypes.Int64'>

    Constructing a Series with a specific dtype:

    >>> s2 = pl.Series("a", [1, 2, 3], dtype=pl.Float32)
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
    def _from_arrow(
        cls, name: str, values: "pa.Array", rechunk: bool = True
    ) -> "Series":
        """
        Construct a Series from an Arrow Array.
        """
        return cls._from_pyseries(arrow_to_pyseries(name, values, rechunk))

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

    def _comp(self, other: Any, op: str) -> "Series":
        if isinstance(other, datetime) and self.dtype == Datetime:
            ts = _datetime_to_pl_timestamp(other, self.time_unit)
            f = get_ffi_func(op + "_<>", Int64, self._s)
            return wrap_s(f(ts))  # type: ignore
        if isinstance(other, date) and self.dtype == Date:
            d = _date_to_pl_date(other)
            f = get_ffi_func(op + "_<>", Int32, self._s)
            return wrap_s(f(d))  # type: ignore

        if isinstance(other, Sequence) and not isinstance(other, str):
            other = Series("", other)
        if isinstance(other, Series):
            return wrap_s(getattr(self._s, op)(other._s))
        other = maybe_cast(other, self.dtype, self.time_unit)
        f = get_ffi_func(op + "_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __eq__(self, other: Any) -> "Series":  # type: ignore[override]
        return self._comp(other, "eq")

    def __ne__(self, other: Any) -> "Series":  # type: ignore[override]
        return self._comp(other, "neq")

    def __gt__(self, other: Any) -> "Series":
        return self._comp(other, "gt")

    def __lt__(self, other: Any) -> "Series":
        return self._comp(other, "lt")

    def __ge__(self, other: Any) -> "Series":
        return self._comp(other, "gt_eq")

    def __le__(self, other: Any) -> "Series":
        return self._comp(other, "lt_eq")

    def _arithmetic(self, other: Any, op_s: str, op_ffi: str) -> "Series":
        if isinstance(other, Series):
            return wrap_s(getattr(self._s, op_s)(other._s))
        other = maybe_cast(other, self.dtype, self.time_unit)
        f = get_ffi_func(op_ffi, self.dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __add__(self, other: Any) -> "Series":
        if isinstance(other, str):
            other = Series("", [other])
        return self._arithmetic(other, "add", "add_<>")

    def __sub__(self, other: Any) -> "Series":
        return self._arithmetic(other, "sub", "sub_<>")

    def __truediv__(self, other: Any) -> "Series":
        if self.is_datelike():
            raise ValueError("first cast to integer before dividing datelike dtypes")

        # this branch is exactly the floordiv function without rounding the floats
        if self.is_float():
            return self._arithmetic(other, "div", "div_<>")

        return self.cast(Float64) / other

    def __floordiv__(self, other: Any) -> "Series":
        if self.is_datelike():
            raise ValueError("first cast to integer before dividing datelike dtypes")
        result = self._arithmetic(other, "div", "div_<>")
        if self.is_float():
            result = result.floor()
        return result

    def __mul__(self, other: Any) -> "Series":
        if self.is_datelike():
            raise ValueError("first cast to integer before multiplying datelike dtypes")
        return self._arithmetic(other, "mul", "mul_<>")

    def __mod__(self, other: Any) -> "Series":
        if self.is_datelike():
            raise ValueError(
                "first cast to integer before applying modulo on datelike dtypes"
            )
        return self._arithmetic(other, "rem", "rem_<>")

    def __rmod__(self, other: Any) -> "Series":
        if self.is_datelike():
            raise ValueError(
                "first cast to integer before applying modulo on datelike dtypes"
            )
        return self._arithmetic(other, "rem", "rem_<>_rhs")

    def __radd__(self, other: Any) -> "Series":
        return self._arithmetic(other, "add", "add_<>_rhs")

    def __rsub__(self, other: Any) -> "Series":
        return self._arithmetic(other, "sub", "sub_<>_rhs")

    def __invert__(self) -> "Series":
        if self.dtype == Boolean:
            return wrap_s(self._s._not())
        return NotImplemented

    def __rtruediv__(self, other: Any) -> np.ndarray:
        if self.is_datelike():
            raise ValueError("first cast to integer before dividing datelike dtypes")
        if self.is_float():
            self.__rfloordiv__(other)

        if isinstance(other, int):
            other = float(other)

        return self.cast(Float64).__rfloordiv__(other)  # type: ignore

    def __rfloordiv__(self, other: Any) -> "Series":
        if self.is_datelike():
            raise ValueError("first cast to integer before dividing datelike dtypes")
        return self._arithmetic(other, "div", "div_<>_rhs")

    def __rmul__(self, other: Any) -> "Series":
        if self.is_datelike():
            raise ValueError("first cast to integer before multiplying datelike dtypes")
        return self._arithmetic(other, "mul", "mul_<>")

    def __pow__(self, power: float, modulo: None = None) -> "Series":
        if self.is_datelike():
            raise ValueError(
                "first cast to integer before raising datelike dtypes to a power"
            )
        return np.power(self, power)  # type: ignore

    def __rpow__(self, other: Any) -> "Series":
        if self.is_datelike():
            raise ValueError(
                "first cast to integer before raising datelike dtypes to a power"
            )
        return np.power(other, self)  # type: ignore

    def __neg__(self) -> "Series":
        return 0 - self

    def __getitem__(self, item: Union[int, "Series", range, slice]) -> Any:
        if isinstance(item, int):
            if item < 0:
                item = self.len() + item
            if self.dtype in (PlList, Object):
                f = get_ffi_func("get_<>", self.dtype, self._s)
                if f is None:
                    return NotImplemented
                out = f(item)
                if self.dtype == PlList:
                    if out is None:
                        return None
                    return wrap_s(out)
                return out

            return self._s.get_idx(item)
        # assume it is boolean mask
        if isinstance(item, Series):
            return wrap_s(self._s.filter(item._s))

        if isinstance(item, range):
            return self[range_to_slice(item)]

        # slice
        if isinstance(item, slice):
            start, stop, stride = item.indices(self.len())
            out = self.slice(start, stop - start)
            if stride != 1:
                return out.take_every(stride)
            else:
                return out

        raise NotImplementedError

    def __setitem__(
        self, key: Union[int, "Series", np.ndarray, List, Tuple], value: Any
    ) -> None:
        if isinstance(value, list):
            raise ValueError("cannot set with a list as value, use a primitive value")
        if isinstance(key, Series):
            if key.dtype == Boolean:
                self._s = self.set(key, value)._s
            elif key.dtype == UInt64:
                self._s = self.set_at_idx(key.cast(UInt32), value)._s
            elif key.dtype == UInt32:
                self._s = self.set_at_idx(key, value)._s
        # TODO: implement for these types without casting to series
        elif isinstance(key, np.ndarray) and key.dtype == np.bool_:
            # boolean numpy mask
            self._s = self.set_at_idx(np.argwhere(key)[:, 0], value)._s
        elif isinstance(key, (np.ndarray, list, tuple)):
            s = wrap_s(PySeries.new_u32("", np.array(key, np.uint32), True))
            self.__setitem__(s, value)
        elif isinstance(key, int) and not isinstance(key, bool):
            self.__setitem__([key], value)
        else:
            raise ValueError(f'cannot use "{key}" for indexing')

    def sqrt(self) -> "Series":
        """
        Compute the square root of the elements

        Syntactic sugar for

        >>> pl.Series([1, 2]) ** 0.5
        shape: (2,)
        Series: '' [f64]
        [
            1
            1.4142135623730951
        ]

        """
        return self ** 0.5

    def any(self) -> "Series":
        """
        Check if any boolean value in the column is `True`

        Returns
        -------
        Boolean literal
        """
        return self.to_frame().select(pli.col(self.name).any()).to_series()

    def all(self) -> "Series":
        """
        Check if all boolean values in the column are `True`

        Returns
        -------
        Boolean literal
        """
        return self.to_frame().select(pli.col(self.name).all()).to_series()

    def log(self) -> "Series":
        """
        Natural logarithm, element-wise.

        The natural logarithm log is the inverse of the exponential function, so that log(exp(x)) = x.
        The natural logarithm is logarithm in base e.
        """
        return np.log(self)  # type: ignore

    def log10(self) -> "Series":
        """
        Return the base 10 logarithm of the input array, element-wise.
        """
        return np.log10(self)  # type: ignore

    def exp(self) -> "Series":
        """
        Return the exponential element-wise
        """
        return np.exp(self)  # type: ignore

    def drop_nulls(self) -> "Series":
        """
        Create a new Series that copies data from this Series without null values.
        """
        return wrap_s(self._s.drop_nulls())

    def to_frame(self) -> "pli.DataFrame":
        """
        Cast this Series to a DataFrame.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> df = s.to_frame()
        >>> df
        shape: (3, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 1   │
        ├╌╌╌╌╌┤
        │ 2   │
        ├╌╌╌╌╌┤
        │ 3   │
        └─────┘

        >>> type(df)
        <class 'polars.internals.frame.DataFrame'>

        """
        return pli.wrap_df(PyDataFrame([self._s]))

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
        return self._s.dtype()

    @property
    def inner_dtype(self) -> Optional[Type[DataType]]:
        """
        Get the inner dtype in of a List typed Series

        Returns
        -------
        DataType
        """
        return self._s.inner_dtype()

    def describe(self) -> "pli.DataFrame":
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
        ┌────────────┬────────────────────┐
        │ statistic  ┆ value              │
        │ ---        ┆ ---                │
        │ str        ┆ f64                │
        ╞════════════╪════════════════════╡
        │ min        ┆ 1                  │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ max        ┆ 5                  │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ null_count ┆ 0.0                │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ mean       ┆ 3                  │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ std        ┆ 1.5811388300841898 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ count      ┆ 5                  │
        └────────────┴────────────────────┘

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
        stats: Dict[str, Union[Optional[float], int, str]]

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

    def sum(self) -> Union[int, float]:
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

    def product(self) -> Union[int, float]:
        """
        Reduce this Series to the product value.
        """
        return self.to_frame().select(pli.col(self.name).product()).to_series()[0]

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
        1

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

    def quantile(self, quantile: float, interpolation: str = "nearest") -> float:
        """
        Get the quantile value of this Series.

        Parameters
        ----------
        quantile
            quantile between 0.0 and 1.0

        interpolation
            interpolation type, options: ['nearest', 'higher', 'lower', 'midpoint', 'linear']

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.quantile(0.5)
        2

        """
        return self._s.quantile(quantile, interpolation)

    def to_dummies(self) -> "pli.DataFrame":
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

    def value_counts(self) -> "pli.DataFrame":
        """
        Count the unique values in a Series.

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
        return pli.wrap_df(self._s.value_counts())

    @property
    def name(self) -> str:
        """
        Get the name of this Series.
        """
        return self._s.name()

    def alias(self, name: str) -> "Series":
        """
        Rename the Series

        Parameters
        ----------
        name
            New name

        Returns
        -------

        """
        s = self.clone()
        s._s.rename(name)
        return s

    @overload
    def rename(self, name: str, in_place: Literal[False] = ...) -> "Series":
        ...

    @overload
    def rename(self, name: str, in_place: Literal[True]) -> None:
        ...

    @overload
    def rename(self, name: str, in_place: bool) -> Optional["Series"]:
        ...

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
            return None
        else:
            return self.alias(name)

    def chunk_lengths(self) -> List[int]:
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
        Series: 'a' [i64]
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
        Series: 'a' [i64]
        [
            1
            2
            3
        ]

        """
        return wrap_s(self._s.cummax(reverse))

    def cumprod(self, reverse: bool = False) -> "Series":
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
        return wrap_s(self._s.cumprod(reverse))

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
        >>> s
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

    def filter(self, predicate: Union["Series", list]) -> "Series":
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
        >>> s.take_every(2)
        shape: (2,)
        Series: 'a' [i64]
        [
            1
            3
        ]

        """
        return wrap_s(self._s.take_every(n))

    @overload
    def sort(
        self, reverse: bool = False, *, in_place: Literal[False] = ...
    ) -> "Series":
        ...

    @overload
    def sort(self, reverse: bool = False, *, in_place: Literal[True]) -> None:
        ...

    @overload
    def sort(
        self, reverse: bool = False, *, in_place: bool = False
    ) -> Optional["Series"]:
        ...

    def sort(
        self, reverse: bool = False, *, in_place: bool = False
    ) -> Optional["Series"]:
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

        Examples
        --------
        >>> s = pl.Series("a", [5, 3, 4, 1, 2])
        >>> s.argsort()
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
        >>> s.unique().sort()
        shape: (3,)
        Series: 'a' [i64]
        [
            1
            2
            3
        ]

        """
        return wrap_s(self._s.unique())

    def take(self, indices: Union[np.ndarray, List[int]]) -> "Series":
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
        return wrap_s(self._s.take(indices))

    def null_count(self) -> int:
        """
        Count the null values in this Series.
        """
        return self._s.null_count()

    def has_validity(self) -> bool:
        """
        Returns True if the Series has a validity bitmask. If there is none, it means that there are no null values.
        Use this to swiftly assert a Series does not have null values.
        """
        return self._s.has_validity()

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
        Series: 'a' [bool]
        [
            false
            false
            false
            true
        ]

        """
        return wrap_s(self._s.is_null())

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
        Series: 'a' [bool]
        [
            true
            true
            true
            false
        ]

        """
        return wrap_s(self._s.is_not_null())

    def is_finite(self) -> "Series":
        """
        Get mask of finite values if Series dtype is Float.

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
        return wrap_s(self._s.is_finite())

    def is_infinite(self) -> "Series":
        """
        Get mask of infinite values if Series dtype is Float.

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
        return wrap_s(self._s.is_infinite())

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
        return wrap_s(self._s.is_nan())

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
        return wrap_s(self._s.is_not_nan())

    def is_in(self, other: Union["Series", List]) -> "Series":
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
        if isinstance(other, list):
            other = Series("", other)
        return wrap_s(self._s.is_in(other._s))

    def arg_true(self) -> "Series":
        """
        Get index values where Boolean Series evaluate True.

        Returns
        -------
        UInt32 Series
        """
        return wrap_s(self._s.arg_true())

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
        return wrap_s(self._s.explode())

    def series_equal(
        self, other: "Series", null_equal: bool = False, strict: bool = False
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
            Don't allow different numerical dtypes, e.g. comparing `pl.UInt32` with a `pl.Int64` will return `False`.

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

    @property
    def shape(self) -> Tuple[int]:
        """
        Shape of this Series.
        """
        return (self._s.len(),)

    def __len__(self) -> int:
        return self.len()

    def cast(
        self,
        dtype: Union[Type[DataType], Type[int], Type[float], Type[str], Type[bool]],
        strict: bool = True,
    ) -> "Series":
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
        pl_dtype = py_type_to_dtype(dtype)
        return wrap_s(self._s.cast(str(pl_dtype), strict))

    def to_physical(self) -> "Series":
        """
        Cast to physical representation of the logical dtype.

        Date -> Int32
        Datetime -> Int64
        Time -> Int64
        other -> other
        """
        return wrap_s(self._s.to_physical())

    def to_list(self, use_pyarrow: bool = False) -> List[Optional[Any]]:
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

    def __iter__(self) -> "SeriesIter":
        return SeriesIter(self.len(), self)

    @overload
    def rechunk(self, in_place: Literal[False] = ...) -> "Series":
        ...

    @overload
    def rechunk(self, in_place: Literal[True]) -> None:
        ...

    @overload
    def rechunk(self, in_place: bool) -> Optional["Series"]:
        ...

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

            Returns a view to a piece of memory that is already dropped:

            >>> pl.Series([1, 3, 5]).sort().view()  # doctest: +IGNORE_RESULT

            Sums invalid data that is missing:

            >>> pl.Series([1, 2, None]).view().sum()  # doctest: +SKIP

        """
        if not ignore_nulls:
            assert not self.has_validity()

        ptr_type = dtype_to_ctype(self.dtype)
        ptr = self._s.as_single_ptr()
        array = _ptr_to_numpy(ptr, self.len(), ptr_type)
        array.setflags(write=False)
        return array

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return self.to_numpy().__array__(dtype)

    def __array_ufunc__(
        self, ufunc: Callable[..., Any], method: str, *inputs: Any, **kwargs: Any
    ) -> "Series":
        """
        Numpy universal functions.
        """
        if self._s.n_chunks() > 1:
            self._s.rechunk(in_place=True)

        if method == "__call__":
            args: List[Union[Number, np.ndarray]] = []
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
                if f is None:
                    return NotImplemented
                series = f(lambda out: ufunc(*args, out=out, **kwargs))
                return wrap_s(series)
            except TypeError:
                # some integer to float ufuncs do not work, try on f64
                s = self.cast(Float64)
                args[0] = s.view(ignore_nulls=True)
                f = get_ffi_func("apply_ufunc_<>", Float64, self._s)
                if f is None:
                    return NotImplemented
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
        kwargs
            kwargs will be sent to pyarrow.Array.to_numpy
        """

        def convert_to_date(arr):  # type: ignore
            if self.dtype == Date:
                tp = "datetime64[D]"
            elif self.dtype == Duration:
                tp = f"timedelta64[{self.time_unit}]"
            else:
                tp = f"datetime64[{self.time_unit}]"
            return arr.astype(tp)

        if _PYARROW_AVAILABLE and not self.is_datelike():
            return self.to_arrow().to_numpy(
                *args, zero_copy_only=zero_copy_only, **kwargs
            )
        else:
            if not self.has_validity():
                if self.is_datelike():
                    return convert_to_date(self.view(ignore_nulls=True))
                return self.view(ignore_nulls=True)
            if self.is_datelike():
                return convert_to_date(self._s.to_numpy())
            return self._s.to_numpy()

    def to_arrow(self) -> "pa.Array":
        """
        Get the underlying Arrow Array. If the Series contains only a single chunk
        this operation is zero copy.

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
        self,
        idx: Union["Series", np.ndarray, List[int], Tuple[int]],
        value: Union[int, float, str, bool],
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

        # the set_at_idx function expects a np.array of dtype u32
        f = get_ffi_func("set_at_idx_<>", self.dtype, self._s)
        if f is None:
            raise ValueError(
                f"could not find the FFI function needed to set at idx for series {self._s}"
            )
        if isinstance(idx, Series):
            # make sure the dtype matches
            idx = idx.cast(UInt32)
            idx_array = idx.view()
        elif isinstance(idx, np.ndarray):
            if not idx.data.c_contiguous:
                idx_array = np.ascontiguousarray(idx, dtype=np.uint32)
            else:
                idx_array = idx
                if idx_array.dtype != np.uint32:
                    idx_array = np.array(idx_array, np.uint32)

        else:
            idx_array = np.array(idx, dtype=np.uint32)

        return wrap_s(f(idx_array, value))

    def clone(self) -> "Series":
        """
        Cheap deep clones.
        """
        return wrap_s(self._s.clone())

    def __copy__(self) -> "Series":
        return self.clone()

    def __deepcopy__(self, memodict: Any = {}) -> "Series":
        return self.clone()

    def fill_nan(
        self, fill_value: Union[str, int, float, bool, "pli.Expr"]
    ) -> "Series":
        """
        Fill floating point NaN value with a fill value
        """
        return (
            self.to_frame().select(pli.col(self.name).fill_nan(fill_value)).to_series()
        )

    def fill_null(self, strategy: Union[str, int, "pli.Expr"]) -> "Series":
        """
        Fill null values with a filling strategy.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, None])
        >>> s.fill_null("forward")
        shape: (4,)
        Series: 'a' [i64]
        [
            1
            2
            3
            3
        ]
        >>> s.fill_null("min")
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
            return self.to_frame().select(pli.col(self.name).fill_null(strategy))[
                self.name
            ]
        return wrap_s(self._s.fill_null(strategy))

    def floor(self) -> "Series":
        """
        Floor underlying floating point array to the lowest integers smaller or equal to the float value.

        Only works on floating point Series
        """
        return wrap_s(self._s.floor())

    def ceil(self) -> "Series":
        """
        Ceil underlying floating point array to the heighest integers smaller or equal to the float value.

        Only works on floating point Series
        """
        return self.to_frame().select(pli.col(self.name).ceil()).to_series()

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

    def sin(self) -> "Series":
        """
        Compute the element-wise value for Trigonometric sine.

        Examples
        --------
        >>> import numpy as np
        >>> s = pl.Series("a", np.array((0.0, np.pi / 2.0, np.pi)))
        >>> s.sin()
        shape: (3,)
        Series: 'a' [f64]
        [
            0.0
            1
            1.2246467991473532e-16
        ]

        """
        return np.sin(self)  # type: ignore

    def cos(self) -> "Series":
        """
        Compute the element-wise value for Trigonometric cosine.

        Examples
        --------
        >>> import numpy as np
        >>> s = pl.Series("a", np.array((0.0, np.pi / 2.0, np.pi)))
        >>> s.cos()
        shape: (3,)
        Series: 'a' [f64]
        [
            1
            6.123233995736766e-17
            -1e0
        ]

        """
        return np.cos(self)  # type: ignore

    def tan(self) -> "Series":
        """
        Compute the element-wise value for Trigonometric tangent.

        Examples
        --------
        >>> import numpy as np
        >>> s = pl.Series("a", np.array((0.0, np.pi / 2.0, np.pi)))
        >>> s.tan()
        shape: (3,)
        Series: 'a' [f64]
        [
            0.0
            1.633123935319537e16
            -1.2246467991473532e-16
        ]

        """
        return np.tan(self)  # type: ignore

    def arcsin(self) -> "Series":
        """
        Compute the element-wise value for Trigonometric Inverse sine.

        Examples
        --------
        >>> import numpy as np
        >>> s = pl.Series("a", np.array((1.0, 0.0, -1)))
        >>> s.arcsin()
        shape: (3,)
        Series: 'a' [f64]
        [
            1.5707963267948966
            0.0
            -1.5707963267948966e0
        ]

        """
        return np.arcsin(self)  # type: ignore

    def arccos(self) -> "Series":
        """
        Compute the element-wise value for Trigonometric Inverse cosine.

        Examples
        --------
        >>> import numpy as np
        >>> s = pl.Series("a", np.array((1.0, 0.0, -1)))
        >>> s.arccos()
        shape: (3,)
        Series: 'a' [f64]
        [
            0.0
            1.5707963267948966
            3.141592653589793
        ]

        """
        return np.arccos(self)  # type: ignore

    def arctan(self) -> "Series":
        """
        Compute the element-wise value for Trigonometric Inverse tangent.

        Examples
        --------
        >>> import numpy as np
        >>> s = pl.Series("a", np.array((1.0, 0.0, -1)))
        >>> s.arctan()
        shape: (3,)
        Series: 'a' [f64]
        [
            0.7853981633974483
            0.0
            -7.853981633974483e-1
        ]

        """
        return np.arctan(self)  # type: ignore

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
        if return_dtype is None:
            pl_return_dtype = None
        else:
            pl_return_dtype = py_type_to_dtype(return_dtype)
        return wrap_s(self._s.apply_lambda(func, pl_return_dtype))

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

    def shift_and_fill(
        self, periods: int, fill_value: Union[int, "pli.Expr"]
    ) -> "Series":
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
            pli.col(self.name).shift_and_fill(periods, fill_value)
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
        weights: Optional[List[float]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> "Series":
        """
        apply a rolling min (moving min) over the values in this array.
        A window of length `window_size` will traverse the array. The values that fill this window
        will (optionally) be multiplied with the weights given by the `weight` vector. The resulting
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
        Series: 'a' [i64]
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
        weights: Optional[List[float]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> "Series":
        """
        Apply a rolling max (moving max) over the values in this array.
        A window of length `window_size` will traverse the array. The values that fill this window
        will (optionally) be multiplied with the weights given by the `weight` vector. The resulting
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
        Series: 'a' [i64]
        [
            null
            200
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
        weights: Optional[List[float]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> "Series":
        """
        Apply a rolling mean (moving mean) over the values in this array.
        A window of length `window_size` will traverse the array. The values that fill this window
        will (optionally) be multiplied with the weights given by the `weight` vector. The resulting
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
        Series: 'a' [f64]
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
        weights: Optional[List[float]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> "Series":
        """
        Apply a rolling sum (moving sum) over the values in this array.
        A window of length `window_size` will traverse the array. The values that fill this window
        will (optionally) be multiplied with the weights given by the `weight` vector. The resulting
        values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
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
        Series: 'a' [i64]
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
        weights: Optional[List[float]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> "Series":
        """
        Compute a rolling std dev

        A window of length `window_size` will traverse the array. The values that fill this window
        will (optionally) be multiplied with the weights given by the `weight` vector. The resulting
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
        weights: Optional[List[float]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> "Series":
        """
        Compute a rolling variance.

        A window of length `window_size` will traverse the array. The values that fill this window
        will (optionally) be multiplied with the weights given by the `weight` vector. The resulting
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
        self,
        function: Callable[["pli.Series"], Any],
        window_size: int,
        weights: Optional[List[float]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> "pli.Series":
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
        function
            Aggregation function
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
        >>> s = pl.Series("A", [1.0, 2.0, 9.0, 2.0, 13.0])
        >>> s.rolling_apply(function=lambda s: s.std(), window_size=3)
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
        if min_periods is None:
            min_periods = window_size
        return self.to_frame().select(
            pli.col(self.name).rolling_apply(
                function, window_size, weights, min_periods, center
            )
        )[self.name]

    def rolling_median(
        self,
        window_size: int,
        weights: Optional[List[float]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> "Series":
        """
        Compute a rolling median

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

        return self.to_frame().select(
            pli.col(self.name).rolling_median(window_size, weights, min_periods, center)
        )[self.name]

    def rolling_quantile(
        self,
        quantile: float,
        interpolation: str = "nearest",
        window_size: int = 2,
        weights: Optional[List[float]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> "Series":
        """
        Compute a rolling quantile

        Parameters
        ----------
        quantile
            quantile to compute
        interpolation
            interpolation type, options: ['nearest', 'higher', 'lower', 'midpoint', 'linear']
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

        return self.to_frame().select(
            pli.col(self.name).rolling_quantile(
                quantile, interpolation, window_size, weights, min_periods, center
            )
        )[self.name]

    def rolling_skew(self, window_size: int, bias: bool = True) -> "Series":
        """
        Compute a rolling skew

        Parameters
        ----------
        window_size
            Size of the rolling window
        bias
            If False, then the calculations are corrected for statistical bias.
        """
        return self.to_frame().select(
            pli.col(self.name).rolling_skew(window_size, bias)
        )[self.name]

    def sample(
        self,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        with_replacement: bool = False,
        seed: int = 0,
    ) -> "Series":
        """
        Sample from this Series by setting either `n` or `frac`.

        Parameters
        ----------
        n
            Number of samples < self.len().
        frac
            Fraction between 0.0 and 1.0 .
        with_replacement
            sample with replacement.
        seed
            Initialization seed

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4, 5])
        >>> s.sample(2)  # doctest: +IGNORE_RESULT
        shape: (2,)
        Series: 'a' [i64]
        [
            1
            5
        ]

        """
        if n is not None:
            return wrap_s(self._s.sample_n(n, with_replacement, seed))
        return wrap_s(self._s.sample_frac(frac, with_replacement, seed))

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

    @overload
    def shrink_to_fit(self, in_place: Literal[False] = ...) -> "Series":
        ...

    @overload
    def shrink_to_fit(self, in_place: Literal[True]) -> None:
        ...

    @overload
    def shrink_to_fit(self, in_place: bool = False) -> Optional["Series"]:
        ...

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

    def hash(self, k0: int = 0, k1: int = 1, k2: int = 2, k3: int = 3) -> "pli.Series":
        """
        Hash the Series.

        The hash value is of type `UInt64`

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.hash(k0=42)  # doctest: +IGNORE_RESULT
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
        return wrap_s(self._s.abs())

    def rank(self, method: str = "average", reverse: bool = False) -> "Series":
        """
        Assign ranks to data, dealing with ties appropriately.

        Parameters
        ----------
        method
            {'average', 'min', 'max', 'dense', 'ordinal', 'random'}, optional
            The method used to assign ranks to tied elements.
            The following methods are available (default is 'average'):
            - 'average': The average of the ranks that would have been assigned to
            all the tied values is assigned to each value.
            - 'min': The minimum of the ranks that would have been assigned to all
            the tied values is assigned to each value.  (This is also
            referred to as "competition" ranking.)
            - 'max': The maximum of the ranks that would have been assigned to all
            the tied values is assigned to each value.
            - 'dense': Like 'min', but the rank of the next highest element is
            assigned the rank immediately after those assigned to the tied
            elements.
            - 'ordinal': All values are given a distinct rank, corresponding to
            the order that the values occur in `a`.
            - 'random': Like 'ordinal', but the rank for ties is not dependent
            on the order that the values occur in `a`.
        reverse
            reverse the operation
        """
        return wrap_s(self._s.rank(method, reverse))

    def diff(self, n: int = 1, null_behavior: str = "ignore") -> "Series":
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

    def pct_change(self, n: int = 1) -> "Series":
        """
        Percentage change (as fraction) between current element and most-recent
        non-null element at least n period(s) before the current element.

        Computes the change from the previous row by default.

        Parameters
        ----------
        n
            periods to shift for forming percent change.

        >>> pl.Series(range(10)).pct_change()
        shape: (10,)
        Series: '' [f64]
        [
            null
            inf
            1
            0.5
            0.3333333333333333
            0.25
            0.2
            0.16666666666666666
            0.14285714285714285
            0.125
        ]

        >>> pl.Series([1, 2, 4, 8, 16, 32, 64, 128, 256, 512]).pct_change(2)
        shape: (10,)
        Series: '' [f64]
        [
            null
            null
            3
            3
            3
            3
            3
            3
            3
            3
        ]
        """
        return self.to_frame().select(pli.col(self.name).pct_change(n)).to_series()

    def skew(self, bias: bool = True) -> Optional[float]:
        r"""Compute the sample skewness of a data set.
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

        .. math:: g_1=\frac{m_3}{m_2^{3/2}}

        where

        .. math:: m_i=\frac{1}{N}\sum_{n=1}^N(x[n]-\bar{x})^i

        is the biased sample :math:`i\texttt{th}` central moment, and
        :math:`\bar{x}` is
        the sample mean.  If ``bias`` is False, the calculations are
        corrected for bias and the value computed is the adjusted
        Fisher-Pearson standardized moment coefficient, i.e.

        .. math::  G_1=\frac{k_3}{k_2^{3/2}}=\frac{\sqrt{N(N-1)}}{N-2}\frac{m_3}{m_2^{3/2}}

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
        Clip (limit) the values in an array to any value that fits in 64 floating poitns range.

        Only works for the following dtypes: {Int32, Int64, Float32, Float64, UInt32}.

        If you want to clip other dtypes, consider writing a when -> then -> otherwise expression

        Parameters
        ----------
        min_val, max_val
            Minimum and maximum value.
        """
        return self.to_frame().select(pli.col(self.name).clip(min_val, max_val))[
            self.name
        ]

    def str_concat(self, delimiter: str = "-") -> "Series":
        """
        Vertically concat the values in the Series to a single string value.

        Returns
        -------
        Series of dtype Utf8

        Examples
        --------
        >>> pl.Series([1, None, 2]).str_concat("-")[0]
        '1-null-2'

        """
        return self.to_frame().select(pli.col(self.name).str_concat(delimiter))[
            self.name
        ]

    def reshape(self, dims: Tuple[int, ...]) -> "Series":
        """
        Reshape this Series to a flat series, shape: (len,)
        or a List series, shape: (rows, cols)

        if a -1 is used in any of the dimensions, that dimension is inferred.

        Parameters
        ----------
        dims
            Tuple of the dimension sizes

        Returns
        -------
        Series
        """
        return wrap_s(self._s.reshape(dims))

    def shuffle(self, seed: int = 0) -> "Series":
        """
        Shuffle the contents of this Series.

        Parameters
        ----------
        seed
            Seed initialization
        """
        return wrap_s(self._s.shuffle(seed))

    def ewm_mean(
        self,
        com: Optional[float] = None,
        span: Optional[float] = None,
        half_life: Optional[float] = None,
        alpha: Optional[float] = None,
        adjust: bool = True,
        min_periods: int = 1,
    ) -> "Series":
        r"""
        Exponential moving average.

        Parameters
        ----------
        com
            Specify decay in terms of center of mass, :math:`alpha = 1/(1 + com) \;for\; com >= 0`.
        span
            Specify decay in terms of span, :math:`alpha = 2/(span + 1) \;for\; span >= 1`
        half_life
            Specify decay in terms of half-life, :math:`alpha = 1 - exp(-ln(2) / halflife) \;for\; halflife > 0`
        alpha
            Specify smoothing factor alpha directly, :math:`0 < alpha < 1`.
        adjust
            Divide by decaying adjustment factor in beginning periods to account for imbalance in relative weightings

                - When adjust = True the EW function is calculated using weights :math:`w_i = (1 - alpha)^i`
                - When adjust = False the EW function is calculated recursively.
        min_periods
            Minimum number of observations in window required to have a value (otherwise result is Null).

        """
        return (
            self.to_frame()
            .select(
                pli.col(self.name).ewm_mean(
                    com, span, half_life, alpha, adjust, min_periods
                )
            )
            .to_series()
        )

    def ewm_std(
        self,
        com: Optional[float] = None,
        span: Optional[float] = None,
        half_life: Optional[float] = None,
        alpha: Optional[float] = None,
        adjust: bool = True,
        min_periods: int = 1,
    ) -> "Series":
        r"""
        Exponential moving standard deviation.

        Parameters
        ----------
        com
            Specify decay in terms of center of mass, :math:`alpha = 1/(1 + com) \;for\; com >= 0`.
        span
            Specify decay in terms of span, :math:`alpha = 2/(span + 1) \;for\; span >= 1`
        half_life
            Specify decay in terms of half-life, :math:`alpha = 1 - exp(-ln(2) / halflife) \;for\; halflife > 0`
        alpha
            Specify smoothing factor alpha directly, :math:`0 < alpha < 1`.
        adjust
            Divide by decaying adjustment factor in beginning periods to account for imbalance in relative weightings

                - When adjust = True the EW function is calculated using weights :math:`w_i = (1 - alpha)^i`
                - When adjust = False the EW function is calculated recursively.
        min_periods
            Minimum number of observations in window required to have a value (otherwise result is Null).

        """
        return (
            self.to_frame()
            .select(
                pli.col(self.name).ewm_std(
                    com, span, half_life, alpha, adjust, min_periods
                )
            )
            .to_series()
        )

    def ewm_var(
        self,
        com: Optional[float] = None,
        span: Optional[float] = None,
        half_life: Optional[float] = None,
        alpha: Optional[float] = None,
        adjust: bool = True,
        min_periods: int = 1,
    ) -> "Series":
        r"""
        Exponential moving standard variation.

        Parameters
        ----------
        com
            Specify decay in terms of center of mass, :math:`alpha = 1/(1 + com) \;for\; com >= 0`.
        span
            Specify decay in terms of span, :math:`alpha = 2/(span + 1) \;for\; span >= 1`
        half_life
            Specify decay in terms of half-life, :math:`alpha = 1 - exp(-ln(2) / halflife) \;for\; halflife > 0`
        alpha
            Specify smoothing factor alpha directly, :math:`0 < alpha < 1`.
        adjust
            Divide by decaying adjustment factor in beginning periods to account for imbalance in relative weightings

                - When adjust = True the EW function is calculated using weights :math:`w_i = (1 - alpha)^i`
                - When adjust = False the EW function is calculated recursively.
        min_periods
            Minimum number of observations in window required to have a value (otherwise result is Null).

        """
        return (
            self.to_frame()
            .select(
                pli.col(self.name).ewm_var(
                    com, span, half_life, alpha, adjust, min_periods
                )
            )
            .to_series()
        )

    def extend(self, value: Optional[Union[int, float, str, bool]], n: int) -> "Series":
        """
        Extend the Series with given number of values.

        Parameters
        ----------
        value
            The value to extend the Series with. This value may be None to fill with nulls.
        n
            The number of values to extend.
        """
        return wrap_s(self._s.extend(value, n))

    @property
    def time_unit(self) -> Optional[str]:
        """
        Get the time unit of underlying Datetime Series as {"ns", "ms"}
        """
        return self._s.time_unit()

    # Below are the namespaces defined. Do not move these up in the definition of Series, as it confuses mypy between the
    # type annotation `str` and the namespace "str

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


class StringNameSpace:
    """
    Series.str namespace.
    """

    def __init__(self, series: "Series"):
        self._s = series._s

    def strptime(
        self,
        datatype: Union[Type[Date], Type[Datetime]],
        fmt: Optional[str] = None,
        strict: bool = True,
        exact: bool = True,
    ) -> Series:
        """
        Parse a Series of dtype Utf8 to a Date/Datetime Series.

        Parameters
        ----------
        datatype
            Date or Datetime.
        fmt
            format to use, see the following link for examples:
            https://docs.rs/chrono/latest/chrono/format/strftime/index.html

            example: "%y-%m-%d".
        strict
            raise an error if any conversion fails
        exact
            - If True, require an exact format match.
            - If False, allow the format to match anywhere in the target string.

        Returns
        -------
        A Date/ Datetime Series
        """
        s = wrap_s(self._s)
        return (
            s.to_frame()
            .select(pli.col(s.name).str.strptime(datatype, fmt, strict, exact))
            .to_series()
        )

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

    def decode(self, encoding: str, strict: bool = False) -> Series:
        """
        Decodes a value using the provided encoding

        Parameters
        ----------
        encoding
            'hex' or 'base64'
        strict
            how to handle invalid inputs
            - True: method will throw error if unable to decode a value
            - False: unhandled values will be replaced with `None`

        Examples
        --------
        >>> s = pl.Series(["666f6f", "626172", None])
        >>> s.str.decode("hex")
        shape: (3,)
        Series: '' [str]
        [
            "foo"
            "bar"
            null
        ]
        """
        if encoding == "hex":
            return wrap_s(self._s.str_hex_decode(strict))
        elif encoding == "base64":
            return wrap_s(self._s.str_base64_decode(strict))
        else:
            raise ValueError("supported encodings are 'hex' and 'base64'")

    def encode(self, encoding: str) -> Series:
        """
        Encodes a value using the provided encoding

        Parameters
        ----------
        encoding
            'hex' or 'base64'

        Returns
        -------
        Utf8 array with values encoded using provided encoding

        Examples
        --------
        >>> s = pl.Series(["foo", "bar", None])
        >>> s.str.encode("hex")
        shape: (3,)
        Series: '' [str]
        [
            "666f6f"
            "626172"
            null
        ]
        """
        if encoding == "hex":
            return wrap_s(self._s.str_hex_encode())
        elif encoding == "base64":
            return wrap_s(self._s.str_base64_encode())
        else:
            raise ValueError("supported encodings are 'hex' and 'base64'")

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

        >>> df = pl.DataFrame(
        ...     {"json_val": ['{"a":"1"}', None, '{"a":2}', '{"a":2.1}', '{"a":true}']}
        ... )
        >>> df.select(pl.col("json_val").str.json_path_match("$.a"))[:, 0]
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

    def extract(self, pattern: str, group_index: int = 1) -> Series:
        r"""
        Extract the target capture group from provided patterns.

        Parameters
        ----------
        pattern
            A valid regex pattern
        group_index
            Index of the targeted capture group.
            Group 0 mean the whole pattern, first group begin at index 1
            Default to the first capture group

        Returns
        -------
        Utf8 array. Contain null if original value is null or regex capture nothing.

        Examples
        --------

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [
        ...             "http://vote.com/ballon_dor?candidate=messi&ref=polars",
        ...             "http://vote.com/ballon_dor?candidat=jorginho&ref=polars",
        ...             "http://vote.com/ballon_dor?candidate=ronaldo&ref=polars",
        ...         ]
        ...     }
        ... )
        >>> df.select([pl.col("a").str.extract(r"candidate=(\w+)", 1)])
        shape: (3, 1)
        ┌─────────┐
        │ a       │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ messi   │
        ├╌╌╌╌╌╌╌╌╌┤
        │ null    │
        ├╌╌╌╌╌╌╌╌╌┤
        │ ronaldo │
        └─────────┘

        """
        return wrap_s(self._s.str_extract(pattern, group_index))

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

    def strip(self) -> Series:
        """
        Remove leading and trailing whitespace.
        """
        s = wrap_s(self._s)
        return s.to_frame().select(pli.col(s.name).str.strip()).to_series()

    def lstrip(self) -> Series:
        """
        Remove leading whitespace.
        """
        s = wrap_s(self._s)
        return s.to_frame().select(pli.col(s.name).str.lstrip()).to_series()

    def rstrip(self) -> Series:
        """
        Remove trailing whitespace.
        """
        s = wrap_s(self._s)
        return s.to_frame().select(pli.col(s.name).str.rstrip()).to_series()

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
        return pli.select(pli.lit(wrap_s(self._s)).arr.sum()).to_series()

    def max(self) -> Series:
        """
        Compute the max value of the arrays in the list
        """
        return pli.select(pli.lit(wrap_s(self._s)).arr.max()).to_series()

    def min(self) -> Series:
        """
        Compute the min value of the arrays in the list
        """
        return pli.select(pli.lit(wrap_s(self._s)).arr.min()).to_series()

    def mean(self) -> Series:
        """
        Compute the mean value of the arrays in the list
        """
        return pli.select(pli.lit(wrap_s(self._s)).arr.mean()).to_series()

    def sort(self, reverse: bool = False) -> Series:
        """
        Sort the arrays in the list
        """
        return pli.select(pli.lit(wrap_s(self._s)).arr.sort(reverse)).to_series()

    def reverse(self) -> Series:
        """
        Reverse the arrays in the list
        """
        return pli.select(pli.lit(wrap_s(self._s)).arr.reverse()).to_series()

    def unique(self) -> Series:
        """
        Get the unique/distinct values in the list
        """
        return pli.select(pli.lit(wrap_s(self._s)).arr.unique()).to_series()

    def concat(self, other: Union[List[Series], Series, List[Any]]) -> "Series":
        """
        Concat the arrays in a Series dtype List in linear time.

        Parameters
        ----------
        other
            Columns to concat into a List Series
        """
        s = wrap_s(self._s)
        return s.to_frame().select(pli.col(s.name).arr.concat(other)).to_series()

    def get(self, index: int) -> "Series":
        """
        Get the value by index in the sublists.
        So index `0` would return the first item of every sublist
        and index `-1` would return the last item of every sublist
        if an index is out of bounds, it will return a `None`.

        Parameters
        ----------
        index
            Index to return per sublist
        """
        return pli.select(pli.lit(wrap_s(self._s)).arr.get(index)).to_series()

    def first(self) -> "Series":
        """
        Get the first value of the sublists.
        """
        return self.get(0)

    def last(self) -> "Series":
        """
        Get the last value of the sublists.
        """
        return self.get(-1)

    def contains(self, item: Union[float, str, bool, int, date, datetime]) -> "Series":
        """
        Check if sublists contain the given item.

        Parameters
        ----------
        item
            Item that will be checked for membership

        Returns
        -------
        Boolean mask
        """
        s = pli.Series("", [item])
        s_list = wrap_s(self._s)
        out = s.is_in(s_list)
        return out.rename(s_list.name)


class DateTimeNameSpace:
    """
    Series.dt namespace.
    """

    def __init__(self, series: Series):
        self._s = series._s

    def truncate(
        self,
        every: Union[str, timedelta],
        offset: Optional[Union[str, timedelta]] = None,
    ) -> Series:
        """
        .. warning::
            This API is experimental and may change without it being considered a breaking change.

        Divide the date/ datetime range into buckets.
        Data must be sorted, if not the output does not make sense.

        The `every` and `offset` argument are created with the
        the following string language:

        1ns # 1 nanosecond
        1us # 1 microsecond
        1ms # 1 millisecond
        1s  # 1 second
        1m  # 1 minute
        1h  # 1 hour
        1d  # 1 day
        1w  # 1 week
        1mo # 1 calendar month
        1y  # 1 calendar year

        3d12h4m25s # 3 days, 12 hours, 4 minutes, and 25 seconds

        Parameters
        ----------
        every
            Every interval start and period length
        offset
            Offset the window

        Returns
        -------
        Date/Datetime series

        Examples
        --------

        >>> from datetime import timedelta, datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 2)
        >>> s = pl.date_range(start, stop, timedelta(minutes=30), name="dates")
        >>> s
        shape: (49,)
        Series: 'dates' [datetime[ns]]
        [
            2001-01-01 00:00:00
            2001-01-01 00:30:00
            2001-01-01 01:00:00
            2001-01-01 01:30:00
            2001-01-01 02:00:00
            2001-01-01 02:30:00
            2001-01-01 03:00:00
            2001-01-01 03:30:00
            2001-01-01 04:00:00
            2001-01-01 04:30:00
            2001-01-01 05:00:00
            2001-01-01 05:30:00
            ...
            2001-01-01 18:30:00
            2001-01-01 19:00:00
            2001-01-01 19:30:00
            2001-01-01 20:00:00
            2001-01-01 20:30:00
            2001-01-01 21:00:00
            2001-01-01 21:30:00
            2001-01-01 22:00:00
            2001-01-01 22:30:00
            2001-01-01 23:00:00
            2001-01-01 23:30:00
            2001-01-02 00:00:00
        ]
        >>> s.dt.truncate("1h")
        shape: (49,)
        Series: 'dates' [datetime[ns]]
        [
            2001-01-01 00:00:00
            2001-01-01 00:00:00
            2001-01-01 01:00:00
            2001-01-01 01:00:00
            2001-01-01 02:00:00
            2001-01-01 02:00:00
            2001-01-01 03:00:00
            2001-01-01 03:00:00
            2001-01-01 04:00:00
            2001-01-01 04:00:00
            2001-01-01 05:00:00
            2001-01-01 05:00:00
            ...
            2001-01-01 18:00:00
            2001-01-01 19:00:00
            2001-01-01 19:00:00
            2001-01-01 20:00:00
            2001-01-01 20:00:00
            2001-01-01 21:00:00
            2001-01-01 21:00:00
            2001-01-01 22:00:00
            2001-01-01 22:00:00
            2001-01-01 23:00:00
            2001-01-01 23:00:00
            2001-01-02 00:00:00
        ]
        >>> assert s.dt.truncate("1h") == s.dt.truncate(timedelta(hours=1))

        """
        return pli.select(
            pli.lit(wrap_s(self._s)).dt.truncate(every, offset)
        ).to_series()

    def __getitem__(self, item: int) -> Union[date, datetime]:
        s = wrap_s(self._s)
        return s[item]

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
        return (self.timestamp() / 1000).apply(
            lambda ts: datetime.utcfromtimestamp(ts), Object
        )

    def min(self) -> Union[date, datetime]:
        """
        Return minimum as python DateTime
        """
        s = wrap_s(self._s)
        out = s.min()
        return _to_python_datetime(out, s.dtype, s.time_unit)

    def max(self) -> Union[date, datetime]:
        """
        Return maximum as python DateTime
        """
        s = wrap_s(self._s)
        out = s.max()
        return _to_python_datetime(out, s.dtype, s.time_unit)

    def median(self) -> Union[date, datetime]:
        """
        Return median as python DateTime
        """
        s = wrap_s(self._s)
        out = int(s.median())
        return _to_python_datetime(out, s.dtype, s.time_unit)

    def mean(self) -> Union[date, datetime]:
        """
        Return mean as python DateTime
        """
        s = wrap_s(self._s)
        out = int(s.mean())
        return _to_python_datetime(out, s.dtype, s.time_unit)

    def epoch_days(self) -> Series:
        """
        Get the number of days since the unix EPOCH.
        If the date is before the unix EPOCH, the number of days will be negative.

        Returns
        -------
        Days as Int32
        """
        return wrap_s(self._s).cast(Date).cast(Int32)

    def epoch_milliseconds(self) -> Series:
        """
        Get the number of milliseconds since the unix EPOCH
        If the date is before the unix EPOCH, the number of milliseconds will be negative.

        Returns
        -------
        Milliseconds as Int64
        """
        return self.timestamp()

    def epoch_seconds(self) -> Series:
        """
        Get the number of seconds since the unix EPOCH
        If the date is before the unix EPOCH, the number of seconds will be negative.

        Returns
        -------
        Milliseconds as Int64
        """
        return wrap_s(self._s.dt_epoch_seconds())

    def and_time_unit(self, tu: str) -> "Series":
        """
        Set time unit a Series of type Datetime

        Parameters
        ----------
        tu
            Time unit for the `Datetime` Series: any of {"ns", "ms"}

        """
        return wrap_s(self._s.and_time_unit(tu))

    def and_time_zone(self, tz: Optional[str]) -> "Series":
        """
        Set time zone a Series of type Datetime

        Parameters
        ----------
        tz
            Time zone for the `Datetime` Series: any of {"ns", "ms"}

        """
        return wrap_s(self._s.and_time_zone(tz))

    def days(self) -> Series:
        """
        Extract the days from a Duration type.

        Returns
        -------
        A series of dtype Int64
        """
        return pli.select(pli.lit(wrap_s(self._s)).dt.days()).to_series()

    def hours(self) -> Series:
        """
        Extract the hours from a Duration type.

        Returns
        -------
        A series of dtype Int64
        """
        return pli.select(pli.lit(wrap_s(self._s)).dt.hours()).to_series()

    def seconds(self) -> Series:
        """
        Extract the seconds from a Duration type.

        Returns
        -------
        A series of dtype Int64
        """
        return pli.select(pli.lit(wrap_s(self._s)).dt.seconds()).to_series()

    def milliseconds(self) -> Series:
        """
        Extract the milliseconds from a Duration type.

        Returns
        -------
        A series of dtype Int64
        """
        return pli.select(pli.lit(wrap_s(self._s)).dt.milliseconds()).to_series()

    def nanoseconds(self) -> Series:
        """
        Extract the nanoseconds from a Duration type.

        Returns
        -------
        A series of dtype Int64
        """
        return pli.select(pli.lit(wrap_s(self._s)).dt.nanoseconds()).to_series()


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
