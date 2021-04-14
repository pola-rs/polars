try:
    from .polars import PySeries
except ImportError:
    import warnings

    warnings.warn("binary files missing")
    __pdoc__ = {
        "wrap_s": False,
        "find_first_non_none": False,
        "out_to_dtype": False,
        "get_ffi_func": False,
        "SeriesIter": False,
    }
import numpy as np
from typing import Optional, List, Sequence, Union, Any, Callable, Tuple
from .ffi import _ptr_to_numpy
from .datatypes import (
    Utf8,
    Int64,
    UInt64,
    UInt32,
    dtypes,
    Boolean,
    Float32,
    Float64,
    DTYPE_TO_FFINAME,
    dtype_to_primitive,
    UInt8,
    dtype_to_ctype,
    DataType,
    Date32,
    Date64,
    Int32,
    Int16,
    Int8,
    UInt16,
)
from . import datatypes
from numbers import Number
import polars
import pyarrow as pa

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .frame import DataFrame


class IdentityDict(dict):
    def __missing__(self, key):
        return key


def get_ffi_func(
    name: str, dtype: str, obj: Optional["Series"] = None, default: Optional = None
):
    """
    Dynamically obtain the proper ffi function/ method.

    Parameters
    ----------
    name
        function or method name where dtype is replaced by <>
        for example
            "call_foo_<>"
    dtype
        polars dtype str
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


def _find_first_non_none(a: "List[Optional[Any]]") -> Any:
    v = a[0]
    if v is None:
        return _find_first_non_none(a[1:])
    else:
        return v


class Series:
    def __init__(
        self,
        name: str,
        values: "Union[np.array, List[Optional[Any]]]" = None,
        nullable: bool = True,
        dtype: "Optional[DataType]" = None,
    ):
        """

        Parameters
        ----------
        name
            Name of the series
        values
            Values of the series
        nullable
            If nullable.
                None values in a list will be interpreted as missing.
                NaN values in a numpy array will be interpreted as missing. Note that missing and NaNs are not the same
                in Polars
            Series creation may be faster if set to False and there are no null values.
        """
        # assume the first input were the values
        if values is None and not isinstance(name, str):
            values = name
            name = ""
        if values.__class__ == self.__class__:
            values.rename(name)
            self._s = values._s
            return

        self._s: PySeries
        # series path
        if isinstance(values, Series):
            self._from_pyseries(values)
            return
        elif isinstance(values, dict):
            raise ValueError(
                f"Constructing a Series with a dict is not supported for {values}"
            )
        elif isinstance(values, pa.Array):
            self._s = self.from_arrow(name, values)._s
            return

        # castable to numpy
        if not isinstance(values, np.ndarray) and not nullable:
            values = np.array(values)

        if dtype is not None:
            if dtype == Int8:
                self._s = PySeries.new_i8(name, values)
            elif dtype == Int16:
                self._s = PySeries.new_i16(name, values)
            elif dtype == Int32:
                self._s = PySeries.new_i32(name, values)
            elif dtype == Int64:
                self._s = PySeries.new_i64(name, values)
            elif dtype == UInt8:
                self._s = PySeries.new_u8(name, values)
            elif dtype == UInt16:
                self._s = PySeries.new_u16(name, values)
            elif dtype == UInt32:
                self._s = PySeries.new_u32(name, values)
            elif dtype == UInt64:
                self._s = PySeries.new_u64(name, values)
            elif dtype == Float32:
                self._s = PySeries.new_f32(name, values)
            elif dtype == Float64:
                self._s = PySeries.new_f64(name, values)
            elif dtype == Boolean:
                self._s = PySeries.new_bool(name, values)
            elif dtype == Utf8:
                self._s = PySeries.new_str(name, values)
            else:
                raise ValueError(
                    f"dtype {dtype} not yet supported when creating a Series"
                )
            return

        # numpy path
        if isinstance(values, np.ndarray):
            if not values.data.contiguous:
                values = np.array(values)
            if len(values.shape) > 1:
                self._s = PySeries.new_object(name, values)
                return
            dtype = values.dtype
            if dtype == np.int64:
                self._s = PySeries.new_i64(name, values)
            elif dtype == np.int32:
                self._s = PySeries.new_i32(name, values)
            elif dtype == np.int16:
                self._s = PySeries.new_i16(name, values)
            elif dtype == np.int8:
                self._s = PySeries.new_i8(name, values)
            elif dtype == np.float32:
                self._s = PySeries.new_f32(name, values, nullable)
            elif dtype == np.float64:
                self._s = PySeries.new_f64(name, values, nullable)
            elif isinstance(values[0], str):
                self._s = PySeries.new_str(name, values)
            elif dtype == np.bool:
                self._s = PySeries.new_bool(name, values)
            elif dtype == np.uint8:
                self._s = PySeries.new_u8(name, values)
            elif dtype == np.uint16:
                self._s = PySeries.new_u16(name, values)
            elif dtype == np.uint32:
                self._s = PySeries.new_u32(name, values)
            elif dtype == np.uint64:
                self._s = PySeries.new_u64(name, values)
            else:
                self._s = PySeries.new_object(name, values)
            return
        # list path
        else:
            dtype = _find_first_non_none(values)
            # order is important as booleans are instance of int in python.
            if isinstance(dtype, bool):
                self._s = PySeries.new_opt_bool(name, values)
            elif isinstance(dtype, int):
                self._s = PySeries.new_opt_i64(name, values)
            elif isinstance(dtype, float):
                self._s = PySeries.new_opt_f64(name, values)
            elif isinstance(dtype, str):
                self._s = PySeries.new_str(name, values)
            # make list array
            elif isinstance(dtype, (list, tuple)):
                value_dtype = _find_first_non_none(dtype)

                # we can expect a failure if we pass `[[12], "foo", 9]`
                # in that case we catch the exception and create an object type
                try:
                    if isinstance(value_dtype, bool):
                        arrow_array = pa.array(values, pa.large_list(pa.bool_()))
                    elif isinstance(value_dtype, int):
                        arrow_array = pa.array(values, pa.large_list(pa.int64()))
                    elif isinstance(value_dtype, float):
                        arrow_array = pa.array(values, pa.large_list(pa.float64()))
                    elif isinstance(value_dtype, str):
                        arrow_array = pa.array(values, pa.large_list(pa.large_utf8()))
                    else:
                        self._s = PySeries.new_object(name, values)
                        return
                    self._s = Series.from_arrow(name, arrow_array)._s

                except pa.lib.ArrowInvalid:
                    self._s = PySeries.new_object(name, values)
            else:
                self._s = PySeries.new_object(name, values)

    @staticmethod
    def _from_pyseries(s: "PySeries") -> "Series":
        self = Series.__new__(Series)
        self._s = s
        return self

    @staticmethod
    def _repeat(name: str, val: str, n: int) -> "Series":
        """
        Only used for strings.
        """
        return Series._from_pyseries(PySeries.repeat(name, val, n))

    @staticmethod
    def from_arrow(name: str, array: "pa.Array"):
        """
        Create a Series from an arrow array.

        Parameters
        ----------
        name
            name of the Series.
        array
            Arrow array.
        """
        return Series._from_pyseries(PySeries.from_arrow(name, array))

    def inner(self) -> "PySeries":
        return self._s

    def __str__(self) -> str:
        return self._s.as_str()

    def __repr__(self) -> str:
        return self.__str__()

    def __and__(self, other):
        return wrap_s(self._s.bitand(other._s))

    def __or__(self, other):
        return wrap_s(self._s.bitor(other._s))

    def __eq__(self, other):
        if isinstance(other, Sequence) and not isinstance(other, str):
            other = Series("", other, nullable=True)
        if isinstance(other, Series):
            return Series._from_pyseries(self._s.eq(other._s))
        f = get_ffi_func("eq_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __ne__(self, other):
        if isinstance(other, Sequence) and not isinstance(other, str):
            other = Series("", other, nullable=True)
        if isinstance(other, Series):
            return Series._from_pyseries(self._s.neq(other._s))
        f = get_ffi_func("neq_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __gt__(self, other):
        if isinstance(other, Sequence) and not isinstance(other, str):
            other = Series("", other, nullable=True)
        if isinstance(other, Series):
            return Series._from_pyseries(self._s.gt(other._s))
        f = get_ffi_func("gt_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __lt__(self, other):
        if isinstance(other, Sequence) and not isinstance(other, str):
            other = Series("", other, nullable=True)
        if isinstance(other, Series):
            return Series._from_pyseries(self._s.lt(other._s))
        f = get_ffi_func("lt_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __ge__(self, other) -> "Series":
        if isinstance(other, Sequence) and not isinstance(other, str):
            other = Series("", other, nullable=True)
        if isinstance(other, Series):
            return Series._from_pyseries(self._s.gt_eq(other._s))
        f = get_ffi_func("gt_eq_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __le__(self, other) -> "Series":
        if isinstance(other, Sequence) and not isinstance(other, str):
            other = Series("", other, nullable=True)
        if isinstance(other, Series):
            return Series._from_pyseries(self._s.lt_eq(other._s))
        f = get_ffi_func("lt_eq_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __add__(self, other) -> "Series":
        if isinstance(other, str):
            other = Series("", [other])
        if isinstance(other, Series):
            return wrap_s(self._s.add(other._s))
        dtype = dtype_to_primitive(self.dtype)
        f = get_ffi_func("add_<>", dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __sub__(self, other) -> "Series":
        if isinstance(other, Series):
            return Series._from_pyseries(self._s.sub(other._s))
        dtype = dtype_to_primitive(self.dtype)
        f = get_ffi_func("sub_<>", dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __truediv__(self, other) -> "Series":
        primitive = dtype_to_primitive(self.dtype)
        if self.dtype != primitive:
            return self.__floordiv__(other)

        if not self.is_float():
            out_dtype = Float64
        else:
            out_dtype = self.dtype
        return np.true_divide(self, other, dtype=out_dtype)

    def __floordiv__(self, other) -> "Series":
        if isinstance(other, Series):
            return Series._from_pyseries(self._s.div(other._s))
        dtype = dtype_to_primitive(self.dtype)
        f = get_ffi_func("div_<>", dtype, self._s)
        return wrap_s(f(other))

    def __mul__(self, other) -> "Series":
        if isinstance(other, Series):
            return Series._from_pyseries(self._s.mul(other._s))
        dtype = dtype_to_primitive(self.dtype)
        f = get_ffi_func("mul_<>", dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __radd__(self, other):
        if isinstance(other, Series):
            return Series._from_pyseries(self._s.add(other._s))
        dtype = dtype_to_primitive(self.dtype)
        f = get_ffi_func("add_<>_rhs", dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __rsub__(self, other):
        if isinstance(other, Series):
            return Series._from_pyseries(other._s.sub(self._s))
        dtype = dtype_to_primitive(self.dtype)
        f = get_ffi_func("sub_<>_rhs", dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __invert__(self):
        if self.dtype == Boolean:
            return wrap_s(self._s._not())
        return NotImplemented

    def __rtruediv__(self, other):

        primitive = dtype_to_primitive(self.dtype)
        if self.dtype != primitive:
            self.__rfloordiv__(other)

        if not self.is_float():
            out_dtype = Float64
        else:
            out_dtype = DTYPE_TO_FFINAME[self.dtype]
        return np.true_divide(other, self, dtype=out_dtype)

    def __rfloordiv__(self, other):
        if isinstance(other, Series):
            return Series._from_pyseries(other._s.div(self._s))
        dtype = dtype_to_primitive(self.dtype)
        f = get_ffi_func("div_<>_rhs", dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __rmul__(self, other):
        if isinstance(other, Series):
            return Series._from_pyseries(self._s.mul(other._s))
        dtype = dtype_to_primitive(self.dtype)
        f = get_ffi_func("mul_<>", dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __getitem__(self, item):
        if isinstance(item, int):
            if item >= self.len():
                raise IndexError
        # assume it is boolean mask
        if isinstance(item, Series):
            return Series._from_pyseries(self._s.filter(item._s))
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
        if self.dtype == polars.datatypes.List:
            return wrap_s(out)
        return out

    def __setitem__(self, key, value):
        if isinstance(key, Series):
            if key.dtype == Boolean:
                self._s = self.set(key, value)._s
            elif key.dtype == UInt64:
                self._s = self.set_at_idx(key, value)._s
            elif key.dtype == UInt32:
                self._s = self.set_at_idx(key.cast_u64(), value)._s
        # TODO: implement for these types without casting to series
        elif isinstance(key, (np.ndarray, list, tuple)):
            s = wrap_s(PySeries.new_u64("", np.array(key, np.uint64)))
            self.__setitem__(s, value)
        elif isinstance(key, int):
            self.__setitem__([key], value)
        else:
            raise ValueError(f'cannot use "{key}" for indexing')

    def drop_nulls(self) -> "Series":
        """
        Create a new Series that copies data from this Series without null values.
        """
        return wrap_s(self._s.drop_nulls())

    def to_frame(self) -> "DataFrame":
        """
        Cast this Series to a DataFrame
        """
        # implementation is in .frame due to circular imports
        pass

    @property
    def dtype(self):
        """
        Get the data type of this Series
        """
        return dtypes[self._s.dtype()]

    def sum(self):
        """
        Reduce this Series to the sum value.
        """
        if self.dtype == Boolean:
            return self._s.sum_u32()
        if self.dtype == UInt8:
            return self.cast(UInt64).sum()
        f = get_ffi_func("sum_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return f()

    def mean(self):
        """
        Reduce this Series to the mean value.
        """
        return self._s.mean()

    def min(self):
        """
        Get the minimal value in this Series
        """
        if self.dtype == Boolean:
            return self._s.min_u32()
        f = get_ffi_func("min_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return f()

    def max(self):
        """
        Get the maximum value in this Series
        """
        if self.dtype == Boolean:
            return self._s.max_u32()
        f = get_ffi_func("max_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return f()

    def std(self) -> float:
        """
        Get standard deviation of this Series
        """
        return np.std(self.drop_nulls().view())

    def var(self) -> float:
        """
        Get variance of this Series
        """
        return np.var(self.drop_nulls().view())

    def median(self) -> float:
        """
        Get median of this Series
        """
        return self._s.median()

    def quantile(self, quantile: float) -> float:
        """
        Get quantile value of this Series
        """
        return self._s.quantile(quantile)

    def to_dummies(self) -> "DataFrame":
        """
        Get dummy variables
        """
        return polars.frame.wrap_df(self._s.to_dummies())

    def value_counts(self) -> "DataFrame":
        """
        Count the unique values in a Series
        """
        return polars.frame.wrap_df(self._s.value_counts())

    @property
    def name(self):
        """
        Get the name of this Series
        """
        return self._s.name()

    def rename(self, name: str):
        """
        Rename this Series.

        Parameters
        ----------
        name
            New name
        """
        self._s.rename(name)

    def chunk_lengths(self) -> "List[int]":
        return self._s.chunk_lengths()

    def n_chunks(self) -> int:
        """
        Get the number of chunks that this Series contains.
        """
        return self._s.n_chunks()

    def cum_sum(self, reverse: bool):
        """
        Get an array with the cumulative sum computed at every element

        Parameters
        ----------
        reverse
            reverse the operation
        """
        return self._s.cum_sum(reverse)

    def cum_min(self, reverse: bool):
        """
        Get an array with the cumulative min computed at every element

        Parameters
        ----------
        reverse
            reverse the operation
        """
        return self._s.cum_min(reverse)

    def cum_max(self, reverse: bool):
        """
        Get an array with the cumulative max computed at every element

        Parameters
        ----------
        reverse
            reverse the operation
        """
        return self._s.cum_max(reverse)

    def limit(self, num_elements: int) -> "Series":
        """
        Take n elements from this Series.

        Parameters
        ----------
        num_elements
            Amount of elements to take.
        """
        return Series._from_pyseries(self._s.limit(num_elements))

    def slice(self, offset: int, length: int) -> "Series":
        """
        Get a slice of this Series

        Parameters
        ----------
        offset
            Offset index.
        length
            Length of the slice.
        """
        return Series._from_pyseries(self._s.slice(offset, length))

    def append(self, other: "Series"):
        """
        Append a Series to this one.

        Parameters
        ----------
        other
            Series to append
        """
        self._s.append(other._s)

    def filter(self, filter: "Series") -> "Series":
        """
        Filter elements by a boolean mask

        Parameters
        ----------
        filter
            Boolean mask
        """
        return Series._from_pyseries(self._s.filter(filter._s))

    def head(self, length: Optional[int] = None) -> "Series":
        """
        Get first N elements as Series

        Parameters
        ----------
        length
            Length of the head
        """
        return Series._from_pyseries(self._s.head(length))

    def tail(self, length: Optional[int] = None) -> "Series":
        """
        Get last N elements as Series

        Parameters
        ----------
        length
            Length of the tail
        """
        return Series._from_pyseries(self._s.tail(length))

    def take_every(self, n: int) -> "Series":
        """
        Take every nth value in the Series and return as new Series.
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
            Reverse sort
        """
        if in_place:
            self._s.sort_in_place(reverse)
        else:
            return wrap_s(self._s.sort(reverse))

    def argsort(self, reverse: bool = False) -> "Series":
        """
        ..deprecate::

        Index location of the sorted variant of this Series.

        Returns
        -------
        indexes
            Indexes that can be used to sort this array.
        """
        return self._s.argsort(reverse)

    def arg_sort(self, reverse: bool = False) -> "Series":
        """
        Index location of the sorted variant of this Series.

        Returns
        -------
        indexes
            Indexes that can be used to sort this array.
        """
        return self._s.argsort(reverse)

    def arg_unique(self) -> "Series":
        """
        Get unique index as Series.
        """
        return self._s.arg_unique()

    def arg_min(self) -> Optional[int]:
        """
        Get the index of the minimal value
        """
        return self._s.arg_min()

    def arg_max(self) -> Optional[int]:
        """
        Get the index of the maxima value
        """
        return self._s.arg_max()

    def unique(self) -> "Series":
        """
        Get unique elements in series.
        """
        return wrap_s(self._s.unique())

    def take(self, indices: "Union[np.ndarray, List[int]]") -> "Series":
        """
        Take values by index.

        Parameters
        ----------
        indices
            Index location used for selection.
        """
        if isinstance(indices, list):
            indices = np.array(indices)
        return Series._from_pyseries(self._s.take(indices))

    def null_count(self) -> int:
        """
        Count the null values in this Series
        """
        return self._s.null_count()

    def is_null(self) -> "Series":
        """
        Get mask of null values

        Returns
        -------
        Boolean Series
        """
        return Series._from_pyseries(self._s.is_null())

    def is_not_null(self) -> "Series":
        """
        Get mask of non null values

        Returns
        -------
        Boolean Series
        """
        return Series._from_pyseries(self._s.is_not_null())

    def is_finite(self) -> "Series":
        """
        Get mask of finite values if Series dtype is Float

        Returns
        -------
        Boolean Series
        """
        return Series._from_pyseries(self._s.is_finite())

    def is_infinite(self) -> "Series":
        """
        Get mask of infinite values if Series dtype is Float

        Returns
        -------
        Boolean Series
        """
        return Series._from_pyseries(self._s.is_infinite())

    def is_nan(self) -> "Series":
        """
        Get mask of NaN values if Series dtype is Float

        Returns
        -------
        Boolean Series
        """
        return Series._from_pyseries(self._s.is_nan())

    def is_not_nan(self) -> "Series":
        """
        Get negated mask of NaN values if Series dtype is_not Float

        Returns
        -------
        Boolean Series
        """
        return Series._from_pyseries(self._s.is_not_nan())

    def is_in(self, list_series: "Series") -> "Series":
        """
        Check if the values in this Series are in the a member of the values in the Series of dtype List
        Returns
        -------
        Boolean Series
        """
        return wrap_s(self._s.is_in(list_series._s))

    def arg_true(self) -> "Series":
        """
        Get index values where Boolean Series evaluate True

        Returns
        -------
        UInt32 Series
        """
        return Series._from_pyseries(self._s.arg_true())

    def is_unique(self) -> "Series":
        """
        Get mask of all unique values

        Returns
        -------
        Boolean Series
        """
        return wrap_s(self._s.is_unique())

    def is_duplicated(self) -> "Series":
        """
        Get mask of all duplicated values

        Returns
        -------
        Boolean Series
        """
        return wrap_s(self._s.is_duplicated())

    def explode(self) -> "Series":
        """
        Explode a list or utf8 Series. This means that every item is expanded to a new row.

        Returns
        -------
        Exploded Series of same dtype
        """
        return wrap_s(self._s.explode())

    def series_equal(self, other: "Series", null_equal: bool = False) -> bool:
        """
        Check if series equal with another Series.

        Parameters
        ----------
        other
            Series to compare with.
        null_equal
            Consider null values as equal.
        """
        return self._s.series_equal(other._s, null_equal)

    def len(self) -> int:
        """
        Length of this Series
        """
        return self._s.len()

    @property
    def shape(self) -> Tuple[int]:
        """
        Shape of this Series
        """
        return (self._s.len(),)

    def __len__(self):
        return self.len()

    def cast(self, data_type="DataType"):
        if data_type == int:
            data_type = Int64
        elif data_type == str:
            data_type = Utf8
        elif data_type == float:
            data_type = Float64
        f = get_ffi_func("cast_<>", data_type, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f())

    def to_list(self) -> "List[Optional[Any]]":
        """
        Convert this Series to a Python List. This operation clones data.
        """
        if self.dtype != datatypes.Object:
            return self.to_arrow().to_pylist()
        return self._s.to_list()

    def __iter__(self):
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
        if not in_place:
            return wrap_s(opt_s)

    def is_numeric(self) -> bool:
        """
        Check if this Series datatype is numeric.
        """
        return self.dtype not in (Utf8, Boolean, List)

    def is_float(self) -> bool:
        """
        Check if this Series has floating point numbers
        """
        return self.dtype in (Float32, Float64)

    def view(self, ignore_nulls: bool = False) -> np.ndarray:
        """
        Get a view into this Series data with a numpy array. This operation doesn't clone data, but does not include
        missing values. Don't use this unless you know what you are doing.

        # Safety.

        This function can lead to undefined behavior in the following cases:

        ```python
        # returns a view to a piece of memory that is already dropped.
        pl.Series([1, 3, 5]).sort().view()

        # Sums invalid data that is missing.
        pl.Series([1, 2, None], nullable=True).view().sum()
        ```
        """
        if not ignore_nulls:
            assert self.null_count() == 0

        ptr_type = dtype_to_ctype(self.dtype)
        ptr = self._s.as_single_ptr()
        array = _ptr_to_numpy(ptr, self.len(), ptr_type)
        array.setflags(write=False)
        return array

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if self._s.n_chunks() > 0:
            self._s.rechunk(in_place=True)

        if method == "__call__":
            args = []
            for arg in inputs:
                if isinstance(arg, Number):
                    args.append(arg)
                elif isinstance(arg, self.__class__):
                    args.append(arg.view(ignore_nulls=True))
                else:
                    return NotImplemented

            if "dtype" in kwargs:
                dtype = kwargs.pop("dtype")
            else:
                dtype = self.dtype

            f = get_ffi_func("apply_ufunc_<>", dtype, self._s)
            series = f(lambda out: ufunc(*args, out=out, **kwargs))
            return wrap_s(series)
        else:
            return NotImplemented

    def to_numpy(self, *args, zero_copy_only=False, **kwargs) -> np.ndarray:
        """
        Convert this Series to numpy. This operation clones data but is completely safe.

        If you want a zero-copy view and know what you are doing, use `.view()`.

        Parameters
        ----------
        args
            args will be sent to pyarrow.Array.to_numpy
        zero_copy_only
            If True, an exception will be raised if the conversion to a numpy
            array would require copying the underlying data (e.g. in presence
            of nulls, or for non-primitive types).
        kwargs
            kwargs will be sent to pyarrow.Array.to_numpy
        """
        return self.to_arrow().to_numpy(*args, zero_copy_only=zero_copy_only, **kwargs)

    def to_arrow(self) -> pa.Array:
        """
        Get the underlying arrow array. If the Series contains only a single chunk
        this operation is zero copy.
        """
        return self._s.to_arrow()

    def set(self, filter: "Series", value: Union[int, float]) -> "Series":
        """
        Set masked values.

        Parameters
        ----------
        filter
            Boolean mask
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
            replacement values

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
        Cheap deep clones
        """
        return wrap_s(self._s.clone())

    def fill_none(self, strategy: str) -> "Series":
        """
        Fill null values with a fill strategy.

        Parameters
        ----------
        strategy
               * "backward"
               * "forward"
               * "min"
               * "max"
               * "mean"
        """
        return wrap_s(self._s.fill_none(strategy))

    def apply(
        self,
        func: "Union[Callable[['Any'], 'Any'], Callable[['Any'], 'Any']]",
        dtype_out: "Optional['DataType']" = None,
    ):
        """
        Apply a function over elements in this Series and return a new Series.

        If the function returns another datatype, the dtype_out arg should be set, otherwise the method will fail.

        Parameters
        ----------
        func
            function or lambda.
        dtype_out
            Output datatype. If none given the same datatype as this Series will be used.

        Returns
        -------
        Series
        """
        if dtype_out == str:
            dtype_out = Utf8
        elif dtype_out == int:
            dtype_out = Int64
        elif dtype_out == float:
            dtype_out = Float64
        elif dtype_out == bool:
            dtype_out = Boolean

        return wrap_s(self._s.apply_lambda(func, dtype_out))

    def shift(self, periods: int) -> "Series":
        """
        Shift the values by a given period and fill the parts that will be empty due to this operation
        with `Nones`.

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).
        """
        return wrap_s(self._s.shift(periods))

    def zip_with(self, mask: "Series", other: "Series") -> "Series":
        """
        Where mask evaluates true take values from self. Where mask evaluates false, take values from other.

        Parameters
        ----------
        mask
            Boolean Series
        other
            Series of same type

        Returns
        -------
        New Series
        """
        return wrap_s(self._s.zip_with(mask._s, other._s))

    def str_lengths(self) -> "Series":
        """
        Get length of the string values in the Series.

        Returns
        -------
        Series[u32]
        """
        return wrap_s(self._s.str_lengths())

    def str_contains(self, pattern: str) -> "Series":
        """
        Check if strings in Series contain regex pattern

        Parameters
        ----------
        pattern
            A valid regex pattern

        Returns
        -------
        Boolean mask
        """
        return wrap_s(self._s.str_contains(pattern))

    def str_replace(self, pattern: str, value: str) -> "Series":
        """
        Replace first regex math with a string value

        Parameters
        ----------
        pattern
            A valid regex pattern
        value
            Substring to replace
        """
        return wrap_s(self._s.str_replace(pattern, value))

    def str_replace_all(self, pattern: str, value: str) -> "Series":
        """
        Replace all regex matches with a string value

        Parameters
        ----------
        pattern
            A valid regex pattern
        value
            Substring to replace
        """
        return wrap_s(self._s.str_replace_all(pattern, value))

    def str_to_lowercase(self) -> "Series":
        """
        Modify the strings to their lowercase equivalent
        """
        return wrap_s(self._s.str_to_lowercase())

    def str_to_uppercase(self) -> "Series":
        """
        Modify the strings to their uppercase equivalent
        """
        return wrap_s(self._s.str_to_uppercase())

    def str_rstrip(self) -> "Series":
        """
        Remove trailing whitespace
        """
        return self.str_replace(r"[ \t]+$", "")

    def str_lstrip(self) -> "Series":
        """
        Remove leading whitespace
        """
        return self.str_replace(r"^\s*", "")

    def as_duration(self) -> "Series":
        """
        .. deprecated::
        If Series is a date32 or a date64 it can be turned into a duration.
        """
        return wrap_s(self._s.as_duration())

    def str_parse_date(
        self, datatype: "DataType", fmt: Optional[str] = None
    ) -> "Series":
        """
        Parse a Series of dtype Utf8 to a Date32/Date64 Series.

        Parameters
        ----------
        datatype
            polars.Date32 or polars.Date64
        fmt
            formatting syntax. [Read more](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html)

        Returns
        -------
        A Date32/ Date64 Series
        """
        if datatype == Date32:
            return wrap_s(self._s.str_parse_date32(fmt))
        if datatype == Date64:
            return wrap_s(self._s.str_parse_date64(fmt))
        raise NotImplementedError

    def rolling_min(
        self,
        window_size: int,
        weight: "Optional[List[float]]" = None,
        ignore_null: bool = True,
        min_periods: "Optional[int]" = None,
    ) -> "Series":
        """
        apply a rolling min (moving min) over the values in this array.
        a window of length `window_size` will traverse the array. the values that fill this window
        will (optionally) be multiplied with the weights given by the `weight` vector. the resultingParameters
        values will be aggregated to their sum.                                                     ----------

        window_size
            The length of the window
        weight
            An optional slice with the same length of the window that will be multiplied
            elementwise with the values in the window.
        ignore_null
            Toggle behavior of aggregation regarding null values in the window.
              `True` -> Null values will be ignored.
              `False` -> Any Null in the window leads to a Null in the aggregation result.
        min_periods
            The number of values in the window that should be non-null before computing a result.
            If None it will be set equal to window size
        """
        if min_periods is None:
            min_periods = window_size
        return wrap_s(
            self._s.rolling_min(window_size, weight, ignore_null, min_periods)
        )

    def rolling_max(
        self,
        window_size: int,
        weight: "Optional[List[float]]" = None,
        ignore_null: bool = True,
        min_periods: "Optional[int]" = None,
    ) -> "Series":
        """
        apply a rolling max (moving max) over the values in this array.
        a window of length `window_size` will traverse the array. the values that fill this window
        will (optionally) be multiplied with the weights given by the `weight` vector. the resultingParameters
        values will be aggregated to their sum.                                                     ----------

        window_size
            The length of the window
        weight
            An optional slice with the same length of the window that will be multiplied
            elementwise with the values in the window.
        ignore_null
            Toggle behavior of aggregation regarding null values in the window.
              `True` -> Null values will be ignored.
              `False` -> Any Null in the window leads to a Null in the aggregation result.
        min_periods
            The number of values in the window that should be non-null before computing a result.
            If None it will be set equal to window size
        """
        if min_periods is None:
            min_periods = window_size
        return wrap_s(
            self._s.rolling_max(window_size, weight, ignore_null, min_periods)
        )

    def rolling_mean(
        self,
        window_size: int,
        weight: "Optional[List[float]]" = None,
        ignore_null: bool = True,
        min_periods: "Optional[int]" = None,
    ) -> "Series":
        """
        apply a rolling mean (moving mean) over the values in this array.
        a window of length `window_size` will traverse the array. the values that fill this window
        will (optionally) be multiplied with the weights given by the `weight` vector. the resultingParameters
        values will be aggregated to their sum.                                                     ----------

        window_size
            The length of the window
        weight
            An optional slice with the same length of the window that will be multiplied
            elementwise with the values in the window.
        ignore_null
            Toggle behavior of aggregation regarding null values in the window.
              `True` -> Null values will be ignored.
              `False` -> Any Null in the window leads to a Null in the aggregation result.
        min_periods
            The number of values in the window that should be non-null before computing a result.
            If None it will be set equal to window size
        """
        if min_periods is None:
            min_periods = window_size
        return wrap_s(
            self._s.rolling_mean(window_size, weight, ignore_null, min_periods)
        )

    def rolling_sum(
        self,
        window_size: int,
        weight: "Optional[List[float]]" = None,
        ignore_null: bool = True,
        min_periods: "Optional[int]" = None,
    ) -> "Series":
        """
        apply a rolling sum (moving sum) over the values in this array.
        a window of length `window_size` will traverse the array. the values that fill this window
        will (optionally) be multiplied with the weights given by the `weight` vector. the resultingParameters
        values will be aggregated to their sum.                                                     ----------

        window_size
            The length of the window
        weight
            An optional slice with the same length of the window that will be multiplied
            elementwise with the values in the window.
        ignore_null
            Toggle behavior of aggregation regarding null values in the window.
              `True` -> Null values will be ignored.
              `False` -> Any Null in the window leads to a Null in the aggregation result.
        min_periods
            The number of values in the window that should be non-null before computing a result.
            If None it will be set equal to window size
        """
        if min_periods is None:
            min_periods = window_size
        return wrap_s(
            self._s.rolling_sum(window_size, weight, ignore_null, min_periods)
        )

    def year(self):
        """
        Extract year from underlying Date representation.
        Can be performed on Date32 and Date64

        Returns the year number in the calendar date.

        Returns
        -------
        Year as Int32
        """
        return wrap_s(self._s.year())

    def month(self):
        """
        Extract month from underlying Date representation.
        Can be performed on Date32 and Date64

        Returns the month number starting from 1.
        The return value ranges from 1 to 12.

        Returns
        -------
        Month as UInt32
        """
        return wrap_s(self._s.month())

    def week(self):
        """
        Extract the week from underlying Date representation.
        Can be performed on Date32 and Date64

        Returns the ISO week number starting from 1.
        The return value ranges from 1 to 53. (The last week of year differs by years.)

        Returns
        -------
        Week number as UInt32
        """
        return wrap_s(self._s.week())

    def weekday(self):
        """
        Extract the week day from underlying Date representation.
        Can be performed on Date32 and Date64

        Returns the weekday number where monday = 0 and sunday = 6

        Returns
        -------
        Week day as UInt32
        """
        return wrap_s(self._s.week())

    def day(self):
        """
        Extract day from underlying Date representation.
        Can be performed on Date32 and Date64

        Returns the day of month starting from 1.
        The return value ranges from 1 to 31. (The last day of month differs by months.)

        Returns
        -------
        Day as UInt32
        """
        return wrap_s(self._s.day())

    def ordinal_day(self):
        """
        Extract ordinal day from underlying Date representation.
        Can be performed on Date32 and Date64

        Returns the day of year starting from 1.
        The return value ranges from 1 to 366. (The last day of year differs by years.)

        Returns
        -------
        Day as UInt32
        """
        return wrap_s(self._s.ordinal_day())

    def hour(self):
        """
        Extract day from underlying DateTime representation.
        Can be performed on Date64

        Returns the hour number from 0 to 23.

        Returns
        -------
        Hour as UInt32
        """
        return wrap_s(self._s.hour())

    def minute(self):
        """
        Extract minutes from underlying DateTime representation.
        Can be performed on Date64

        Returns the minute number from 0 to 59.

        Returns
        -------
        Minute as UInt32
        """
        return wrap_s(self._s.minute())

    def second(self):
        """
        Extract seconds from underlying DateTime representation.
        Can be performed on Date64

        Returns the second number from 0 to 59.

        Returns
        -------
        Second as UInt32
        """
        return wrap_s(self._s.second())

    def nanosecond(self):
        """
        Extract seconds from underlying DateTime representation.
        Can be performed on Date64

        Returns the number of nanoseconds since the whole non-leap second.
        The range from 1,000,000,000 to 1,999,999,999 represents the leap second.

        Returns
        -------
        Nanosecond as UInt32
        """
        return wrap_s(self._s.nanosecond())

    def datetime_str_fmt(self, fmt):
        """
        Format date32/date64 with a formatting rule: See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).

        Returns
        -------
        Utf8 Series
        """
        return wrap_s(self._s.date_str_fmt(fmt))

    @staticmethod
    def parse_date(
        name: str, values: Sequence[str], dtype: "DataType", fmt: str
    ) -> "Series":
        """
        .. deprecated::
        """
        f = get_ffi_func("parse_<>_from_str_slice", dtype, PySeries)
        if f is None:
            return NotImplemented
        return wrap_s(f(name, values, fmt))

    def sample(
        self,
        n: "Optional[int]" = None,
        frac: "Optional[float]" = None,
        with_replacement: bool = False,
    ) -> "DataFrame":
        """
        Sample from this Series by setting either `n` or `frac`

        Parameters
        ----------
        n
            Number of samples < self.len()
        frac
            Fraction between 0.0 and 1.0
        with_replacement
            sample with replacement
        """
        if n is not None:
            return wrap_s(self._s.sample_n(n, with_replacement))
        return wrap_s(self._s.sample_frac(frac, with_replacement))

    def peak_max(self):
        """
        Get a boolean mask of the local maximum peaks.
        """
        return wrap_s(self._s.peak_max())

    def peak_min(self):
        """
        Get a boolean mask of the local minimum peaks.
        """
        return wrap_s(self._s.peak_min())


class SeriesIter:
    def __init__(self, length: int, s: "Series"):
        self.len = length
        self.i = 0
        self.s = s

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.len:
            i = self.i
            self.i += 1
            return self.s[i]
        else:
            raise StopIteration


def out_to_dtype(out: Any) -> "Union[datatypes.DataType, np.ndarray]":
    if isinstance(out, float):
        return datatypes.Float64
    if isinstance(out, int):
        return datatypes.Int64
    if isinstance(out, str):
        return datatypes.Utf8
    if isinstance(out, bool):
        return datatypes.Boolean
    if isinstance(out, Series):
        return datatypes.List
    if isinstance(out, np.ndarray):
        return np.ndarray
    raise NotImplementedError
