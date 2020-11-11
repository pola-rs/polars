from __future__ import annotations

try:
    from .pypolars import PySeries
except:
    import warnings

    warnings.warn("binary files missing")
    __pdoc__ = {
        "wrap_s": False,
        "find_first_non_none": False,
        "out_to_dtype": False,
        "get_ffi_func": False,
    }
import numpy as np
from typing import Optional, List, Sequence, Union, Any, Callable
from .ffi import _ptr_to_numpy
from .datatypes import *
from numbers import Number


class IdentityDict(dict):
    def __missing__(self, key):
        return key


def get_ffi_func(
    name: str, dtype: str, obj: Optional[Series] = None, default: Optional = None
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


def wrap_s(s: PySeries) -> Series:
    return Series._from_pyseries(s)


def find_first_non_none(a: List[Optional[Any]]) -> Any:
    v = a[0]
    if v is None:
        return find_first_non_none(a[1:])
    else:
        return v


class Series:
    def __init__(
        self,
        name: str,
        values: Union[np.array, List[Optional[Any]]],
        nullable: bool = False,
    ):
        """

        Parameters
        ----------
        name
            Name of the series
        values
            Values of the series
        nullable
            If nullable. List[Optional[Any]] will remain lists where None values will be interpreted as nulls
        """
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

        # castable to numpy
        if not isinstance(values, np.ndarray) and not nullable:
            values = np.array(values)

        # numpy path
        if isinstance(values, np.ndarray):
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
                self._s = PySeries.new_f32(name, values)
            elif dtype == np.float64:
                self._s = PySeries.new_f64(name, values)
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
                raise ValueError(f"dtype: {dtype} not known")
        # list path
        else:
            dtype = find_first_non_none(values)
            if isinstance(dtype, int):
                self._s = PySeries.new_opt_i64(name, values)
            elif isinstance(dtype, float):
                self._s = PySeries.new_opt_f64(name, values)
            elif isinstance(dtype, str):
                self._s = PySeries.new_str(name, values)
            elif isinstance(dtype, bool):
                self._s = PySeries.new_opt_bool(name, values)
            else:
                raise ValueError(f"dtype: {dtype} not known")

    @staticmethod
    def _from_pyseries(s: "PySeries") -> "Series":
        self = Series.__new__(Series)
        self._s = s
        return self

    def inner(self) -> "PySeries":
        return self._s

    def __str__(self) -> str:
        return self._s.as_str()

    def __repr__(self) -> str:
        return self.__str__()

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

    def __ge__(self, other) -> Series:
        if isinstance(other, Sequence) and not isinstance(other, str):
            other = Series("", other, nullable=True)
        if isinstance(other, Series):
            return Series._from_pyseries(self._s.gt_eq(other._s))
        f = get_ffi_func("gt_eq_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __le__(self, other) -> Series:
        if isinstance(other, Sequence) and not isinstance(other, str):
            other = Series("", other, nullable=True)
        if isinstance(other, Series):
            return Series._from_pyseries(self._s.lt_eq(other._s))
        f = get_ffi_func("lt_eq_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __add__(self, other) -> Series:
        if isinstance(other, Series):
            return Series._from_pyseries(self._s.add(other._s))
        f = get_ffi_func("add_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __sub__(self, other) -> Series:
        if isinstance(other, Series):
            return Series._from_pyseries(self._s.sub(other._s))
        f = get_ffi_func("sub_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __truediv__(self, other) -> Series:
        if not self.is_float():
            out_dtype = Float64
        else:
            out_dtype = DTYPE_TO_FFINAME[self.dtype]
        return np.true_divide(self, other, dtype=out_dtype)

    def __floordiv__(self, other) -> Series:
        return np.floor_divide(self, other)

    def __mul__(self, other) -> Series:
        if isinstance(other, Series):
            return Series._from_pyseries(self._s.mul(other._s))
        f = get_ffi_func("mul_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __radd__(self, other):
        if isinstance(other, Series):
            return Series._from_pyseries(self._s.add(other._s))
        f = get_ffi_func("add_<>_rhs", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __rsub__(self, other):
        if isinstance(other, Series):
            return Series._from_pyseries(other._s.sub(self._s))
        f = get_ffi_func("sub_<>_rhs", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __invert__(self):
        if self.dtype == Bool:
            return wrap_s(self._s._not())
        return NotImplemented

    def __rtruediv__(self, other):
        if not self.is_float():
            out_dtype = Float64
        else:
            out_dtype = DTYPE_TO_FFINAME[self.dtype]
        return np.true_divide(other, self, dtype=out_dtype)

    def __rfloordiv__(self, other):
        if isinstance(other, Series):
            return Series._from_pyseries(other._s.div(self._s))
        f = get_ffi_func("div_<>_rhs", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __rmul__(self, other):
        if isinstance(other, Series):
            return Series._from_pyseries(self._s.mul(other._s))
        f = get_ffi_func("mul_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __getitem__(self, item):
        # assume it is boolean mask
        if isinstance(item, Series):
            return Series._from_pyseries(self._s.filter(item._s))
        # slice
        if type(item) == slice:
            start, stop, stride = item.indices(self.len())
            if stride != 1:
                return NotImplemented
            return self.slice(start, stop - start)
        f = get_ffi_func("get_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return f(item)

    def __setitem__(self, key, value):
        if isinstance(key, Series):
            if key.dtype == Bool:
                self._s = self.set(key, value)._s
            elif key.dtype == UInt64:
                self._s = self.set_at_idx(key, value)._s
            elif key.dtype == UInt32:
                self._s = self.set_at_idx(key.cast_u64(), value)._s
        # TODO: implement for these types without casting to series
        if isinstance(key, (np.ndarray, list, tuple)):
            s = wrap_s(PySeries.new_u64("", np.array(key, np.uint64)))
            self.__setitem__(s, value)

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
        if self.dtype == Bool:
            return self._s.sum_u32()
        f = get_ffi_func("sum_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return f()

    def mean(self):
        """
        Reduce this Series to the mean value.
        """
        # use float type for mean aggregations no matter of base type
        return self._s.mean_f64()

    def min(self):
        """
        Get the minimal value in this Series
        """
        if self.dtype == Bool:
            return self._s.min_u32()
        f = get_ffi_func("min_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return f()

    def max(self):
        """
        Get the maximum value in this Series
        """
        if self.dtype == Bool:
            return self._s.max_u32()
        f = get_ffi_func("max_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return f()

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

    def chunk_lengths(self) -> List[int]:
        return self._s.chunk_lengths()

    def n_chunks(self) -> int:
        """
        Get the number of chunks that this Series contains.
        """
        return self._s.n_chunks()

    def limit(self, num_elements: int) -> Series:
        """
        Take n elements from this Series.

        Parameters
        ----------
        num_elements
            Amount of elements to take.
        """
        return Series._from_pyseries(self._s.limit(num_elements))

    def slice(self, offset: int, length: int) -> Series:
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

    def append(self, other: Series):
        """
        Append a Series to this one.

        Parameters
        ----------
        other
            Series to append
        """
        self._s.append(other._s)

    def filter(self, filter: Series) -> Series:
        """
        Filter elements by a boolean mask

        Parameters
        ----------
        filter
            Boolean mask
        """
        return Series._from_pyseries(self._s.filter(filter._s))

    def head(self, length: Optional[int] = None) -> Series:
        """
        Get first N elements as Series

        Parameters
        ----------
        length
            Length of the head
        """
        return Series._from_pyseries(self._s.head(length))

    def tail(self, length: Optional[int] = None) -> Series:
        """
        Get last N elements as Series

        Parameters
        ----------
        length
            Length of the tail
        """
        return Series._from_pyseries(self._s.tail(length))

    def sort(self, in_place: bool = False, reverse: bool = False) -> Optional[Series]:
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

    def argsort(self, reverse: bool = False) -> Sequence[int]:
        """
        Index location of the sorted variant of this Series.

        Returns
        -------
        indexes
            Indexes that can be used to sort this array.
        """
        return self._s.argsort(reverse)

    def arg_unique(self) -> List[int]:
        """
        Get unique arguments.

        Returns
        -------
        indexes
            Indexes of the unique values
        """
        return self._s.arg_unique()

    def unique(self) -> Series:
        """
        Get unique elements in series.
        """
        return wrap_s(self._s.unique())

    def take(self, indices: Union[np.ndarray, List[int]]) -> Series:
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

    def is_null(self) -> Series:
        """
        Get mask of null values

        Returns
        -------
        Boolean Series
        """
        return Series._from_pyseries(self._s.is_null())

    def is_not_null(self) -> Series:
        """
        Get mask of non null values

        Returns
        -------
        Boolean Series
        """
        return Series._from_pyseries(self._s.is_not_null())

    def is_unique(self) -> Series:
        """
        Get mask of all unique values

        Returns
        -------
        Boolean Series
        """
        return wrap_s(self._s.is_unique())

    def is_duplicated(self) -> Series:
        """
        Get mask of all duplicated values

        Returns
        -------
        Boolean Series
        """
        return wrap_s(self._s.is_unique())

    def series_equal(self, other: Series, null_equal: bool = False) -> bool:
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

    def to_list(self) -> List[Optional[Any]]:
        """
        Convert this Series to a Python List. This operation clones data.
        """
        if self.dtype == List:
            column = []
            for i in range(len(self)):
                subseries = self[i]
                column.append(subseries.to_numpy())
            return column
        return self._s.to_list()

    def rechunk(self, in_place: bool = False) -> Optional[Series]:
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
        return self.dtype not in (Utf8, Bool, List)

    def is_float(self) -> bool:
        """
        Check if this Series has floating point numbers
        """
        return self.dtype in (Float32, Float64)

    def view(self) -> np.ndarray:
        """
        Get a view into this Series data with a numpy array. This operation doesn't clone data, but does not include
        missing values
        """
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
                    args.append(arg.view())
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

    def to_numpy(self) -> np.ndarray:
        """
        Convert this Series to numpy. This operation clones data.
        """
        if self.dtype == List:
            return np.array(self.to_list())

        a = self._s.to_numpy()
        # strings are returned in lists
        if isinstance(a, list):
            return np.array(a)
        return a

    def set(self, filter: Series, value: Union[int, float]) -> Series:
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
        self, idx: Union[Series, np.ndarray], value: Union[int, float]
    ) -> Series:
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

    def clone(self) -> Series:
        """
        Cheap deep clones
        """
        return wrap_s(self._s.clone())

    def fill_none(self, strategy: str) -> Series:
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
        func: Union[Callable[["T"], "T"], Callable[["T"], "S"]],
        dtype_out: Optional["DataType"] = None,
        sniff_dtype: bool = True,
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
        if sniff_dtype and dtype_out is None:
            if self.is_numeric():
                dtype_out = out_to_dtype(func(1))
            elif self.dtype == Bool:
                dtype_out = out_to_dtype(func(True))
            elif self.dtype == List:
                dtype_out = out_to_dtype(func(Series("", [1, 2])))
            elif self.dtype == Utf8:
                dtype_out = out_to_dtype(func("foo"))
            else:
                # try a numeric input for temporal data
                dtype_out = out_to_dtype(func(1))

        if dtype_out is None:
            return wrap_s(self._s.apply_lambda(func))
        else:
            dt = dtype_to_int(dtype_out)
            return wrap_s(self._s.apply_lambda(func, dt))

    def shift(self, periods: int) -> Series:
        """
        Shift the values by a given period and fill the parts that will be empty due to this operation
        with `Nones`.

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).
        """
        return wrap_s(self._s.shift(periods))

    def zip_with(self, mask: Series, other: Series) -> Series:
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

    def str_lengths(self) -> Series:
        """
        Get length of the string values in the Series.

        Returns
        -------
        Series[u32]
        """
        return wrap_s(self._s.str_lengths())

    def str_contains(self, pattern: str) -> Series:
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

    def str_replace(self, pattern: str, value: str) -> Series:
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

    def str_replace_all(self, pattern: str, value: str) -> Series:
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

    def str_to_lowercase(self) -> Series:
        """
        Modify the strings to their lowercase equivalent
        """
        return wrap_s(self._s.str_to_lowercase())

    def str_to_uppercase(self) -> Series:
        """
        Modify the strings to their uppercase equivalent
        """
        return wrap_s(self._s.str_to_uppercase())

    def as_duration(self) -> Series:
        """
        If Series is a date32 or a date64 it can be turned into a duration.
        """
        return wrap_s(self._s.as_duration())

    def str_parse_date(self, datatype: "DataType", fmt: Optional[str] = None):
        if datatype == Date32:
            return wrap_s(self._s.str_parse_date32(fmt))
        if datatype == Date64:
            return wrap_s(self._s.str_parse_date64(fmt))
        return NotImplemented

    def rolling_min(
        self,
        window_size: int,
        weight: Optional[List[float]] = None,
        ignore_null: bool = False,
    ):
        return wrap_s(self._s.rolling_min(window_size, weight, ignore_null))

    def rolling_max(
        self,
        window_size: int,
        weight: Optional[List[float]] = None,
        ignore_null: bool = False,
    ):
        return wrap_s(self._s.rolling_max(window_size, weight, ignore_null))

    def rolling_mean(
        self,
        window_size: int,
        weight: Optional[List[float]] = None,
        ignore_null: bool = False,
    ):
        return wrap_s(self._s.rolling_mean(window_size, weight, ignore_null))

    def rolling_sum(
        self,
        window_size: int,
        weight: Optional[List[float]] = None,
        ignore_null: bool = False,
    ):
        return wrap_s(self._s.rolling_sum(window_size, weight, ignore_null))

    @staticmethod
    def parse_date(
        name: str, values: Sequence[str], dtype: "DataType", fmt: str
    ) -> Series:
        f = get_ffi_func("parse_<>_from_str_slice", dtype, PySeries)
        if f is None:
            return NotImplemented
        return wrap_s(f(name, values, fmt))


def out_to_dtype(out: Any) -> "Datatype":
    if isinstance(out, float):
        return Float64
    if isinstance(out, int):
        return Int64
    if isinstance(out, str):
        return Utf8
    if isinstance(out, bool):
        return Bool
    if isinstance(out, Series):
        return List
    return NotImplemented
