from __future__ import annotations
from .pypolars import PySeries
import numpy as np
from typing import Optional, List, Sequence, Union, Any
from .ffi import ptr_to_numpy, aligned_array_f64, series_from_ptr_f64
import ctypes
from numbers import Number


def wrap_s(s: PySeries) -> Series:
    return Series.from_pyseries(s)


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
                self._s = PySeries.new_opt_str(name, values)
            elif isinstance(dtype, bool):
                self._s = PySeries.new_opt_bool(name, values)
            else:
                raise ValueError(f"dtype: {dtype} not known")

    @staticmethod
    def from_pyseries(s: PySeries) -> Series:
        self = Series.__new__(Series)
        self._s = s
        return self

    def inner(self) -> PySeries:
        return self._s

    def __str__(self) -> str:
        return self._s.as_str()

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, Sequence) and not isinstance(other, str):
            other = Series("", other)
        if isinstance(other, Series):
            return Series.from_pyseries(self._s.eq(other._s))
        f = getattr(self._s, f"eq_{self.dtype}", None)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __ne__(self, other):
        if isinstance(other, Sequence) and not isinstance(other, str):
            other = Series("", other)
        if isinstance(other, Series):
            return Series.from_pyseries(self._s.neq(other._s))
        f = getattr(self._s, f"neq_{self.dtype}", None)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __gt__(self, other):
        if isinstance(other, Sequence) and not isinstance(other, str):
            other = Series("", other)
        if isinstance(other, Series):
            return Series.from_pyseries(self._s.gt(other._s))
        f = getattr(self._s, f"gt_{self.dtype}", None)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __lt__(self, other):
        if isinstance(other, Sequence) and not isinstance(other, str):
            other = Series("", other)
        if isinstance(other, Series):
            return Series.from_pyseries(self._s.lt(other._s))
        f = getattr(self._s, f"lt_{self.dtype}", None)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __ge__(self, other) -> Series:
        if isinstance(other, Sequence) and not isinstance(other, str):
            other = Series("", other)
        if isinstance(other, Series):
            return Series.from_pyseries(self._s.gt_eq(other._s))
        f = getattr(self._s, f"gt_eq_{self.dtype}", None)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __le__(self, other) -> Series:
        if isinstance(other, Sequence) and not isinstance(other, str):
            other = Series("", other)
        if isinstance(other, Series):
            return Series.from_pyseries(self._s.lt_eq(other._s))
        f = getattr(self._s, f"lt_eq_{self.dtype}", None)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __add__(self, other) -> Series:
        if isinstance(other, Series):
            return Series.from_pyseries(self._s.add(other._s))
        f = getattr(self._s, f"add_{self.dtype}", None)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __sub__(self, other) -> Series:
        if isinstance(other, Series):
            return Series.from_pyseries(self._s.sub(other._s))
        f = getattr(self._s, f"sub_{self.dtype}", None)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __truediv__(self, other) -> Series:
        if isinstance(other, Series):
            return Series.from_pyseries(self._s.div(other._s))
        f = getattr(self._s, f"div_{self.dtype}", None)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __mul__(self, other) -> Series:
        if isinstance(other, Series):
            return Series.from_pyseries(self._s.mul(other._s))
        f = getattr(self._s, f"mul_{self.dtype}", None)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __radd__(self, other):
        if isinstance(other, Series):
            return Series.from_pyseries(self._s.add(other._s))
        f = getattr(self._s, f"add_{self.dtype}_rhs", None)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __rsub__(self, other):
        if isinstance(other, Series):
            return Series.from_pyseries(other._s.sub(self._s))
        f = getattr(self._s, f"sub_{self.dtype}_rhs", None)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __rtruediv__(self, other):
        if isinstance(other, Series):
            return Series.from_pyseries(other._s.div(self._s))
        f = getattr(self._s, f"div_{self.dtype}_rhs", None)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __rmul__(self, other):
        if isinstance(other, Series):
            return Series.from_pyseries(self._s.mul(other._s))
        f = getattr(self._s, f"mul_{self.dtype}_rhs", None)
        if f is None:
            return NotImplemented
        return wrap_s(f(other))

    def __getitem__(self, item):
        # assume it is boolean mask
        if isinstance(item, Series):
            return Series.from_pyseries(self._s.filter(item._s))
        # slice
        if type(item) == slice:
            start, stop, stride = item.indices(self.len())
            if stride != 1:
                return NotImplemented
            return self.slice(start, stop - start)
        f = getattr(self._s, f"get_{self.dtype}", None)
        if f is None:
            return NotImplemented
        return f(item)

    def __setitem__(self, key, value):
        if isinstance(key, Series):
            if key.dtype == "bool":
                self._s = self.set(key, value)._s
            elif key.dtype == "u64":
                self._s = self.set_at_idx(key, value)._s
            elif key.dtype == "u32":
                self._s = self.set_at_idx(key.cast_u64(), value)._s
        # TODO: implement for these types without casting to series
        if isinstance(key, (np.ndarray, list, tuple)):
            s = wrap_s(PySeries.new_u64("", np.array(key, np.uint64)))
            self.__setitem__(s, value)

    @property
    def dtype(self):
        return self._s.dtype()

    def sum(self):
        if self.dtype == "bool":
            return self._s.sum_u32()
        f = getattr(self._s, f"sum_{self.dtype}", None)
        if f is None:
            return NotImplemented
        return f()

    def mean(self):
        # use float type for mean aggregations no matter of base type
        return self._s.mean_f64()

    def min(self):
        if self.dtype == "bool":
            return self._s.min_u32()
        f = getattr(self._s, f"min_{self.dtype}", None)
        if f is None:
            return NotImplemented
        return f()

    def max(self):
        if self.dtype == "bool":
            return self._s.max_u32()
        f = getattr(self._s, f"max_{self.dtype}", None)
        if f is None:
            return NotImplemented
        return f()

    @property
    def name(self):
        return self._s.name()

    def rename(self, name: str):
        self._s.rename(name)

    def n_chunks(self) -> int:
        return self._s.n_chunks()

    def limit(self, num_elements: int) -> Series:
        return Series.from_pyseries(self._s.limit(num_elements))

    def slice(self, offset: int, length: int) -> Series:
        return Series.from_pyseries(self._s.slice(offset, length))

    def append(self, other: Series):
        self._s.append(other._s)

    def filter(self, filter: Series) -> Series:
        return Series.from_pyseries(self._s.filter(filter._s))

    def head(self, length: Optional[int] = None) -> Series:
        return Series.from_pyseries(self._s.head(length))

    def tail(self, length: Optional[int] = None) -> Series:
        return Series.from_pyseries(self._s.tail(length))

    def sort(self, in_place: bool = False, reverse: bool = False) -> Optional[Series]:
        if in_place:
            self._s.sort_in_place(reverse)
        else:
            return wrap_s(self._s.sort(reverse))

    def argsort(self, reverse: bool = False) -> Sequence[int]:
        """
        Returns
        -------
        indexes: np.ndarray[int]
        """
        return self._s.argsort(reverse)

    def arg_unique(self) -> List[int]:
        """
        Returns
        -------
        indexes: np.ndarray[int]
        """
        return self._s.arg_unique()

    def take(self, indices: Union[np.ndarray, List[int]]) -> Series:
        if isinstance(indices, list):
            indices = np.array(indices)
        return Series.from_pyseries(self._s.take(indices))

    def null_count(self) -> int:
        return self._s.null_count()

    def is_null(self) -> Series:
        return Series.from_pyseries(self._s.is_null())

    def series_equal(self, other: Series) -> bool:
        return self._s.series_equal(other._s)

    def len(self) -> int:
        return self._s.len()

    def __len__(self):
        return self.len()

    def cast_u8(self):
        return wrap_s(self._s.cast_u8())

    def cast_u16(self):
        return wrap_s(self._s.cast_u16())

    def cast_u32(self):
        return wrap_s(self._s.cast_u32())

    def cast_u64(self):
        return wrap_s(self._s.cast_u64())

    def cast_i8(self):
        return wrap_s(self._s.cast_i8())

    def cast_i16(self):
        return wrap_s(self._s.cast_i16())

    def cast_i32(self):
        return wrap_s(self._s.cast_i32())

    def cast_i64(self):
        return wrap_s(self._s.cast_i64())

    def cast_f32(self):
        return wrap_s(self._s.cast_f32())

    def cast_f64(self):
        return wrap_s(self._s.cast_f64())

    def cast_date32(self):
        return wrap_s(self._s.cast_date32())

    def cast_date64(self):
        return wrap_s(self._s.cast_date64())

    def cast_time64ns(self):
        return wrap_s(self._s.cast_time64ns())

    def cast_duration_ns(self):
        return wrap_s(self._s.cast_duration_ns())

    def to_list(self) -> List[Optional[Any]]:
        return self._s.to_list()

    def rechunk(self, in_place: bool = False) -> Optional[Series]:
        opt_s = self._s.rechunk(in_place)
        if not in_place:
            return wrap_s(opt_s)

    def is_numeric(self) -> bool:
        dtype = self.dtype
        if dtype == "bool":
            return False
        if dtype == "utf8":
            return False
        return True

    def view(self) -> np.ndarray:
        dtype = self.dtype
        if dtype == "u8":
            ptr_type = ctypes.c_uint8
        elif dtype == "u16":
            ptr_type = ctypes.c_uint16
        elif dtype == "u32":
            ptr_type = ctypes.c_uint
        elif dtype == "u64":
            ptr_type = ctypes.c_ulong
        elif dtype == "i8":
            ptr_type = ctypes.c_int8
        elif dtype == "i16":
            ptr_type = ctypes.c_int16
        elif dtype == "i32":
            ptr_type = ctypes.c_int
        elif dtype == "i64":
            ptr_type = ctypes.c_long
        elif dtype == "f32":
            ptr_type = ctypes.c_float
        elif dtype == "f64":
            ptr_type = ctypes.c_double
        elif dtype == "date32":
            ptr_type = ctypes.c_int
        elif dtype == "date64":
            ptr_type = ctypes.c_long
        elif dtype == "time64(ns)":
            ptr_type = ctypes.c_long
        elif dtype == "duration(ns)":
            ptr_type = ctypes.c_long
        else:
            return NotImplemented
        ptr = self._s.as_single_ptr()
        array = ptr_to_numpy(ptr, self.len(), ptr_type)
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
            (out, ptr) = aligned_array_f64(self.len())
            kwargs["out"] = out
            ufunc(*args, **kwargs)
            # get method for current dtype
            f = getattr(self._s, f"unsafe_from_ptr_{self.dtype}")
            return wrap_s(f(ptr, self.len()))

        else:
            return NotImplemented

    def to_numpy(self) -> np.ndarray:
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
        f = getattr(self._s, f"set_with_mask_{self.dtype}", None)
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
        f = getattr(self._s, f"set_at_idx_{self.dtype}", None)
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
