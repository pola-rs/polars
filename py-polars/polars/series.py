from __future__ import annotations
from .polars import PySeries
import numpy as np
from typing import Optional, List, Sequence, Union, Any


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
        if isinstance(values, Series):
            values.rename(name)
            self._s = values._s
            return

        self._s: PySeries
        # numpy path
        if not isinstance(values, np.ndarray) and not nullable:
            values = np.array(values)

            dtype = values.dtype
            if dtype == np.int64:
                self._s = PySeries.new_i64(name, values)
            elif dtype == np.int32:
                self._s = PySeries.new_i32(name, values)
            elif dtype == np.float32:
                self._s = PySeries.new_f32(name, values)
            elif dtype == np.float64:
                self._s = PySeries.new_f64(name, values)
            elif isinstance(values[0], str):
                self._s = PySeries.new_str(name, values)
            elif dtype == np.bool:
                self._s = PySeries.new_bool(name, values)
            elif dtype == np.uint32:
                self._s = PySeries.new_u32(name, values)
            elif dtype == np.uint64:
                self._s = PySeries.new_u32(name, np.array(values, dtype=np.uint32))
            else:
                raise ValueError(f"dtype: {dtype} not known")

        # list path
        else:
            dtype = find_first_non_none(values)
            if isinstance(dtype, int):
                self._s = PySeries.new_opt_i64(name, values)
            if isinstance(dtype, float):
                self._s = PySeries.new_opt_f64(name, values)
            if isinstance(dtype, str):
                self._s = PySeries.new_opt_str(name, values)
            if isinstance(dtype, bool):
                self._s = PySeries.new_opt_bool(name, values)

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
        dtype = self.dtype
        if dtype == "u32":
            return Series.from_pyseries(self._s.eq_u32(other))
        if dtype == "i32":
            return Series.from_pyseries(self._s.eq_i32(other))
        if dtype == "i64":
            return Series.from_pyseries(self._s.eq_i64(other))
        if dtype == "f32":
            return Series.from_pyseries(self._s.eq_f32(other))
        if dtype == "f64":
            return Series.from_pyseries(self._s.eq_f64(other))
        if dtype == "str":
            return Series.from_pyseries(self._s.eq_str(other))
        if dtype == "date32":
            return Series.from_pyseries(self._s.eq_i32(other))
        if dtype == "date64":
            return Series.from_pyseries(self._s.eq_i64(other))
        if dtype == "time64(ns)":
            return Series.from_pyseries(self._s.eq_i64(other))
        if dtype == "duration(ns)":
            return Series.from_pyseries(self._s.eq_i64(other))
        raise NotImplemented

    def __ne__(self, other):
        if isinstance(other, Sequence) and not isinstance(other, str):
            other = Series("", other)
        if isinstance(other, Series):
            return Series.from_pyseries(self._s.neq(other._s))
        dtype = self.dtype
        if dtype == "u32":
            return Series.from_pyseries(self._s.neq_u32(other))
        if dtype == "i32":
            return Series.from_pyseries(self._s.neq_i32(other))
        if dtype == "i64":
            return Series.from_pyseries(self._s.neq_i64(other))
        if dtype == "f32":
            return Series.from_pyseries(self._s.neq_f32(other))
        if dtype == "f64":
            return Series.from_pyseries(self._s.neq_f64(other))
        if dtype == "str":
            return Series.from_pyseries(self._s.neq_str(other))
        if dtype == "date32":
            return Series.from_pyseries(self._s.neq_i32(other))
        if dtype == "date64":
            return Series.from_pyseries(self._s.neq_i64(other))
        if dtype == "time64(ns)":
            return Series.from_pyseries(self._s.neq_i64(other))
        if dtype == "duration(ns)":
            return Series.from_pyseries(self._s.neq_i64(other))
        raise NotImplemented

    def __gt__(self, other):
        if isinstance(other, Sequence) and not isinstance(other, str):
            other = Series("", other)
        if isinstance(other, Series):
            return Series.from_pyseries(self._s.gt(other._s))
        dtype = self.dtype
        if dtype == "u32":
            return Series.from_pyseries(self._s.gt_u32(other))
        if dtype == "i32":
            return Series.from_pyseries(self._s.gt_i32(other))
        if dtype == "i64":
            return Series.from_pyseries(self._s.gt_i64(other))
        if dtype == "f32":
            return Series.from_pyseries(self._s.gt_f32(other))
        if dtype == "f64":
            return Series.from_pyseries(self._s.gt_f64(other))
        if dtype == "str":
            return Series.from_pyseries(self._s.gt_str(other))
        if dtype == "date32":
            return Series.from_pyseries(self._s.gt_i32(other))
        if dtype == "date64":
            return Series.from_pyseries(self._s.gt_i64(other))
        if dtype == "time64(ns)":
            return Series.from_pyseries(self._s.gt_i64(other))
        if dtype == "duration(ns)":
            return Series.from_pyseries(self._s.gt_i64(other))
        raise NotImplemented

    def __lt__(self, other):
        if isinstance(other, Sequence) and not isinstance(other, str):
            other = Series("", other)
        if isinstance(other, Series):
            return Series.from_pyseries(self._s.lt(other._s))
        dtype = self.dtype
        if dtype == "u32":
            return Series.from_pyseries(self._s.lt_u32(other))
        if dtype == "i32":
            return Series.from_pyseries(self._s.lt_i32(other))
        if dtype == "i64":
            return Series.from_pyseries(self._s.lt_i64(other))
        if dtype == "f32":
            return Series.from_pyseries(self._s.lt_f32(other))
        if dtype == "f64":
            return Series.from_pyseries(self._s.lt_f64(other))
        if dtype == "str":
            return Series.from_pyseries(self._s.lt_str(other))
        if dtype == "date32":
            return Series.from_pyseries(self._s.lt_i32(other))
        if dtype == "date64":
            return Series.from_pyseries(self._s.lt_i64(other))
        if dtype == "time64(ns)":
            return Series.from_pyseries(self._s.lt_i64(other))
        if dtype == "duration(ns)":
            return Series.from_pyseries(self._s.lt_i64(other))
        raise NotImplemented

    def __ge__(self, other) -> Series:
        if isinstance(other, Sequence) and not isinstance(other, str):
            other = Series("", other)
        if isinstance(other, Series):
            return Series.from_pyseries(self._s.gt_eq(other._s))
        dtype = self.dtype
        if dtype == "u32":
            return Series.from_pyseries(self._s.gt_eq_u32(other))
        if dtype == "i32":
            return Series.from_pyseries(self._s.gt_eq_i32(other))
        if dtype == "i64":
            return Series.from_pyseries(self._s.gt_eq_i64(other))
        if dtype == "f32":
            return Series.from_pyseries(self._s.gt_eq_f32(other))
        if dtype == "f64":
            return Series.from_pyseries(self._s.gt_eq_f64(other))
        if dtype == "str":
            return Series.from_pyseries(self._s.gt_eq_str(other))
        if dtype == "date32":
            return Series.from_pyseries(self._s.gt_eq_i32(other))
        if dtype == "date64":
            return Series.from_pyseries(self._s.gt_eq_i64(other))
        if dtype == "time64(ns)":
            return Series.from_pyseries(self._s.gt_eq_i64(other))
        if dtype == "duration(ns)":
            return Series.from_pyseries(self._s.gt_eq_i64(other))
        raise NotImplemented

    def __le__(self, other) -> Series:
        if isinstance(other, Sequence) and not isinstance(other, str):
            other = Series("", other)
        if isinstance(other, Series):
            return Series.from_pyseries(self._s.lt_eq(other._s))
        dtype = self.dtype
        if dtype == "u32":
            return Series.from_pyseries(self._s.lt_eq_u32(other))
        if dtype == "i32":
            return Series.from_pyseries(self._s.lt_eq_i32(other))
        if dtype == "i64":
            return Series.from_pyseries(self._s.lt_eq_i64(other))
        if dtype == "f32":
            return Series.from_pyseries(self._s.lt_eq_f32(other))
        if dtype == "f64":
            return Series.from_pyseries(self._s.lt_eq_f64(other))
        if dtype == "str":
            return Series.from_pyseries(self._s.lt_eq_str(other))
        if dtype == "date32":
            return Series.from_pyseries(self._s.lt_eq_i32(other))
        if dtype == "date64":
            return Series.from_pyseries(self._s.lt_eq_i64(other))
        if dtype == "time64(ns)":
            return Series.from_pyseries(self._s.lt_eq_i64(other))
        if dtype == "duration(ns)":
            return Series.from_pyseries(self._s.lt_eq_i64(other))
        raise NotImplemented

    def __add__(self, other) -> Series:
        if isinstance(other, Series):
            return Series.from_pyseries(self._s.add(other._s))
        dtype = self.dtype
        if dtype == "u32":
            return Series.from_pyseries(self._s.add_u32(other))
        if dtype == "i32":
            return Series.from_pyseries(self._s.add_i32(other))
        if dtype == "i64":
            return Series.from_pyseries(self._s.add_i64(other))
        if dtype == "f32":
            return Series.from_pyseries(self._s.add_f32(other))
        if dtype == "f64":
            return Series.from_pyseries(self._s.add_f64(other))
        if dtype == "date32":
            return Series.from_pyseries(self._s.add_i32(other))
        if dtype == "date64":
            return Series.from_pyseries(self._s.add_i64(other))
        if dtype == "time64(ns)":
            return Series.from_pyseries(self._s.add_i64(other))
        if dtype == "duration(ns)":
            return Series.from_pyseries(self._s.add_i64(other))
        raise NotImplemented

    def __sub__(self, other) -> Series:
        if isinstance(other, Series):
            return Series.from_pyseries(self._s.sub(other._s))
        dtype = self.dtype
        if dtype == "u32":
            return Series.from_pyseries(self._s.sub_u32(other))
        if dtype == "i32":
            return Series.from_pyseries(self._s.sub_i32(other))
        if dtype == "i64":
            return Series.from_pyseries(self._s.sub_i64(other))
        if dtype == "f32":
            return Series.from_pyseries(self._s.sub_f32(other))
        if dtype == "f64":
            return Series.from_pyseries(self._s.sub_f64(other))
        if dtype == "date32":
            return Series.from_pyseries(self._s.sub_i32(other))
        if dtype == "date64":
            return Series.from_pyseries(self._s.sub_i64(other))
        if dtype == "time64(ns)":
            return Series.from_pyseries(self._s.sub_i64(other))
        if dtype == "duration(ns)":
            return Series.from_pyseries(self._s.sub_i64(other))
        raise NotImplemented

    def __truediv__(self, other) -> Series:
        if isinstance(other, Series):
            return Series.from_pyseries(self._s.div(other._s))
        dtype = self.dtype
        if dtype == "u32":
            return Series.from_pyseries(self._s.div_u32(other))
        if dtype == "i32":
            return Series.from_pyseries(self._s.div_i32(other))
        if dtype == "i64":
            return Series.from_pyseries(self._s.div_i64(other))
        if dtype == "f32":
            return Series.from_pyseries(self._s.div_f32(other))
        if dtype == "f64":
            return Series.from_pyseries(self._s.div_f64(other))
        if dtype == "date32":
            return Series.from_pyseries(self._s.div_i32(other))
        if dtype == "date64":
            return Series.from_pyseries(self._s.div_i64(other))
        if dtype == "time64(ns)":
            return Series.from_pyseries(self._s.div_i64(other))
        if dtype == "duration(ns)":
            return Series.from_pyseries(self._s.div_i64(other))
        raise NotImplemented

    def __mul__(self, other) -> Series:
        if isinstance(other, Series):
            return Series.from_pyseries(self._s.mul(other._s))
        dtype = self.dtype
        if dtype == "u32":
            return Series.from_pyseries(self._s.mul_u32(other))
        if dtype == "i32":
            return Series.from_pyseries(self._s.mul_i32(other))
        if dtype == "i64":
            return Series.from_pyseries(self._s.mul_i64(other))
        if dtype == "f32":
            return Series.from_pyseries(self._s.mul_f32(other))
        if dtype == "f64":
            return Series.from_pyseries(self._s.mul_f64(other))
        if dtype == "date32":
            return Series.from_pyseries(self._s.mul_i32(other))
        if dtype == "date64":
            return Series.from_pyseries(self._s.mul_i64(other))
        if dtype == "time64(ns)":
            return Series.from_pyseries(self._s.mul_i64(other))
        if dtype == "duration(ns)":
            return Series.from_pyseries(self._s.mul_i64(other))
        raise NotImplemented

    def __radd__(self, other):
        if isinstance(other, Series):
            return Series.from_pyseries(self._s.add(other._s))
        dtype = self.dtype
        if dtype == "u32":
            return Series.from_pyseries(self._s.add_u32_rhs(other))
        if dtype == "i32":
            return Series.from_pyseries(self._s.add_i32_rhs(other))
        if dtype == "i64":
            return Series.from_pyseries(self._s.add_i64_rhs(other))
        if dtype == "f32":
            return Series.from_pyseries(self._s.add_f32_rhs(other))
        if dtype == "f64":
            return Series.from_pyseries(self._s.add_f64_rhs(other))
        if dtype == "date32":
            return Series.from_pyseries(self._s.add_i32_rhs(other))
        if dtype == "date64":
            return Series.from_pyseries(self._s.add_i64_rhs(other))
        if dtype == "time64(ns)":
            return Series.from_pyseries(self._s.add_i64_rhs(other))
        if dtype == "duration(ns)":
            return Series.from_pyseries(self._s.add_i64_rhs(other))
        raise NotImplemented

    def __rsub__(self, other):
        if isinstance(other, Series):
            return Series.from_pyseries(other._s.sub(self._s))
        dtype = self.dtype
        if dtype == "u32":
            return Series.from_pyseries(self._s.sub_u32_rhs(other))
        if dtype == "i32":
            return Series.from_pyseries(self._s.sub_i32_rhs(other))
        if dtype == "i64":
            return Series.from_pyseries(self._s.sub_i64_rhs(other))
        if dtype == "f32":
            return Series.from_pyseries(self._s.sub_f32_rhs(other))
        if dtype == "f64":
            return Series.from_pyseries(self._s.sub_f64_rhs(other))
        if dtype == "date32":
            return Series.from_pyseries(self._s.sub_i32_rhs(other))
        if dtype == "date64":
            return Series.from_pyseries(self._s.sub_i64_rhs(other))
        if dtype == "time64(ns)":
            return Series.from_pyseries(self._s.sub_i64_rhs(other))
        if dtype == "duration(ns)":
            return Series.from_pyseries(self._s.sub_i64_rhs(other))
        raise NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, Series):
            return Series.from_pyseries(other._s.div(self._s))
        dtype = self.dtype
        if dtype == "u32":
            return Series.from_pyseries(self._s.div_u32_rhs(other))
        if dtype == "i32":
            return Series.from_pyseries(self._s.div_i32_rhs(other))
        if dtype == "i64":
            return Series.from_pyseries(self._s.div_i64_rhs(other))
        if dtype == "f32":
            return Series.from_pyseries(self._s.div_f32_rhs(other))
        if dtype == "f64":
            return Series.from_pyseries(self._s.div_f64_rhs(other))
        if dtype == "date32":
            return Series.from_pyseries(self._s.div_i32_rhs(other))
        if dtype == "date64":
            return Series.from_pyseries(self._s.div_i64_rhs(other))
        if dtype == "time64(ns)":
            return Series.from_pyseries(self._s.div_i64_rhs(other))
        if dtype == "duration(ns)":
            return Series.from_pyseries(self._s.div_i64_rhs(other))
        raise NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Series):
            return Series.from_pyseries(self._s.mul(other._s))
        dtype = self.dtype
        if dtype == "u32":
            return Series.from_pyseries(self._s.mul_u32_rhs(other))
        if dtype == "i32":
            return Series.from_pyseries(self._s.mul_i32_rhs(other))
        if dtype == "i64":
            return Series.from_pyseries(self._s.mul_i64_rhs(other))
        if dtype == "f32":
            return Series.from_pyseries(self._s.mul_f32_rhs(other))
        if dtype == "f64":
            return Series.from_pyseries(self._s.mul_f64_rhs(other))
        if dtype == "date32":
            return Series.from_pyseries(self._s.mul_i32_rhs(other))
        if dtype == "date64":
            return Series.from_pyseries(self._s.mul_i64_rhs(other))
        if dtype == "time64(ns)":
            return Series.from_pyseries(self._s.mul_i64_rhs(other))
        if dtype == "duration(ns)":
            return Series.from_pyseries(self._s.mul_i64_rhs(other))
        raise NotImplemented

    def __getitem__(self, item):
        # assume it is boolean mask
        if isinstance(item, Series):
            return Series.from_pyseries(self._s.filter(item._s))
        # slice
        if type(item) == slice:
            start, stop, stride = item.indices(self.len())
            if stride != 1:
                raise NotImplemented
            return self.slice(start, stop - start)
        raise NotImplemented

    @property
    def dtype(self):
        return self._s.dtype()

    def sum(self):
        dtype = self.dtype
        if dtype == "u32":
            return self._s.sum_u32()
        if dtype == "i32":
            return self._s.sum_i32()
        if dtype == "i64":
            return self._s.sum_i64()
        if dtype == "f32":
            return self._s.sum_f32()
        if dtype == "f64":
            return self._s.sum_f64()
        if dtype == "bool":
            return self._s.sum_u32()
        raise NotImplemented

    def mean(self):
        # use float type for mean aggregations no matter of base type
        dtype = self.dtype
        if dtype == "u32":
            return self._s.mean_f64()
        if dtype == "i32":
            return self._s.mean_f64()
        if dtype == "i64":
            return self._s.mean_f64()
        if dtype == "f32":
            return self._s.mean_f32()
        if dtype == "f64":
            return self._s.mean_f64()
        if dtype == "bool":
            return self._s.mean_f32()
        raise NotImplemented

    def min(self):
        dtype = self.dtype
        if dtype == "u32":
            return self._s.min_u32()
        if dtype == "i32":
            return self._s.min_i32()
        if dtype == "i64":
            return self._s.min_i64()
        if dtype == "f32":
            return self._s.min_f32()
        if dtype == "f64":
            return self._s.min_f64()
        if dtype == "bool":
            return self._s.min_u32()
        raise NotImplemented

    def max(self):
        dtype = self.dtype
        if dtype == "u32":
            return self._s.max_u32()
        if dtype == "i32":
            return self._s.max_i32()
        if dtype == "i64":
            return self._s.max_i64()
        if dtype == "f32":
            return self._s.max_f32()
        if dtype == "f64":
            return self._s.max_f64()
        if dtype == "bool":
            return self._s.max_u32()
        raise NotImplemented

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

    def cast_u32(self):
        return wrap_s(self._s.cast_u32())

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
