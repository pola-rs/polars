from __future__ import annotations
from .polars import PySeries
import numpy as np


class Series:
    def __init__(self, name: str, values: np.array):
        if not isinstance(values, np.ndarray):
            values = np.array(values)

        self._s: PySeries

        dtype = values.dtype
        if dtype == np.int64:
            self._s = PySeries.new_i64(name, values)
        if dtype == np.int32:
            self._s = PySeries.new_i32(name, values)
        if dtype == np.float32:
            self._s = PySeries.new_f32(name, values)
        if dtype == np.float64:
            self._s = PySeries.new_f64(name, values)
        if isinstance(values[0], str):
            self._s = PySeries.new_str(name, values)
        if dtype == np.bool:
            self._s = PySeries.new_bool(name, values)
        if dtype == np.uint32:
            self._s = PySeries.new_u32(name, values)
        if dtype == np.uint64:
            self._s = PySeries.new_u32(name, np.array(values, dtype=np.uint32))

    @staticmethod
    def from_pyseries(s: PySeries) -> Series:
        self = Series.__new__(Series)
        self._s = s
        return self

    def __str__(self):
        return self._s.as_str()

    def __eq__(self, other):
        return Series.from_pyseries(self._s.eq(other._s))

    def __ne__(self, other):
        return Series.from_pyseries(self._s.neq(other._s))

    def __gt__(self, other):
        return Series.from_pyseries(self._s.gt(other._s))

    def __lt__(self, other):
        return Series.from_pyseries(self._s.lt(other._s))

    def __ge__(self, other):
        return Series.from_pyseries(self._s.gt_eq(other._s))

    def __le__(self, other):
        return Series.from_pyseries(self._s.lt_eq(other._s))

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
        else:
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
        else:
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
        else:
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
        else:
            raise NotImplemented
