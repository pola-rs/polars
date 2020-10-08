from __future__ import annotations
import os
from typing import Union, List

from pypolars.frame import DataFrame, wrap_df

if not os.environ.get("DOC_BUILDING", False):
    from .pypolars import PyLazyFrame, col, lit, binary_expr, PyExpr, PyLazyGroupBy


def lazy(self) -> "LazyFrame":
    return wrap_ldf(self._df.lazy())


DataFrame.lazy = lazy


def wrap_ldf(ldf: PyLazyFrame) -> LazyFrame:
    return LazyFrame.from_pyldf(ldf)


class LazyGroupBy:
    def __init__(self, lgb: PyLazyGroupBy):
        self.lgb = lgb

    def agg(self, aggs: List[PyExpr]) -> LazyFrame:
        return wrap_ldf(self.lgb.agg(aggs))


class LazyFrame:
    def __init__(self):
        self._ldf = None

    @staticmethod
    def from_pyldf(ldf: "PyLazyFrame") -> "LazyFrame":
        self = LazyFrame.__new__(LazyFrame)
        self._ldf = ldf
        return self

    def describe_plan(self) -> str:
        return self._ldf.describe_plan()

    def describe_optimized_plan(self) -> str:
        return self._ldf.describe_optimized_plan()

    def sort(self, by_column: str) -> LazyFrame:
        return wrap_ldf(self._ldf.sort(by_column))

    def collect(self) -> DataFrame:
        return wrap_df(self._ldf.collect())

    def filter(self, predicate: PyExpr) -> LazyFrame:
        return wrap_ldf(self._ldf.filter(predicate))

    def select(self, exprs: PyExpr) -> LazyFrame:
        if not isinstance(exprs, list):
            exprs = [exprs]
        return wrap_ldf(self._ldf.select(exprs))

    def groupby(self, by: Union[str, List[str]]) -> LazyGroupBy:
        """
        Start a groupby operation

        Parameters
        ----------
        by
            Column(s) to group by.
        """
        if isinstance(by, str):
            by = [by]
        lgb = self._ldf.groupby(by)
        return LazyGroupBy(lgb)

    def join(
        self,
        ldf: LazyFrame,
        left_on: str,
        right_on: str,
        how="inner",
    ) -> LazyFrame:
        if how == "inner":
            inner = self._ldf.inner_join(ldf._ldf, left_on, right_on)
        elif how == "left":
            inner = self._ldf.left_join(ldf._ldf, left_on, right_on)
        elif how == "outer":
            inner = self._ldf.outer_join(ldf._ldf, left_on, right_on)
        else:
            return NotImplemented
        return wrap_ldf(inner)

    def with_columns(self, exprs: List[PyExpr]) -> LazyFrame:
        return wrap_ldf(self._ldf.with_columns(exprs))

    def with_column(self, expr: PyExpr) -> LazyFrame:
        return self.with_columns([expr])
