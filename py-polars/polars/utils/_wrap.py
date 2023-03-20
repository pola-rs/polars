from __future__ import annotations

from typing import TYPE_CHECKING

from polars.dataframe import DataFrame
from polars.expr import Expr
from polars.lazyframe import LazyFrame
from polars.series import Series

if TYPE_CHECKING:
    from polars.polars import PyDataFrame, PyExpr, PyLazyFrame, PySeries


def wrap_df(df: PyDataFrame) -> DataFrame:
    return DataFrame._from_pydf(df)


def wrap_ldf(ldf: PyLazyFrame) -> LazyFrame:
    return LazyFrame._from_pyldf(ldf)


def wrap_s(s: PySeries) -> Series:
    return Series._from_pyseries(s)


def wrap_expr(pyexpr: PyExpr) -> Expr:
    return Expr._from_pyexpr(pyexpr)
