from typing import Union, List, Callable, Optional

from pypolars import Series
from pypolars.frame import DataFrame, wrap_df

try:
    from .pypolars import (
        PyLazyFrame,
        col as pycol,
        lit as pylit,
        binary_expr,
        PyExpr,
        PyLazyGroupBy,
        when as pywhen,
    )
except:
    import warnings

    warnings.warn("binary files missing")

    __pdoc__ = {"wrap_ldf": False, "wrap_expr": False}


def lazy(self) -> "LazyFrame":
    return wrap_ldf(self._df.lazy())


DataFrame.lazy = lazy


def _selection_to_pyexpr_list(exprs) -> "List[PyExpr]":
    if not isinstance(exprs, list):
        if isinstance(exprs, str):
            exprs = col(exprs)
        exprs = [exprs._pyexpr]
    else:
        new = []
        for expr in exprs:
            if isinstance(expr, str):
                expr = col(expr)
            new.append(expr._pyexpr)
        exprs = new
    return exprs


def wrap_ldf(ldf: "PyLazyFrame") -> "LazyFrame":
    return LazyFrame.from_pyldf(ldf)


class LazyGroupBy:
    def __init__(self, lgb: "PyLazyGroupBy"):
        self.lgb = lgb

    def agg(self, aggs: "Union[List[Expr], Expr]") -> "LazyFrame":
        aggs = _selection_to_pyexpr_list(aggs)
        return wrap_ldf(self.lgb.agg(aggs))


class LazyFrame:
    def __init__(self):
        self._ldf = None

    @staticmethod
    def from_pyldf(ldf: "PyLazyFrame") -> "LazyFrame":
        self = LazyFrame.__new__(LazyFrame)
        self._ldf = ldf
        return self

    def pipe(self, func: Callable, *args, **kwargs):
        """
        Apply a function on Self

        Parameters
        ----------
        func
            Callable
        args
            Arguments
        kwargs
            Keyword arguments
        """
        return func(self, *args, **kwargs)

    def describe_plan(self) -> str:
        return self._ldf.describe_plan()

    def describe_optimized_plan(self) -> str:
        return self._ldf.describe_optimized_plan()

    def sort(self, by_column: str) -> "LazyFrame":
        return wrap_ldf(self._ldf.sort(by_column))

    def collect(
        self,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
    ) -> DataFrame:
        ldf = self._ldf.optimization_toggle(
            type_coercion, predicate_pushdown, projection_pushdown
        )
        return wrap_df(ldf.collect())

    def filter(self, predicate: "Expr") -> "LazyFrame":
        return wrap_ldf(self._ldf.filter(predicate._pyexpr))

    def select(self, exprs: "Expr") -> "LazyFrame":
        exprs = _selection_to_pyexpr_list(exprs)
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
        ldf: "LazyFrame",
        left_on: "Expr",
        right_on: "Expr",
        how="inner",
    ) -> "LazyFrame":
        left_on = left_on._pyexpr
        right_on = right_on._pyexpr
        if how == "inner":
            inner = self._ldf.inner_join(ldf._ldf, left_on, right_on)
        elif how == "left":
            inner = self._ldf.left_join(ldf._ldf, left_on, right_on)
        elif how == "outer":
            inner = self._ldf.outer_join(ldf._ldf, left_on, right_on)
        else:
            return NotImplemented
        return wrap_ldf(inner)

    def with_columns(self, exprs: "List[Expr]") -> "LazyFrame":
        exprs = [e._pyexpr for e in exprs]
        return wrap_ldf(self._ldf.with_columns(exprs))

    def with_column(self, expr: "Expr") -> "LazyFrame":
        return self.with_columns([expr])

    def reverse(self) -> "LazyFrame":
        """
        Reverse the DataFrame.
        """
        return wrap_ldf(self._ldf.reverse())

    def shift(self, periods: int):
        """
        Shift the values by a given period and fill the parts that will be empty due to this operation
        with `Nones`.

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).
        """
        return wrap_ldf(self._ldf.shift(periods))

    def max(self) -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their maximum value
        """
        return wrap_ldf(self._ldf.max())

    def min(self) -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their minimum value
        """
        return wrap_ldf(self._ldf.min())

    def sum(self) -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their sum value
        """
        return wrap_ldf(self._ldf.sum())

    def mean(self) -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their mean value
        """
        return wrap_ldf(self._ldf.mean())

    def median(self) -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their median value
        """
        return wrap_ldf(self._ldf.median())

    def quantile(self, quantile: float) -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their quantile value
        """
        return wrap_ldf(self._ldf.quantile(quantile))

    def explode(self, column: str) -> "LazyFrame":
        """
        Explode lists to long format
        """
        return wrap_ldf(self._ldf.explode(column))

    def drop_duplicates(
        self, maintain_order: bool, subset: "Optional[List[str]]" = None
    ) -> "LazyFrame":
        """
        Drop duplicate rows from this DataFrame.
        Note that this fails if there is a column of type `List` in the DataFrame.
        """
        if subset is not None and not isinstance(subset, list):
            subset = [subset]
        return wrap_ldf(self._ldf.drop_duplicates(maintain_order, subset))

    def drop_nulls(self, subset: "Optional[List[str]]" = None) -> "LazyFrame":
        """
        Drop rows with null values from this DataFrame.
        """
        if subset is not None and not isinstance(subset, list):
            subset = [subset]
        return wrap_ldf(self._ldf.drop_nulls(subset))


def wrap_expr(pyexpr: "PyExpr") -> "Expr":
    return Expr.from_pyexpr(pyexpr)


class Expr:
    def __init__(self):
        self._pyexpr = None

    @staticmethod
    def from_pyexpr(pyexpr: "PyExpr") -> "Expr":
        self = Expr.__new__(Expr)
        self._pyexpr = pyexpr
        return self

    def __add__(self, other):
        return wrap_expr(self._pyexpr + other._pyexpr)

    def __sub__(self, other):
        return wrap_expr(self._pyexpr - other._pyexpr)

    def __mul__(self, other):
        return wrap_expr(self._pyexpr * other._pyexpr)

    def __truediv__(self, other):
        return wrap_expr(self._pyexpr / other._pyexpr)

    def __ge__(self, other):
        return self.gt_eq(other)

    def __le__(self, other):
        return self.lt_eq(other)

    def __eq__(self, other):
        return self.eq(other)

    def __ne__(self, other):
        return self.neq(other)

    def __lt__(self, other):
        return self.lt(other)

    def __gt__(self, other):
        return self.gt(other)

    def eq(self, other: "Expr") -> "Expr":
        return wrap_expr(self._pyexpr.eq(other._pyexpr))

    def neq(self, other: "Expr") -> "Expr":
        return wrap_expr(self._pyexpr.neq(other._pyexpr))

    def gt(self, other: "Expr") -> "Expr":
        return wrap_expr(self._pyexpr.gt(other._pyexpr))

    def gt_eq(self, other: "Expr") -> "Expr":
        return wrap_expr(self._pyexpr.gt_eq(other._pyexpr))

    def lt_eq(self, other: "Expr") -> "Expr":
        return wrap_expr(self._pyexpr.lt_eq(other._pyexpr))

    def lt(self, other: "Expr") -> "Expr":
        return wrap_expr(self._pyexpr.lt(other._pyexpr))

    def alias(self, name: str) -> "Expr":
        return wrap_expr(self._pyexpr.alias(name))

    def is_not(self) -> "Expr":
        return wrap_expr(self._pyexpr.is_not())

    def is_null(self) -> "Expr":
        return wrap_expr(self._pyexpr.is_null())

    def is_not_null(self) -> "Expr":
        return wrap_expr(self._pyexpr.is_not_null())

    def agg_min(self) -> "Expr":
        return wrap_expr(self._pyexpr.agg_min())

    def agg_max(self) -> "Expr":
        return wrap_expr(self._pyexpr.agg_max())

    def agg_mean(self) -> "Expr":
        return wrap_expr(self._pyexpr.agg_mean())

    def agg_median(self) -> "Expr":
        return wrap_expr(self._pyexpr.agg_median())

    def agg_sum(self) -> "Expr":
        return wrap_expr(self._pyexpr.agg_sum())

    def agg_n_unique(self) -> "Expr":
        return wrap_expr(self._pyexpr.agg_n_unique())

    def agg_first(self) -> "Expr":
        return wrap_expr(self._pyexpr.agg_first())

    def agg_last(self) -> "Expr":
        return wrap_expr(self._pyexpr.agg_last())

    def agg_list(self) -> "Expr":
        return wrap_expr(self._pyexpr.agg_list())

    def agg_quantile(self, quantile: float) -> "Expr":
        return wrap_expr(self._pyexpr.agg_quantile(quantile))

    def agg_groups(self) -> "Expr":
        return wrap_expr(self._pyexpr.agg_groups())

    def agg_count(self) -> "Expr":
        return wrap_expr(self._pyexpr.agg_count())

    def cast(self, data_type: "DataType") -> "Expr":
        return wrap_expr(self._pyexpr.cast(data_type))

    def sort(self, reverse: bool) -> "Expr":
        return wrap_expr(self._pyexpr.sort(reverse))

    def shift(self, periods: int) -> "Expr":
        return wrap_expr(self._pyexpr.shift(periods))

    def fill_none(self, strategy: str) -> "Expr":
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
        return wrap_expr(self._pyexpr.fill_none(strategy))

    def max(self) -> "Expr":
        return wrap_expr(self._pyexpr.max())

    def min(self) -> "Expr":
        return wrap_expr(self._pyexpr.min())

    def sum(self) -> "Expr":
        return wrap_expr(self._pyexpr.sum())

    def mean(self) -> "Expr":
        return wrap_expr(self._pyexpr.mean())

    def median(self) -> "Expr":
        return wrap_expr(self._pyexpr.mean())

    def is_unique(self) -> "Expr":
        return wrap_expr(self._pyexpr.is_unique())

    def is_duplicated(self) -> "Expr":
        return wrap_expr(self._pyexpr.is_duplicated())

    def quantile(self, quantile: float) -> "Expr":
        return wrap_expr(self._pyexpr.quantile(quantile))

    def str_lengths(self) -> "Expr":
        return wrap_expr(self._pyexpr.str_lengths())

    def str_contains(self, pattern: str) -> "Expr":
        return wrap_expr(self._pyexpr.str_contains(pattern))

    def str_replace(self, pattern: str, value: str) -> "Expr":
        return wrap_expr(self._pyexpr.str_replace(pattern, value))

    def str_replace_all(self, pattern: str, value: str) -> "Expr":
        return wrap_expr(self._pyexpr.str_replace_all(pattern, value))

    def apply(
        self,
        f: "Union[UDF, Callable[[Series], Series]]",
        output_type: Optional["DataType"] = None,
    ) -> "Expr":
        if isinstance(f, UDF):
            output_type = f.output_type
            f = f.f
        return wrap_expr(self._pyexpr.apply(f, output_type))


class WhenThen:
    def __init__(self, pywhenthen: "PyWhenThen"):
        self._pywhenthen = pywhenthen

    def otherwise(self, expr: "Expr") -> "Expr":
        return wrap_expr(self._pywhenthen.otherwise(expr._pyexpr))


class When:
    def __init__(self, pywhen: "PyWhen"):
        self._pywhen = pywhen

    def then(self, expr: "Expr") -> WhenThen:
        whenthen = self._pywhen.then(expr._pyexpr)
        return WhenThen(whenthen)


def when(expr: "Expr") -> When:
    pw = pywhen(expr._pyexpr)
    return When(pw)


def col(name: str) -> "Expr":
    return wrap_expr(pycol(name))


def lit(value: Union[float, int]) -> "Expr":
    return wrap_expr(pylit(value))


class UDF:
    def __init__(self, f: Callable[[Series], Series], output_type: "DataType"):
        self.f = f
        self.output_type = output_type


def udf(f: Callable[[Series], Series], output_type: "DataType"):
    return UDF(f, output_type)
