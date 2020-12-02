from . import col


def count(column: str) -> "Expr":
    return col(column).agg_count()


def sum(column: str) -> "Expr":
    return col(column).agg_sum()


def min(column: str) -> "Expr":
    return col(column).agg_min()


def max(column: str) -> "Expr":
    return col(column).agg_max()


def first(column: str) -> "Expr":
    return col(column).agg_first()


def last(column: str) -> "Expr":
    return col(column).agg_last()


def list(column: str) -> "Expr":
    return col(column).agg_list()


def groups(column: str) -> "Expr":
    return col(column).agg_groups()


def mean(column: str) -> "Expr":
    return col(column).agg_mean()


def median(column: str) -> "Expr":
    return col(column).agg_median()


def n_unique(column: str) -> "Expr":
    return col(column).agg_n_unique()


def quantile(column: str, quantile: float) -> "Expr":
    return col(column).agg_quantile(quantile)
