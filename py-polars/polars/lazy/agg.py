# flake8: noqa

from . import col


def count(column: str) -> "Expr":
    return col(column).count()


def sum(column: str) -> "Expr":
    return col(column).sum()


def min(column: str) -> "Expr":
    return col(column).min()


def max(column: str) -> "Expr":
    return col(column).max()


def first(column: str) -> "Expr":
    return col(column).first()


def last(column: str) -> "Expr":
    return col(column).last()


def list(column: str) -> "Expr":
    return col(column).list()


def groups(column: str) -> "Expr":
    return col(column).groups()


def mean(column: str) -> "Expr":
    return col(column).mean()


def median(column: str) -> "Expr":
    return col(column).median()


def n_unique(column: str) -> "Expr":
    return col(column).n_unique()


def quantile(column: str, quantile: float) -> "Expr":
    return col(column).quantile(quantile)
