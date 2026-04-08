from __future__ import annotations

from inspect import isfunction
from typing import TYPE_CHECKING, Generic, TypeVar
from warnings import warn

import polars._reexport as pl
from polars._utils.various import find_stacklevel

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars import DataFrame, Expr, LazyFrame, Series


__all__ = [
    "register_dataframe_namespace",
    "register_expr_namespace",
    "register_lazyframe_namespace",
    "register_series_namespace",
]

# do not allow override of polars' own namespaces (as registered by '_accessors')
_reserved_namespaces: set[str] = set.union(
    *(cls._accessors for cls in (pl.DataFrame, pl.Expr, pl.LazyFrame, pl.Series))
)


NS = TypeVar("NS")


class NameSpace(Generic[NS]):
    """Establish property-like namespace object for user-defined functionality."""

    def __init__(self, name: str, namespace: type[NS]) -> None:
        self._accessor = name
        self._ns = namespace

    def __get__(self, instance: NS | None, cls: type[NS]) -> NS | type[NS]:
        if instance is None:
            return self._ns

        ns_instance = self._ns(instance)  # type: ignore[call-arg]
        setattr(instance, self._accessor, ns_instance)
        return ns_instance


def _create_namespace(
    name: str, cls: type[Expr | DataFrame | LazyFrame | Series]
) -> Callable[[type[NS]], type[NS]]:
    """Register custom namespace against the underlying Polars class."""

    def namespace(ns_class: type[NS]) -> type[NS]:
        if name in _reserved_namespaces:
            msg = f"cannot override reserved namespace {name!r}"
            raise AttributeError(msg)
        elif (attr := getattr(cls, name, None)) is not None:
            if isfunction(attr) or isinstance(attr, property) or name.startswith("_"):
                msg = f"cannot override `{cls.__name__}.{name}` with custom namespace {ns_class.__name__!r}"
                raise AttributeError(msg)
            warn(
                f"overriding existing custom namespace {name!r} (on {cls.__name__})",
                UserWarning,
                stacklevel=find_stacklevel(),
            )

        setattr(cls, name, NameSpace(name, ns_class))
        cls._accessors.add(name)
        return ns_class

    return namespace


def register_expr_namespace(name: str) -> Callable[[type[NS]], type[NS]]:
    """
    Decorator for registering custom functionality with a Polars Expr.

    Parameters
    ----------
    name
        Name under which the functionality will be accessed.

    See Also
    --------
    register_dataframe_namespace : Register functionality on a DataFrame.
    register_lazyframe_namespace : Register functionality on a LazyFrame.
    register_series_namespace : Register functionality on a Series.

    Examples
    --------
    >>> @pl.api.register_expr_namespace("pow_n")
    ... class PowersOfN:
    ...     def __init__(self, expr: pl.Expr) -> None:
    ...         self._expr = expr
    ...
    ...     def next(self, p: int) -> pl.Expr:
    ...         return (p ** (self._expr.log(p).ceil()).cast(pl.Int64)).cast(pl.Int64)
    ...
    ...     def previous(self, p: int) -> pl.Expr:
    ...         return (p ** (self._expr.log(p).floor()).cast(pl.Int64)).cast(pl.Int64)
    ...
    ...     def nearest(self, p: int) -> pl.Expr:
    ...         return (p ** (self._expr.log(p)).round(0).cast(pl.Int64)).cast(pl.Int64)
    >>>
    >>> df = pl.DataFrame([1.4, 24.3, 55.0, 64.001], schema=["n"])
    >>> df.select(
    ...     pl.col("n"),
    ...     pl.col("n").pow_n.next(p=2).alias("next_pow2"),
    ...     pl.col("n").pow_n.previous(p=2).alias("prev_pow2"),
    ...     pl.col("n").pow_n.nearest(p=2).alias("nearest_pow2"),
    ... )
    shape: (4, 4)
    ┌────────┬───────────┬───────────┬──────────────┐
    │ n      ┆ next_pow2 ┆ prev_pow2 ┆ nearest_pow2 │
    │ ---    ┆ ---       ┆ ---       ┆ ---          │
    │ f64    ┆ i64       ┆ i64       ┆ i64          │
    ╞════════╪═══════════╪═══════════╪══════════════╡
    │ 1.4    ┆ 2         ┆ 1         ┆ 1            │
    │ 24.3   ┆ 32        ┆ 16        ┆ 32           │
    │ 55.0   ┆ 64        ┆ 32        ┆ 64           │
    │ 64.001 ┆ 128       ┆ 64        ┆ 64           │
    └────────┴───────────┴───────────┴──────────────┘
    """
    return _create_namespace(name, pl.Expr)


def register_dataframe_namespace(name: str) -> Callable[[type[NS]], type[NS]]:
    """
    Decorator for registering custom functionality with a Polars DataFrame.

    Parameters
    ----------
    name
        Name under which the functionality will be accessed.

    See Also
    --------
    register_expr_namespace : Register functionality on an Expr.
    register_lazyframe_namespace : Register functionality on a LazyFrame.
    register_series_namespace : Register functionality on a Series.

    Examples
    --------
    >>> @pl.api.register_dataframe_namespace("split")
    ... class SplitFrame:
    ...     def __init__(self, df: pl.DataFrame) -> None:
    ...         self._df = df
    ...
    ...     def by_first_letter_of_column_names(self) -> list[pl.DataFrame]:
    ...         return [
    ...             self._df.select([col for col in self._df.columns if col[0] == f])
    ...             for f in dict.fromkeys(col[0] for col in self._df.columns)
    ...         ]
    ...
    ...     def by_first_letter_of_column_values(self, col: str) -> list[pl.DataFrame]:
    ...         return [
    ...             self._df.filter(pl.col(col).str.starts_with(c))
    ...             for c in sorted(
    ...                 set(df.select(pl.col(col).str.slice(0, 1)).to_series())
    ...             )
    ...         ]
    >>>
    >>> df = pl.DataFrame(
    ...     data=[["xx", 2, 3, 4], ["xy", 4, 5, 6], ["yy", 5, 6, 7], ["yz", 6, 7, 8]],
    ...     schema=["a1", "a2", "b1", "b2"],
    ...     orient="row",
    ... )
    >>> df
    shape: (4, 4)
    ┌─────┬─────┬─────┬─────┐
    │ a1  ┆ a2  ┆ b1  ┆ b2  │
    │ --- ┆ --- ┆ --- ┆ --- │
    │ str ┆ i64 ┆ i64 ┆ i64 │
    ╞═════╪═════╪═════╪═════╡
    │ xx  ┆ 2   ┆ 3   ┆ 4   │
    │ xy  ┆ 4   ┆ 5   ┆ 6   │
    │ yy  ┆ 5   ┆ 6   ┆ 7   │
    │ yz  ┆ 6   ┆ 7   ┆ 8   │
    └─────┴─────┴─────┴─────┘
    >>> df.split.by_first_letter_of_column_names()
    [shape: (4, 2)
    ┌─────┬─────┐
    │ a1  ┆ a2  │
    │ --- ┆ --- │
    │ str ┆ i64 │
    ╞═════╪═════╡
    │ xx  ┆ 2   │
    │ xy  ┆ 4   │
    │ yy  ┆ 5   │
    │ yz  ┆ 6   │
    └─────┴─────┘,
    shape: (4, 2)
    ┌─────┬─────┐
    │ b1  ┆ b2  │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 3   ┆ 4   │
    │ 5   ┆ 6   │
    │ 6   ┆ 7   │
    │ 7   ┆ 8   │
    └─────┴─────┘]
    >>> df.split.by_first_letter_of_column_values("a1")
    [shape: (2, 4)
    ┌─────┬─────┬─────┬─────┐
    │ a1  ┆ a2  ┆ b1  ┆ b2  │
    │ --- ┆ --- ┆ --- ┆ --- │
    │ str ┆ i64 ┆ i64 ┆ i64 │
    ╞═════╪═════╪═════╪═════╡
    │ xx  ┆ 2   ┆ 3   ┆ 4   │
    │ xy  ┆ 4   ┆ 5   ┆ 6   │
    └─────┴─────┴─────┴─────┘, shape: (2, 4)
    ┌─────┬─────┬─────┬─────┐
    │ a1  ┆ a2  ┆ b1  ┆ b2  │
    │ --- ┆ --- ┆ --- ┆ --- │
    │ str ┆ i64 ┆ i64 ┆ i64 │
    ╞═════╪═════╪═════╪═════╡
    │ yy  ┆ 5   ┆ 6   ┆ 7   │
    │ yz  ┆ 6   ┆ 7   ┆ 8   │
    └─────┴─────┴─────┴─────┘]
    """
    return _create_namespace(name, pl.DataFrame)


def register_lazyframe_namespace(name: str) -> Callable[[type[NS]], type[NS]]:
    """
    Decorator for registering custom functionality with a Polars LazyFrame.

    Parameters
    ----------
    name
        Name under which the functionality will be accessed.

    See Also
    --------
    register_expr_namespace : Register functionality on an Expr.
    register_dataframe_namespace : Register functionality on a DataFrame.
    register_series_namespace : Register functionality on a Series.

    Examples
    --------
    >>> @pl.api.register_lazyframe_namespace("types")
    ... class DTypeOperations:
    ...     def __init__(self, lf: pl.LazyFrame) -> None:
    ...         self._lf = lf
    ...
    ...     def split_by_column_dtypes(self) -> list[pl.LazyFrame]:
    ...         return [
    ...             self._lf.select(pl.col(tp))
    ...             for tp in dict.fromkeys(self._lf.collect_schema().dtypes())
    ...         ]
    ...
    ...     def upcast_integer_types(self) -> pl.LazyFrame:
    ...         return self._lf.with_columns(
    ...             pl.col(tp).cast(pl.Int64) for tp in (pl.Int8, pl.Int16, pl.Int32)
    ...         )
    >>>
    >>> lf = pl.LazyFrame(
    ...     data={"a": [1, 2], "b": [3, 4], "c": [5.6, 6.7]},
    ...     schema=[("a", pl.Int16), ("b", pl.Int32), ("c", pl.Float32)],
    ... )
    >>> lf.collect()
    shape: (2, 3)
    ┌─────┬─────┬─────┐
    │ a   ┆ b   ┆ c   │
    │ --- ┆ --- ┆ --- │
    │ i16 ┆ i32 ┆ f32 │
    ╞═════╪═════╪═════╡
    │ 1   ┆ 3   ┆ 5.6 │
    │ 2   ┆ 4   ┆ 6.7 │
    └─────┴─────┴─────┘
    >>> lf.types.upcast_integer_types().collect()
    shape: (2, 3)
    ┌─────┬─────┬─────┐
    │ a   ┆ b   ┆ c   │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ f32 │
    ╞═════╪═════╪═════╡
    │ 1   ┆ 3   ┆ 5.6 │
    │ 2   ┆ 4   ┆ 6.7 │
    └─────┴─────┴─────┘

    >>> lf = pl.LazyFrame(
    ...     data=[["xx", 2, 3, 4], ["xy", 4, 5, 6], ["yy", 5, 6, 7], ["yz", 6, 7, 8]],
    ...     schema=["a1", "a2", "b1", "b2"],
    ...     orient="row",
    ... )
    >>> lf.collect()
    shape: (4, 4)
    ┌─────┬─────┬─────┬─────┐
    │ a1  ┆ a2  ┆ b1  ┆ b2  │
    │ --- ┆ --- ┆ --- ┆ --- │
    │ str ┆ i64 ┆ i64 ┆ i64 │
    ╞═════╪═════╪═════╪═════╡
    │ xx  ┆ 2   ┆ 3   ┆ 4   │
    │ xy  ┆ 4   ┆ 5   ┆ 6   │
    │ yy  ┆ 5   ┆ 6   ┆ 7   │
    │ yz  ┆ 6   ┆ 7   ┆ 8   │
    └─────┴─────┴─────┴─────┘
    >>> pl.collect_all(lf.types.split_by_column_dtypes())
    [shape: (4, 1)
    ┌─────┐
    │ a1  │
    │ --- │
    │ str │
    ╞═════╡
    │ xx  │
    │ xy  │
    │ yy  │
    │ yz  │
    └─────┘, shape: (4, 3)
    ┌─────┬─────┬─────┐
    │ a2  ┆ b1  ┆ b2  │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ i64 │
    ╞═════╪═════╪═════╡
    │ 2   ┆ 3   ┆ 4   │
    │ 4   ┆ 5   ┆ 6   │
    │ 5   ┆ 6   ┆ 7   │
    │ 6   ┆ 7   ┆ 8   │
    └─────┴─────┴─────┘]
    """
    return _create_namespace(name, pl.LazyFrame)


def register_series_namespace(name: str) -> Callable[[type[NS]], type[NS]]:
    """
    Decorator for registering custom functionality with a Polars Series.

    Parameters
    ----------
    name
        Name under which the functionality will be accessed.

    See Also
    --------
    register_expr_namespace : Register functionality on an Expr.
    register_dataframe_namespace : Register functionality on a DataFrame.
    register_lazyframe_namespace : Register functionality on a LazyFrame.

    Examples
    --------
    >>> @pl.api.register_series_namespace("math")
    ... class MathShortcuts:
    ...     def __init__(self, s: pl.Series) -> None:
    ...         self._s = s
    ...
    ...     def square(self) -> pl.Series:
    ...         return self._s * self._s
    ...
    ...     def cube(self) -> pl.Series:
    ...         return self._s * self._s * self._s
    >>>
    >>> s = pl.Series("n", [1.5, 31.0, 42.0, 64.5])
    >>> s.math.square().alias("s^2")
    shape: (4,)
    Series: 's^2' [f64]
    [
        2.25
        961.0
        1764.0
        4160.25
    ]
    >>> s = pl.Series("n", [1, 2, 3, 4, 5])
    >>> s.math.cube().alias("s^3")
    shape: (5,)
    Series: 's^3' [i64]
    [
        1
        8
        27
        64
        125
    ]
    """
    return _create_namespace(name, pl.Series)
