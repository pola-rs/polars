"""
Register Expressions extension with extra functionality.

Enables you to write

    pl.col("dist_a").dist.jaccard_similarity("dist_b")

instead of

    dist.jaccard_similarity("dist_a", "dist_b")

However, note that:

- you will need to add `import expression_lib.extension` to your code.
  Add `# noqa: F401` to avoid linting errors due to unused imports.
- static typing will not recognise your custom namespace. Errors such
  as `"Expr" has no attribute "dist"  [attr-defined]`.
"""

from __future__ import annotations

from typing import Any, Callable

import polars as pl

from expression_lib import date_util, dist, language, panic


@pl.api.register_expr_namespace("language")
class Language:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def __getattr__(self, attr: str) -> Callable[..., pl.Expr]:
        if attr in ("pig_latinnify", "append_args"):

            def func(*args: Any, **kwargs: Any) -> pl.Expr:
                return getattr(language, attr)(self._expr, *args, **kwargs)

            return func
        raise AttributeError(f"{self.__class__} has no attribute {attr}")


@pl.api.register_expr_namespace("dist")
class Distance:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def __getattr__(self, attr: str) -> Callable[..., pl.Expr]:
        if attr in ("hamming_distance", "jaccard_similarity", "haversine"):

            def func(*args: Any, **kwargs: Any) -> pl.Expr:
                return getattr(dist, attr)(self._expr, *args, **kwargs)

            return func
        raise AttributeError(f"{self.__class__} has no attribute {attr}")


@pl.api.register_expr_namespace("date_util")
class DateUtil:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def __getattr__(self, attr: str) -> Callable[..., pl.Expr]:
        if attr in ("change_time_zone", "is_leap_year"):

            def func(*args: Any, **kwargs: Any) -> pl.Expr:
                return getattr(date_util, attr)(self._expr, *args, **kwargs)

            return func
        raise AttributeError(f"{self.__class__} has no attribute {attr}")


@pl.api.register_expr_namespace("panic")
class Panic:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def __getattr__(self, attr: str) -> Callable[..., pl.Expr]:
        if attr in ("panic",):

            def func(*args: Any, **kwargs: Any) -> pl.Expr:
                return getattr(panic, attr)(self._expr, *args, **kwargs)

            return func
        raise AttributeError(f"{self.__class__} has no attribute {attr}")
