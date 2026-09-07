from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sys
    from collections.abc import Collection

    import pyarrow.compute

    import polars as pl
    from polars._typing import SchemaDict

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self


class LazyFrameResolver(abc.ABC):
    """
    LazyFrame resolver.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.
    """

    def lazy(self) -> pl.LazyFrame:
        """Create a LazyFrame with this resolver."""
        import polars as pl

        return pl.LazyFrame.from_lazyframe_resolver(self)

    @abc.abstractmethod
    def schema(self) -> SchemaDict:
        """Fetch the schema of the table."""

    @abc.abstractmethod
    def resolve_lazyframe(
        self,
        *,
        projection: list[str] | None,
        limit: int | None,
        filters: list[FilterExpr],
        filter_columns: list[str],
        filter_drop_columns_idx: int | None,
        existing_resolved_version_key: str | None,
    ) -> pl.LazyFrame | tuple[pl.LazyFrame | None, ResolvedLazyFrameProps]:
        """
        Resolve to a LazyFrame with the provided limit, filters and projection.

        Parameters
        ----------
        projection
            Columns to project. If set to `None`, all columns should be projected.

            In the presence of filters, this may not contain all of the column
            names being referenced by filters. These can be found in `filter_columns`.
        limit
            Row limit.
        filters
            Filters to be applied.
        filter_columns
            Columns used by filter exprs.
        filter_drop_columns_idx
            Columns at and beyond this index in `projection` are only used for
            evaluation of the filter mask, and not afterwards.

            If this is not `None`, `projection` is also guaranteed to not be `None`.
        existing_resolved_version_key
            Version key of a previously resolved LazyFrame. Can be used to skip
            re-resolving.
        """

    def cse_eq(self, other: Self) -> bool:  # noqa: ARG002
        """
        Equality evaluation of `self` with `other` for common subplan elimination.

        Controls whether to allow the evaluation result of `self` to be re-used
        for `other`.
        """
        return False


@dataclass(kw_only=True)
class FilterExpr:
    """
    Filter expression.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.
    """

    expr: pl.Expr
    pyarrow_str: str | None
    _pyarrow_expr: pyarrow.compute.Expression | Exception | None

    @property
    def pyarrow_expr(self) -> pyarrow.compute.Expression | None:
        if isinstance(exc := self._pyarrow_expr, Exception):
            raise exc

        return self._pyarrow_expr


@dataclass(kw_only=True)
class ResolvedLazyFrameProps:
    """
    Additional properties for a resolved LazyFrame.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.

    Parameters
    ----------
    version_key
        Version of the resolved LazyFrame. If this is set, the resolver will be
        called at every `collect()`. If it is unset, the resolver is not called
        again on subsequent `collect()`s if the parameters to `resolve_lazyframe()`
        are unchanged.
    applied_filters
        Indices of filters which will be applied by the resolved LazyFrame.
        These filters will not be applied by the consumer plan.
    """

    version_key: str | None = None
    applied_filters: Collection[int] = ()
