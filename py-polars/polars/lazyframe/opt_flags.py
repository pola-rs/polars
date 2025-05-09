from __future__ import annotations

import contextlib

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import PyOptFlags

import inspect
from functools import wraps
from typing import TYPE_CHECKING, Callable, TypeVar

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 10):
        from typing import ParamSpec
    else:
        from typing_extensions import ParamSpec

    P = ParamSpec("P")
    T = TypeVar("T")


class QueryOptFlags:
    """
    Optimization flags used during query optimization.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.
    """

    def __init__(
        self,
        *,
        _type_check: bool = True,
        _type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        slice_pushdown: bool = True,
        comm_subplan_elim: bool = True,
        comm_subexpr_elim: bool = True,
        cluster_with_columns: bool = True,
        collapse_joins: bool = True,
        check_order_observe: bool = True,
    ) -> None:
        self._pyoptflags = PyOptFlags.empty()

        self._pyoptflags.type_check = _type_check
        self._pyoptflags.type_coercion = _type_coercion
        self._pyoptflags.predicate_pushdown = predicate_pushdown
        self._pyoptflags.projection_pushdown = projection_pushdown
        self._pyoptflags.simplify_expression = simplify_expression
        self._pyoptflags.slice_pushdown = slice_pushdown
        self._pyoptflags.comm_subplan_elim = comm_subplan_elim
        self._pyoptflags.comm_subexpr_elim = comm_subexpr_elim
        self._pyoptflags.collapse_joins = collapse_joins
        self._pyoptflags.check_order_observe = check_order_observe

    @staticmethod
    def none() -> QueryOptFlags:
        """Create new empty set off optimizations."""
        optflags = QueryOptFlags()
        optflags.no_optimizations()
        return optflags

    def no_optimizations(self) -> None:
        """Remove selected optimizations."""
        self._pyoptflags.no_optimizations()

    @property
    def type_coercion(self) -> bool:
        """Do type coercion."""
        return self._pyoptflags.type_coercion

    @type_coercion.setter
    def type_coercion(self, value: bool) -> None:
        self._pyoptflags.type_coercion = value

    @property
    def projection_pushdown(self) -> bool:
        """Only read columns that are used later in the query."""
        return self._pyoptflags.projection_pushdown

    @projection_pushdown.setter
    def projection_pushdown(self, value: bool) -> None:
        self._pyoptflags.projection_pushdown = value

    @property
    def predicate_pushdown(self) -> bool:
        """Apply predicates/filters as early as possible."""
        return self._pyoptflags.predicate_pushdown

    @predicate_pushdown.setter
    def predicate_pushdown(self, value: bool) -> None:
        self._pyoptflags.predicate_pushdown = value

    @property
    def cluster_with_columns(self) -> bool:
        """Cluster sequential `with_columns` calls to independent calls."""
        return self._pyoptflags.cluster_with_columns

    @cluster_with_columns.setter
    def cluster_with_columns(self, value: bool) -> None:
        self._pyoptflags.cluster_with_columns = value

    @property
    def simplify_expression(self) -> bool:
        """Run many expression optimization rules until fixed point."""
        return self._pyoptflags.simplify_expression

    @simplify_expression.setter
    def simplify_expression(self, value: bool) -> None:
        self._pyoptflags.simplify_expression = value

    @property
    def slice_pushdown(self) -> bool:
        """Pushdown slices/limits."""
        return self._pyoptflags.slice_pushdown

    @slice_pushdown.setter
    def slice_pushdown(self, value: bool) -> None:
        self._pyoptflags.slice_pushdown = value

    @property
    def common_subplan_elim(self) -> bool:
        """Elide duplicate plans and caches their outputs."""
        return self._pyoptflags.common_subplan_elim

    @common_subplan_elim.setter
    def common_subplan_elim(self, value: bool) -> None:
        self._pyoptflags.common_subplan_elim = value

    @property
    def common_subexpr_elim(self) -> bool:
        """Elide duplicate expressions and caches their outputs."""
        return self._pyoptflags.common_subexpr_elim

    @common_subexpr_elim.setter
    def common_subexpr_elim(self, value: bool) -> None:
        self._pyoptflags.common_subexpr_elim = value

    @property
    def collapse_joins(self) -> bool:
        """Collapse slower joins with filters into faster joins."""
        return self._pyoptflags.collapse_joins

    @collapse_joins.setter
    def collapse_joins(self, value: bool) -> None:
        self._pyoptflags.collapse_joins = value

    @property
    def check_order_observe(self) -> bool:
        """Do not maintain order if the order would not be observed."""
        return self._pyoptflags.check_order_observe

    @check_order_observe.setter
    def check_order_observe(self, value: bool) -> None:
        self._pyoptflags.check_order_observe = value


DEFAULT_QUERY_OPT_FLAGS = QueryOptFlags()


def forward_old_opt_flags() -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to mark to forward the old optimization flags."""

    def helper(f: QueryOptFlags, field_name: str, value: bool) -> QueryOptFlags:
        setattr(f, field_name, value)
        return f

    def helper_hidden(f: QueryOptFlags, field_name: str, value: bool) -> QueryOptFlags:
        setattr(f._pyoptflags, field_name, value)
        return f

    def clear_optimizations(f: QueryOptFlags, value: bool) -> QueryOptFlags:
        if value:
            return QueryOptFlags.none()
        else:
            return f

    def eager(f: QueryOptFlags, value: bool) -> QueryOptFlags:
        if value:
            f.no_optimization()
            f._pyoptflags.eager = True
            return f
        else:
            return f

    OLD_OPT_PARAMETERS_MAPPING = {
        "no_optimization": lambda f, v: clear_optimizations(f, v),
        "_eager": lambda f, v: eager(f, v),
        "type_coercion": lambda f, v: helper(f, "_type_coercion", v),
        "_type_check": lambda f, v: helper(f, "_type_check", v),
        "predicate_pushdown": lambda f, v: helper(f, "predicate_pushdown", v),
        "projection_pushdown": lambda f, v: helper(f, "projection_pushdown", v),
        "simplify_expression": lambda f, v: helper(f, "simplify_expression", v),
        "slice_pushdown": lambda f, v: helper(f, "slice_pushdown", v),
        "comm_subplan_elim": lambda f, v: helper(f, "comm_subplan_elim", v),
        "comm_subexpr_elim": lambda f, v: helper(f, "comm_subexpr_elim", v),
        "cluster_with_columns": lambda f, v: helper(f, "cluster_with_columns", v),
        "collapse_joins": lambda f, v: helper(f, "collapse_joins", v),
        "_check_order": lambda f, v: helper(f, "check_order_observe", v),
    }

    def decorate(function: Callable[P, T]) -> Callable[P, T]:
        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            optflags: QueryOptFlags = kwargs.get(
                "optimizations", DEFAULT_QUERY_OPT_FLAGS
            )
            for key in list(kwargs.keys()):
                cb = OLD_OPT_PARAMETERS_MAPPING.get(key)
                if cb is not None:
                    from polars._utils.various import issue_warning

                    message = f"optimization flag `{key}` is deprecated. Please use `optimizations` parameter\n(Deprecated in version 1.30.0)"
                    issue_warning(message, DeprecationWarning)
                    optflags = cb(optflags, kwargs.pop(key))

            kwargs["optimizations"] = optflags
            return function(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]
        return wrapper

    return decorate
