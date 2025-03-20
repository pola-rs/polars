import contextlib

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import PyOptFlags


class OptFlags:
    """Optimization flags used during query optimization."""

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

    def no_optimizations(self) -> None:
        """Remove selected optimizations."""
        self._pyoptflags.no_optimizations()

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
