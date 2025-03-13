use polars::prelude::OptFlags;
use pyo3::pymethods;

use super::PyOptFlags;

macro_rules! flag_getter_setters {
    ($(($flag:ident, $getter:ident, $setter:ident))+) => {
        #[pymethods]
        impl PyOptFlags {
            #[staticmethod]
            pub fn empty() -> Self {
                Self {
                    inner: OptFlags::empty()
                }
            }

            pub fn no_optimizations(&mut self) {
                self.inner.remove(OptFlags::PREDICATE_PUSHDOWN);
                self.inner.remove(OptFlags::PROJECTION_PUSHDOWN);
                self.inner.remove(OptFlags::COMM_SUBPLAN_ELIM);
                self.inner.remove(OptFlags::COMM_SUBEXPR_ELIM);
                self.inner.remove(OptFlags::CLUSTER_WITH_COLUMNS);
                self.inner.remove(OptFlags::COLLAPSE_JOINS);
                self.inner.remove(OptFlags::CHECK_ORDER_OBSERVE);
                self.inner.remove(OptFlags::SIMPLIFY_EXPR);
                self.inner.remove(OptFlags::SLICE_PUSHDOWN);
            }

            $(
            #[getter]
            fn $getter(&self) -> bool {
                self.inner.contains(OptFlags::$flag)
            }
            #[setter]
            fn $setter(&mut self, value: bool) {
                self.inner.set(OptFlags::$flag, value)
            }
            )+
        }
    };
}

flag_getter_setters! {
    (PROJECTION_PUSHDOWN, get_projection_pushdown, set_projection_pushdown)
    (PREDICATE_PUSHDOWN, get_predicate_pushdown, set_predicate_pushdown)
    (CLUSTER_WITH_COLUMNS, get_cluster_with_columns, set_cluster_with_columns)
    (TYPE_COERCION, get_type_coercion, set_type_coercion)
    (SIMPLIFY_EXPR, get_simplify_expression, set_simplify_expression)
    (TYPE_CHECK, get_type_check, set_type_check)
    (SLICE_PUSHDOWN, get_slice_pushdown, set_slice_pushdown)
    (COMM_SUBPLAN_ELIM, get_comm_subplan_elim, set_comm_subplan_elim)
    (COMM_SUBEXPR_ELIM, get_comm_subexpr_elim, set_comm_subexpr_elim)
    (COLLAPSE_JOINS, get_collapse_joins, set_collapse_joins)
    (CHECK_ORDER_OBSERVE, get_check_order_observe, set_check_order_observe)
}
