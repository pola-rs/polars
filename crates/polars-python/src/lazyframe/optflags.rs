use polars::prelude::OptFlags;
use pyo3::pymethods;

use super::PyOptFlags;

macro_rules! flag_getter_setters {
    ($(($flag:ident, $getter:ident, $setter:ident, clear=$clear:literal))+) => {
        #[pymethods]
        impl PyOptFlags {
            #[staticmethod]
            pub fn empty() -> Self {
                Self {
                    inner: OptFlags::empty()
                }
            }

            #[staticmethod]
            #[allow(clippy::should_implement_trait)]
            pub fn default() -> Self {
                Self { inner: OptFlags::default() }
            }

            pub fn no_optimizations(&mut self) {
                $(if $clear {
                    self.inner.remove(OptFlags::$flag);
                })+
            }

            pub fn copy(&self) -> Self {
                Self { inner: self.inner }
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
    (TYPE_COERCION, get_type_coercion, set_type_coercion, clear=false)
    (TYPE_CHECK, get_type_check, set_type_check, clear=false)

    (PROJECTION_PUSHDOWN, get_projection_pushdown, set_projection_pushdown, clear=true)
    (PREDICATE_PUSHDOWN, get_predicate_pushdown, set_predicate_pushdown, clear=true)
    (CLUSTER_WITH_COLUMNS, get_cluster_with_columns, set_cluster_with_columns, clear=true)
    (SIMPLIFY_EXPR, get_simplify_expression, set_simplify_expression, clear=true)
    (SLICE_PUSHDOWN, get_slice_pushdown, set_slice_pushdown, clear=true)
    (COMM_SUBPLAN_ELIM, get_comm_subplan_elim, set_comm_subplan_elim, clear=true)
    (COMM_SUBEXPR_ELIM, get_comm_subexpr_elim, set_comm_subexpr_elim, clear=true)
    (COLLAPSE_JOINS, get_collapse_joins, set_collapse_joins, clear=true)
    (CHECK_ORDER_OBSERVE, get_check_order_observe, set_check_order_observe, clear=true)
    (FAST_PROJECTION, get_fast_projection, set_fast_projection, clear=true)

    (EAGER, get_eager, set_eager, clear=true)
    (STREAMING, get_old_streaming, set_old_streaming, clear=true)
    (NEW_STREAMING, get_streaming, set_streaming, clear=true)
}
