#![cfg_attr(
    feature = "allow_unused",
    allow(unused, dead_code, irrefutable_let_patterns)
)] // Maybe be caused by some feature
mod executors;
pub mod function_ir;
mod planner;
mod prelude;
pub mod scan_predicate;

pub use executors::{Executor, column_to_mask};
#[cfg(feature = "python")]
pub use planner::python_scan_predicate;
pub use planner::{StreamingExecutorBuilder, create_multiple_physical_plans, create_physical_plan};

#[cfg_attr(not(feature = "dynamic_group_by"), allow(dead_code))]
pub(crate) fn unique_column_name() -> polars_utils::pl_str::PlSmallStr {
    static COUNTER: polars_utils::relaxed_cell::RelaxedCell<u64> =
        polars_utils::relaxed_cell::RelaxedCell::new_u64(0);
    let idx = COUNTER.fetch_add(1);
    polars_utils::format_pl_smallstr!("_POLARS_TMP_MEM_{idx}")
}
