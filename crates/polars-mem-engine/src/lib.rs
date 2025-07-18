#![cfg_attr(
    feature = "allow_unused",
    allow(unused, dead_code, irrefutable_let_patterns)
)] // Maybe be caused by some feature
mod executors;
mod planner;
mod predicate;
mod prelude;

pub use executors::Executor;
#[cfg(feature = "python")]
pub use planner::python_scan_predicate;
pub use planner::{
    StreamingExecutorBuilder, create_multiple_physical_plans, create_physical_plan,
    create_scan_predicate,
};
pub use predicate::ScanPredicate;
