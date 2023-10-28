//! This module defines the function registry and user defined functions.

use polars_error::{polars_bail, PolarsResult};
use polars_plan::prelude::udf::UserDefinedFunction;
pub use polars_plan::prelude::{Context, FunctionOptions};
/// A registry that holds user defined functions.
pub trait FunctionRegistry: Send + Sync {
    /// Register a function.
    fn register(&mut self, name: &str, fun: UserDefinedFunction) -> PolarsResult<()>;
    /// Call a user defined function.
    fn get_udf(&self, name: &str) -> PolarsResult<Option<UserDefinedFunction>>;
    /// Check if a function is registered.
    fn contains(&self, name: &str) -> bool;
}

/// A default registry that does not support registering or calling functions.
pub struct DefaultFunctionRegistry {}

impl FunctionRegistry for DefaultFunctionRegistry {
    fn register(&mut self, _name: &str, _fun: UserDefinedFunction) -> PolarsResult<()> {
        polars_bail!(ComputeError: "'register' not implemented on DefaultFunctionRegistry'")
    }

    fn get_udf(&self, _name: &str) -> PolarsResult<Option<UserDefinedFunction>> {
        polars_bail!(ComputeError: "'get_udf' not implemented on DefaultFunctionRegistry'")
    }
    fn contains(&self, _name: &str) -> bool {
        false
    }
}
