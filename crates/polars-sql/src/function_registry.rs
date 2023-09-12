//! This module defines the function registry and user defined functions.

use polars_arrow::error::{polars_bail, PolarsResult};
use polars_plan::prelude::udf::UserDefinedFunction;
use polars_plan::prelude::Expr;
pub use polars_plan::prelude::{Context, FunctionOptions};
/// A registry that holds user defined functions.
pub trait FunctionRegistry: Send + Sync {
    /// Register a function.
    fn register(&mut self, name: &str, fun: UserDefinedFunction) -> PolarsResult<()>;
    /// Call a user defined function.
    fn call(&self, name: &str, args: Vec<Expr>) -> PolarsResult<Expr>;
    /// Check if a function is registered.
    fn contains(&self, name: &str) -> bool;
}

/// A default registry that does not support registering or calling functions.
pub struct DefaultFunctionRegistry {}

impl FunctionRegistry for DefaultFunctionRegistry {
    fn register(&mut self, _name: &str, _fun: UserDefinedFunction) -> PolarsResult<()> {
        polars_bail!(ComputeError: "'register' not implemented on DefaultFunctionRegistry'")
    }

    fn call(&self, _name: &str, _args: Vec<Expr>) -> PolarsResult<Expr> {
        polars_bail!(ComputeError: "'call_udf' not implemented on DefaultFunctionRegistry'")
    }
    fn contains(&self, _name: &str) -> bool {
        false
    }
}
