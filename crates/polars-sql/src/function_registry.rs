//! This module defines the function registry and user defined functions.
use std::sync::Arc;

use polars_arrow::error::{polars_bail, PolarsResult};
use polars_core::prelude::Field;
use polars_core::schema::Schema;
use polars_core::series::Series;
pub use polars_plan::prelude::{Context, FunctionOptions};
use polars_plan::prelude::{Expr, FunctionOutputField, GetOutput, SeriesUdf};
/// A registry that holds user defined functions.
pub trait FunctionRegistry: Send + Sync {
    /// Register a function.
    fn register(&mut self, name: &str, fun: UserDefinedFunction) -> PolarsResult<()>;
    /// Call a function.
    fn call_udf(&self, name: &str, args: Vec<Expr>) -> PolarsResult<Expr>;

    /// Check if a function is registered.
    fn contains(&self, name: &str) -> bool;
}

/// A default registry that does not support registering or calling functions.
pub struct DefaultFunctionRegistry {}
impl FunctionRegistry for DefaultFunctionRegistry {
    fn register(&mut self, _name: &str, _fun: UserDefinedFunction) -> PolarsResult<()> {
        polars_bail!(ComputeError: "'register' not implemented on DefaultFunctionRegistry'")
    }

    fn call_udf(&self, _name: &str, _args: Vec<Expr>) -> PolarsResult<Expr> {
        polars_bail!(ComputeError: "'call_udf' not implemented on DefaultFunctionRegistry'")
    }
    fn contains(&self, _name: &str) -> bool {
        false
    }
}
/// A wrapper struct for a user defined function.
pub struct UserDefinedFunction {
    /// function to apply
    pub function: Arc<dyn SeriesUdf>,
    /// output dtype of the function
    #[cfg_attr(feature = "serde", serde(skip))]
    pub output_type: GetOutput,
    /// options of the function
    pub options: FunctionOptions,
}

impl FunctionOutputField for UserDefinedFunction {
    fn get_field(&self, input_schema: &Schema, cntxt: Context, fields: &[Field]) -> Field {
        self.output_type.get_field(input_schema, cntxt, fields)
    }
}

impl SeriesUdf for UserDefinedFunction {
    fn call_udf(&self, s: &mut [Series]) -> PolarsResult<Option<Series>> {
        self.function.call_udf(s)
    }
}
