//! This module defines the function registry and user defined functions.
use polars_arrow::error::{polars_bail, PolarsResult};
use polars_core::prelude::Field;
use polars_core::schema::Schema;
use polars_core::series::Series;
pub use polars_plan::prelude::{Context, FunctionOptions};
use polars_plan::prelude::{Expr, FunctionOutputField, GetOutput, SeriesUdf};
/// A registry that holds user defined functions.
pub trait FunctionRegistry {
    /// Register a function.
    fn register(&mut self, name: &str, fun: &dyn UserDefinedFunction) -> PolarsResult<()>;
    /// Call a user defined function.
    fn call(&self, name: &str, args: Vec<Expr>) -> PolarsResult<Expr>;
    /// Check if a function is registered.
    fn contains(&self, name: &str) -> bool;
}

/// A default registry that does not support registering or calling functions.
pub struct DefaultFunctionRegistry {}

impl FunctionRegistry for DefaultFunctionRegistry {
    fn register(&mut self, _name: &str, _fun: &dyn UserDefinedFunction) -> PolarsResult<()> {
        polars_bail!(ComputeError: "'register' not implemented on DefaultFunctionRegistry'")
    }

    fn call(&self, _name: &str, _args: Vec<Expr>) -> PolarsResult<Expr> {
        polars_bail!(ComputeError: "'call_udf' not implemented on DefaultFunctionRegistry'")
    }
    fn contains(&self, _name: &str) -> bool {
        false
    }
}

/// Represents a user-defined function that can be used within the system.
///
/// This trait provides several methods to describe and manipulate the user-defined function.
/// This is to be used with the [`FunctionRegistry`].
pub trait UserDefinedFunction {
    /// Casts the function to a generic `Any` type.
    ///
    /// This method allows for dynamic type checking at runtime.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Retrieves the output data type of the user-defined function.
    fn output_type(&self) -> GetOutput;
    /// options of the function
    fn options(&self) -> FunctionOptions {
        FunctionOptions::default()
    }
    /// Calls the user-defined function with the provided series as input.
    fn call_udf(&self, s: &mut [Series]) -> PolarsResult<Option<Series>>;
    /// Attempts to serialize the user-defined function into a byte buffer.
    ///
    /// This method provides a default implementation that raises an error,
    /// since serialization is not always supported.
    ///
    /// #### Arguments
    ///
    /// * `buf`: A mutable reference to a byte vector where the serialized data will be stored.
    ///
    /// #### Returns
    ///
    /// A result indicating the success or failure of the serialization.
    fn try_serialize(&self, _buf: &mut Vec<u8>) -> PolarsResult<()> {
        polars_bail!(ComputeError: "serialize not supported for this 'opaque' function")
    }
}

impl FunctionOutputField for dyn UserDefinedFunction + Send + Sync {
    fn get_field(&self, input_schema: &Schema, cntxt: Context, fields: &[Field]) -> Field {
        self.output_type().get_field(input_schema, cntxt, fields)
    }
}

impl SeriesUdf for dyn UserDefinedFunction + Send + Sync {
    fn call_udf(&self, s: &mut [Series]) -> PolarsResult<Option<Series>> {
        self.call_udf(s)
    }

    fn try_serialize(&self, _buf: &mut Vec<u8>) -> PolarsResult<()> {
        self.try_serialize(_buf)
    }

    fn get_output(&self) -> Option<GetOutput> {
        Some(self.output_type())
    }
}
