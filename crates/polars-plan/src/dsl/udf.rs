use polars_utils::pl_str::PlSmallStr;

use super::{ColumnsUdf, Expr, GetOutput, OpaqueColumnUdf};
use crate::prelude::{FunctionOptions, new_column_udf};

/// Represents a user-defined function
#[derive(Clone)]
pub struct UserDefinedFunction {
    /// name
    pub name: PlSmallStr,
    /// The function output type.
    pub return_type: GetOutput,
    /// The function implementation.
    pub fun: OpaqueColumnUdf,
    /// Options for the function.
    pub options: FunctionOptions,
}

impl std::fmt::Debug for UserDefinedFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("UserDefinedFunction")
            .field("name", &self.name)
            .field("fun", &"<FUNC>")
            .field("options", &self.options)
            .finish()
    }
}

impl UserDefinedFunction {
    /// Create a new UserDefinedFunction
    pub fn new(name: PlSmallStr, return_type: GetOutput, fun: impl ColumnsUdf + 'static) -> Self {
        Self {
            name,
            return_type,
            fun: new_column_udf(fun),
            options: FunctionOptions::default(),
        }
    }

    /// creates a logical expression with a call of the UDF
    pub fn call(self, args: Vec<Expr>) -> Expr {
        Expr::AnonymousFunction {
            input: args,
            function: self.fun,
            output_type: self.return_type.clone(),
            options: self.options,
        }
    }
}
