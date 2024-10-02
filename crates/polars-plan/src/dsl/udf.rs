use arrow::legacy::error::{polars_bail, PolarsResult};
use polars_core::prelude::Field;
use polars_core::schema::Schema;
use polars_utils::pl_str::PlSmallStr;

use super::{ColumnsUdf, Expr, GetOutput, OpaqueColumnUdf};
use crate::prelude::{new_column_udf, Context, FunctionOptions};

/// Represents a user-defined function
#[derive(Clone)]
pub struct UserDefinedFunction {
    /// name
    pub name: PlSmallStr,
    /// The function signature.
    pub input_fields: Vec<Field>,
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
            .field("signature", &self.input_fields)
            .field("fun", &"<FUNC>")
            .field("options", &self.options)
            .finish()
    }
}

impl UserDefinedFunction {
    /// Create a new UserDefinedFunction
    pub fn new(
        name: PlSmallStr,
        input_fields: Vec<Field>,
        return_type: GetOutput,
        fun: impl ColumnsUdf + 'static,
    ) -> Self {
        Self {
            name,
            input_fields,
            return_type,
            fun: new_column_udf(fun),
            options: FunctionOptions::default(),
        }
    }

    /// creates a logical expression with a call of the UDF
    /// This utility allows using the UDF without requiring access to the registry.
    /// The schema is validated and the query will fail if the schema is invalid.
    pub fn call(self, args: Vec<Expr>) -> PolarsResult<Expr> {
        if args.len() != self.input_fields.len() {
            polars_bail!(InvalidOperation: "expected {} arguments, got {}", self.input_fields.len(), args.len())
        }
        let schema = Schema::from_iter(self.input_fields);

        if args
            .iter()
            .map(|e| e.to_field(&schema, Context::Default))
            .collect::<PolarsResult<Vec<_>>>()
            .is_err()
        {
            polars_bail!(InvalidOperation: "unexpected field in UDF \nexpected: {:?}\n received {:?}", schema, args)
        };

        Ok(Expr::AnonymousFunction {
            input: args,
            function: self.fun,
            output_type: self.return_type,
            options: self.options,
        })
    }

    /// creates a logical expression with a call of the UDF
    /// This does not do any schema validation and is therefore faster.
    ///
    /// Only use this if you are certain that the schema is correct.
    /// If the schema is invalid, the query will fail at runtime.
    pub fn call_unchecked(self, args: Vec<Expr>) -> Expr {
        Expr::AnonymousFunction {
            input: args,
            function: self.fun,
            output_type: self.return_type.clone(),
            options: self.options,
        }
    }
}
