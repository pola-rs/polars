use polars_core::prelude::{AnyValue, Column, DataType, Field};
use polars_core::scalar::Scalar;
use polars_error::{PolarsResult, polars_err};
use polars_utils::pl_str::PlSmallStr;

use super::{AnonymousColumnsUdf, Expr, OpaqueColumnUdf};
use crate::prelude::{FunctionOptions, new_column_udf};

/// Represents a user-defined function
#[derive(Clone)]
pub struct UserDefinedFunction {
    /// name
    pub name: PlSmallStr,
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
    pub fn new(name: PlSmallStr, fun: impl AnonymousColumnsUdf + 'static) -> Self {
        Self {
            name,
            fun: new_column_udf(fun),
            options: FunctionOptions::default(),
        }
    }

    /// creates a logical expression with a call of the UDF
    pub fn call(self, args: Vec<Expr>) -> Expr {
        Expr::AnonymousFunction {
            input: args,
            function: self.fun,
            options: self.options,
            fmt_str: Box::new(PlSmallStr::EMPTY),
        }
    }
}

/// Try to infer the output datatype of a UDF.
///
/// This will call the UDF in a few ways and see if it can get an output type without erroring.
pub fn infer_udf_output_dtype(
    f: &dyn Fn(&[Column]) -> PolarsResult<Column>,
    input_fields: &[Field],
) -> Option<DataType> {
    // NOTE! It is important that this does not start having less capability as that would mess
    // API. We can add more passes though.

    // Pass 1: Provide default values for all columns.
    {
        let numeric_to_one = true; // A lot of functions error on 0, just give a 1.
        let num_list_values = 1; // Give at least 1 value, so UDFs have something to go off.
        let params = input_fields
            .iter()
            .map(|f| {
                let av = AnyValue::default_value(f.dtype(), numeric_to_one, num_list_values);
                let scalar = Scalar::new(f.dtype().clone(), av);

                // Give each column with 2 dummy values.
                Column::new_scalar(f.name().clone(), scalar, 2)
            })
            .collect::<Vec<_>>();

        if let Ok(c) = f(&params) {
            return Some(c.dtype().clone());
        }
    }

    None
}

/// Try to infer the output datatype of a UDF.
///
/// This will call the UDF in a few ways and see if it can get an output type without erroring.
pub fn try_infer_udf_output_dtype(
    f: &dyn Fn(&[Column]) -> PolarsResult<Column>,
    input_fields: &[Field],
) -> PolarsResult<DataType> {
    infer_udf_output_dtype(f, input_fields).ok_or_else(||
        polars_err!(
            InvalidOperation:
            "UDF called without return type, but was not able to infer the output type.\n\nThis used to be allowed but lead to unpredictable results. To fix this problem, either provide a return datatype or execute the UDF in an eager context (e.g. in `map_columns`)."
        )
    )
}
