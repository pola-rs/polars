use std::fmt;
use std::sync::Arc;

use polars_core::schema::{Schema, SchemaExt};
use polars_utils::itertools::Itertools;
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "ir_serde")]
use serde::{Deserialize, Serialize};

use crate::prelude::*;

/// These IR nodes are generated to dispatch to specific functionality in the
/// streaming engine, and aren't optimized across. They should not be generated
/// directly by operations, only lowered to post-optimization.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
pub enum UnoptimizedOperation {
    /// Calls the given IRFunctionExpr with the columns of the inputs in order.
    /// The inputs do not have to be the same height.
    ColumnarFunction {
        function: IRFunctionExpr,
        options: FunctionOptions,
        output_name: PlSmallStr,
    },
}

impl UnoptimizedOperation {
    pub fn schema(&self, inputs: &[Arc<Schema>]) -> Arc<Schema> {
        match self {
            UnoptimizedOperation::ColumnarFunction {
                function,
                options: _,
                output_name,
            } => {
                let input_fields = inputs.iter().flat_map(|i| i.iter_fields()).collect_vec();
                let output_field = function.get_field(&input_fields).unwrap();
                Arc::new(Schema::from_iter([
                    output_field.with_name(output_name.clone())
                ]))
            },
        }
    }
}

impl fmt::Display for UnoptimizedOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ColumnarFunction {
                function,
                output_name,
                ..
            } => write!(f, "{output_name} = {}(...)", function),
        }
    }
}
