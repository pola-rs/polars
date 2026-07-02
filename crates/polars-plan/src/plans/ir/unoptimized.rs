use std::fmt;
use std::sync::Arc;

use polars_core::datatypes::Field;
use polars_core::schema::{Schema, SchemaRef};
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
    /// Calls the given IRFunctionExpr with the columns selected from the inputs.
    /// The inputs do not have to be the same height.
    ColumnarFunction {
        function: IRFunctionExpr,
        options: FunctionOptions,
        output_name: PlSmallStr,
    },

    AnonymousColumnsUdf {
        function: OpaqueColumnUdf,
        options: FunctionOptions,
        output_name: PlSmallStr,
        fmt_str: Box<PlSmallStr>,
        ctx_schema: Arc<Schema>,
    },

    DynamicSlice {
        output_name: PlSmallStr,
    },
}

impl UnoptimizedOperation {
    pub fn schema(&self, inputs: &[Arc<Schema>], arg_map: &FunctionArgMap) -> Arc<Schema> {
        match self {
            UnoptimizedOperation::ColumnarFunction {
                function,
                options: _,
                output_name,
            } => {
                let input_fields: Vec<_> = arg_map.arg_fields(inputs).collect();
                let output_field = function.get_field(&input_fields).unwrap();
                Arc::new(Schema::from_iter([
                    output_field.with_name(output_name.clone())
                ]))
            },

            UnoptimizedOperation::AnonymousColumnsUdf {
                function,
                ctx_schema,
                output_name,
                ..
            } => {
                let input_fields: Vec<_> = arg_map.arg_fields(inputs).collect();
                let output_field = function
                    .clone()
                    .materialize()
                    .unwrap()
                    .get_field(ctx_schema, &input_fields)
                    .unwrap()
                    .with_name(output_name.clone());
                Arc::new(Schema::from_iter([output_field]))
            },

            UnoptimizedOperation::DynamicSlice { output_name } => {
                let fields = arg_map
                    .arg_fields(inputs)
                    .take(1)
                    .map(|f| f.with_name(output_name.clone()));
                Arc::new(Schema::from_iter(fields))
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

            Self::AnonymousColumnsUdf {
                fmt_str,
                output_name,
                ..
            } => write!(f, "{output_name} = {}(...)", fmt_str),

            Self::DynamicSlice { output_name } => write!(f, "{output_name} = dynamic-slice(...)"),
        }
    }
}

pub type InputIdx = usize;
pub type ColumnIdx = usize;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
pub struct FunctionArgMap {
    /// Mapping from function args to input columns
    /// arg[i] = inputs[arg_map[i].0].column_by_idx(arg_map[i].1).alias(arg_map[i].2)
    map: Vec<(InputIdx, ColumnIdx, PlSmallStr)>,
}

impl FunctionArgMap {
    pub fn new(map: Vec<(InputIdx, ColumnIdx, PlSmallStr)>) -> Self {
        Self { map }
    }

    /// Returns a collection of `Field`s representing the args. Renaming is applied.
    pub fn arg_fields(&self, input_schemas: &[SchemaRef]) -> impl Iterator<Item = Field> {
        self.map.iter().map(|(input_idx, column_idx, arg_name)| {
            let (_, dtype) = input_schemas[*input_idx].get_at_index(*column_idx).unwrap();
            Field::new(arg_name.clone(), dtype.clone())
        })
    }

    /// Returns a collection of `ExprIR`s that select args from zipped inputs. Renaming is applied.
    pub fn arg_selectors(
        &self,
        input_schemas: &[SchemaRef],
        expr_arena: &mut Arena<AExpr>,
    ) -> impl Iterator<Item = ExprIR> {
        self.map.iter().map(|(input, col, arg_name)| {
            let col_name = input_schemas[*input].get_at_index(*col).unwrap().0;
            AExprBuilder::col(col_name.clone(), expr_arena).expr_ir(arg_name.clone())
        })
    }

    /// Returns an iterator over `(input_idx, column_idx, arg_name)` tuples.
    pub fn iter(&self) -> impl Iterator<Item = (InputIdx, ColumnIdx, &'_ PlSmallStr)> + '_ {
        self.map.iter().map(|(c, i, name)| (*c, *i, name))
    }
}
