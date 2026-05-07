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
        // Arg names for functions that use them. At the time of writing, it was
        // AsStruct, FoldHorizontal, ReduceHorizontal, CumReduceHorizontal, CumFoldHorizontal
        arg_names: Option<Vec<PlSmallStr>>,
        output_name: PlSmallStr,
    },

    AnonymousColumnsUdf {
        function: OpaqueColumnUdf,
        options: FunctionOptions,
        arg_names: Vec<PlSmallStr>,
        output_name: PlSmallStr,
        fmt_str: Box<PlSmallStr>,
        ctx_schema: Arc<Schema>,
    },
}

impl UnoptimizedOperation {
    pub fn schema(&self, inputs: &[Arc<Schema>], arg_map: &FunctionArgMap) -> Arc<Schema> {
        match self {
            UnoptimizedOperation::ColumnarFunction {
                function,
                options: _,
                arg_names: input_names,
                output_name,
            } => {
                let input_fields = arg_map.arg_fields(inputs, input_names.as_deref());
                let output_field = function.get_field(&input_fields).unwrap();
                Arc::new(Schema::from_iter([
                    output_field.with_name(output_name.clone())
                ]))
            },

            UnoptimizedOperation::AnonymousColumnsUdf {
                function,
                arg_names: input_names,
                ctx_schema,
                ..
            } => {
                let input_fields = arg_map.arg_fields(inputs, Some(input_names));
                let output_field = function
                    .clone()
                    .materialize()
                    .unwrap()
                    .get_field(ctx_schema, &input_fields)
                    .unwrap();
                Arc::new(Schema::from_iter([output_field]))
            },
        }
    }

    pub fn arg_names(&self) -> Option<&[PlSmallStr]> {
        match self {
            Self::ColumnarFunction { arg_names, .. } => arg_names.as_deref(),
            Self::AnonymousColumnsUdf { arg_names, .. } => Some(arg_names),
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
        }
    }
}

pub type InputIdx = usize;
pub type ColumnIdx = usize;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
pub struct FunctionArgMap {
    /// Mapping from function args to input columns
    /// arg[i] = inputs[arg_map[i].0].column_by_idx(arg_map[i].1)
    map: Vec<(InputIdx, ColumnIdx)>,
}

impl FunctionArgMap {
    pub fn new(map: Vec<(InputIdx, ColumnIdx)>) -> Self {
        Self { map }
    }

    /// Returns a collection of `Field`s representing the args. Renaming is applied.
    pub fn arg_fields(
        &self,
        input_schemas: &[SchemaRef],
        arg_names: Option<&[PlSmallStr]>,
    ) -> Vec<Field> {
        if let Some(arg_names) = arg_names {
            assert_eq!(self.map.len(), arg_names.len());
        }
        self.map
            .iter()
            .enumerate()
            .map(|(i, &(input_idx, column_idx))| {
                let (input_name, dtype) =
                    input_schemas[input_idx].get_at_index(column_idx).unwrap();
                let arg_name = arg_names.map(|a| &a[i]).unwrap_or(input_name).clone();
                Field::new(arg_name, dtype.clone())
            })
            .collect()
    }

    /// Returns a collection of `ExprIR`s that select args from zipped inputs. Renaming is applied.
    pub fn arg_selectors(
        &self,
        input_schemas: &[SchemaRef],
        arg_names: Option<&[PlSmallStr]>,
        expr_arena: &mut Arena<AExpr>,
    ) -> Vec<ExprIR> {
        self.iter_args_with_names(input_schemas, arg_names)
            .map(|(_, col_name, arg_name)| {
                AExprBuilder::col(col_name.clone(), expr_arena).expr_ir(arg_name.clone())
            })
            .collect()
    }

    /// Returns an iterator over `(input_idx, input_col_name, arg_name)` tuples.
    pub fn iter_args_with_names<'a>(
        &'a self,
        input_schemas: &'a [SchemaRef],
        arg_names: Option<&'a [PlSmallStr]>,
    ) -> impl Iterator<Item = (InputIdx, &'a PlSmallStr, &'a PlSmallStr)> + 'a {
        if let Some(arg_names) = arg_names {
            assert_eq!(self.map.len(), arg_names.len());
        }
        self.map.iter().enumerate().map(move |(i, (input, col))| {
            let col_name = input_schemas[*input].get_at_index(*col).unwrap().0;
            let arg_name = arg_names.map(|r| &r[i]).unwrap_or(col_name);
            (*input, col_name, arg_name)
        })
    }

    /// Returns an iterator over `(input_idx, column_idx)` tuples.
    pub fn iter_arg_inputs(&self) -> impl Iterator<Item = (InputIdx, ColumnIdx)> + '_ {
        self.map.iter().copied()
    }
}
