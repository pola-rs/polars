use polars_core::error::{PolarsResult, polars_ensure};
use polars_core::prelude::DataType;
use polars_utils::arena::{Arena, Node};

use super::{AExpr, IR, OptimizationRule};
use crate::plans::Context;
use crate::plans::conversion::get_schema;

pub struct TypeCheckRule;

impl OptimizationRule for TypeCheckRule {
    fn optimize_plan(
        &mut self,
        ir_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> PolarsResult<Option<IR>> {
        let ir = ir_arena.get(node);
        match ir {
            IR::Scan {
                predicate: Some(predicate),
                ..
            } => {
                let input_schema = get_schema(ir_arena, node);
                let dtype = predicate.dtype(input_schema.as_ref(), Context::Default, expr_arena)?;

                polars_ensure!(
                    matches!(dtype, DataType::Boolean | DataType::Unknown(_)),
                    InvalidOperation: "filter predicate must be of type `Boolean`, got `{dtype:?}`"
                );

                Ok(None)
            },
            IR::Filter { predicate, .. } => {
                let input_schema = get_schema(ir_arena, node);
                let dtype = predicate.dtype(input_schema.as_ref(), Context::Default, expr_arena)?;

                polars_ensure!(
                    matches!(dtype, DataType::Boolean | DataType::Unknown(_)),
                    InvalidOperation: "filter predicate must be of type `Boolean`, got `{dtype:?}`"
                );

                Ok(None)
            },
            _ => Ok(None),
        }
    }
}
