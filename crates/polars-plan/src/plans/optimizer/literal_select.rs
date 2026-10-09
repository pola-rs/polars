use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::*;
use polars_utils::arena::{Arena, Node};

use super::OptimizationRule;
use crate::prelude::*;

/// Replaces a select of only scalar literals with a scan of its one row, so that the input does
/// not run.
pub struct LiteralSelect {}

impl OptimizationRule for LiteralSelect {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> PolarsResult<Option<IR>> {
        let IR::Select { expr, schema, .. } = lp_arena.get(node) else {
            return Ok(None);
        };
        if expr.is_empty() {
            return Ok(None);
        }

        let mut columns = Vec::with_capacity(expr.len());
        for (e, (name, dtype)) in expr.iter().zip(schema.iter()) {
            let scalar = match expr_arena.get(e.node()) {
                AExpr::Literal(LiteralValue::Scalar(sc)) => Some(sc.clone()),
                AExpr::Literal(LiteralValue::Dyn(d)) => d
                    .clone()
                    .try_materialize_to_dtype(dtype, CastOptions::Strict)
                    .ok(),
                _ => None,
            };
            let Some(scalar) = scalar.filter(|sc| sc.dtype() == dtype) else {
                return Ok(None);
            };
            columns.push(Column::new_scalar(name.clone(), scalar, 1));
        }

        Ok(Some(IR::DataFrameScan {
            df: Arc::new(DataFrame::new(1, columns)?),
            schema: schema.clone(),
            output_schema: None,
        }))
    }
}
