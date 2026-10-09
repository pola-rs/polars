use IR::*;
use polars_core::error::PolarsResult;

use super::OptimizationRule;
use crate::prelude::IR;

pub struct FlattenUnionRule {}

impl OptimizationRule for FlattenUnionRule {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut polars_utils::arena::Arena<IR>,
        _expr_arena: &mut polars_utils::arena::Arena<crate::prelude::AExpr>,
        node: polars_utils::arena::Node,
    ) -> PolarsResult<Option<IR>> {
        let lp = lp_arena.get(node);

        match lp {
            Union { inputs, options }
                if inputs.iter().any(|node| match lp_arena.get(*node) {
                    Union { options, .. } => !options.flattened_by_opt && options.slice.is_none(),
                    _ => false,
                }) =>
            {
                let mut new_inputs = Vec::with_capacity(inputs.len() * 2);
                let mut options = *options;

                for node in inputs {
                    match lp_arena.get(*node) {
                        IR::Union {
                            inputs, options, ..
                        } if options.slice.is_none() => new_inputs.extend_from_slice(inputs),
                        _ => new_inputs.push(*node),
                    }
                }
                options.flattened_by_opt = true;

                Ok(Some(Union {
                    inputs: new_inputs,
                    options,
                }))
            },
            _ => Ok(None),
        }
    }
}
