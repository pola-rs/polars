use std::sync::Arc;

use polars_core::prelude::PolarsResult;
use polars_ops::frame::{AsOfManyOptions, AsofJoinPair};

use crate::plans::aexpr::AExpr;
use crate::plans::schema::det_join_schema;
use crate::prelude::*;

pub struct FuseAsofMany {}

impl OptimizationRule for FuseAsofMany {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> PolarsResult<Option<IR>> {
        let IR::Join {
            input_left,
            input_right,
            schema,
            left_on,
            right_on,
            options,
        } = lp_arena.get(node)
        else {
            return Ok(None);
        };

        let (top_asof, top_pairs) = match &options.args.how {
            JoinType::AsOf(top_asof) => {
                if left_on.len() != 1 || right_on.len() != 1 {
                    return Ok(None);
                }

                (
                    top_asof.as_ref(),
                    vec![AsofJoinPair {
                        left_on_name: left_on[0].output_name().clone(),
                        right_on_name: right_on[0].output_name().clone(),
                        suffix: options.args.suffix.clone(),
                    }],
                )
            },
            JoinType::AsOfMany(top_asof_many) => {
                if left_on.len() != top_asof_many.pairs.len()
                    || right_on.len() != top_asof_many.pairs.len()
                {
                    return Ok(None);
                }

                (&top_asof_many.options, top_asof_many.pairs.clone())
            },
            _ => return Ok(None),
        };

        let IR::Join {
            input_left: previous_left,
            input_right: previous_right,
            schema: _,
            left_on: prev_left_on,
            right_on: prev_right_on,
            options: prev_options,
        } = lp_arena.get(*input_left)
        else {
            return Ok(None);
        };

        let (original_left, prev_pairs) = match &prev_options.args.how {
            JoinType::AsOf(_prev_asof) => {
                if prev_left_on.len() != 1 || prev_right_on.len() != 1 {
                    return Ok(None);
                }

                (
                    *previous_left,
                    vec![AsofJoinPair {
                        left_on_name: prev_left_on[0].output_name().clone(),
                        right_on_name: prev_right_on[0].output_name().clone(),
                        suffix: prev_options.args.suffix.clone(),
                    }],
                )
            },
            JoinType::AsOfMany(prev_asof_many) => (
                *previous_left,
                prev_asof_many.pairs.clone(),
            ),
            _ => return Ok(None),
        };

        let same_right_input = if input_right == previous_right {
            true
        } else {
            matches!(
                (lp_arena.get(*input_right), lp_arena.get(*previous_right)),
                (IR::Cache { id: left_id, .. }, IR::Cache { id: right_id, .. }) if left_id == right_id
            )
        };
        if !same_right_input {
            return Ok(None);
        }

        if prev_left_on.len() != prev_pairs.len() || prev_right_on.len() != prev_pairs.len() {
            return Ok(None);
        }

        if top_asof != match &prev_options.args.how {
            JoinType::AsOf(prev_asof) => prev_asof.as_ref(),
            JoinType::AsOfMany(prev_asof_many) => &prev_asof_many.options,
            _ => unreachable!(),
        }
            || options.allow_parallel != prev_options.allow_parallel
            || options.force_parallel != prev_options.force_parallel
            || options.args.coalesce != prev_options.args.coalesce
        {
            return Ok(None);
        }

        let original_left_schema = lp_arena.get(original_left).schema(lp_arena);
        for expr in prev_left_on
            .iter()
            .cloned()
            .chain(left_on.iter().cloned())
        {
            let all_from_original_left = aexpr_to_leaf_names_iter(expr.node(), expr_arena)
                .all(|name| original_left_schema.contains(name.as_str()));
            if !all_from_original_left {
                return Ok(None);
            }
        }

        if prev_pairs.iter().any(|existing| {
            top_pairs
                .iter()
                .any(|top_pair| existing.suffix == top_pair.suffix)
        }) {
            return Ok(None);
        }

        let pairs = prev_pairs
            .iter()
            .cloned()
            .chain(top_pairs.iter().cloned())
            .collect::<Vec<_>>();

        let fused_options = JoinOptionsIR {
            allow_parallel: options.allow_parallel,
            force_parallel: options.force_parallel,
            args: JoinArgs {
                how: JoinType::AsOfMany(Box::new(AsOfManyOptions {
                    options: top_asof.clone(),
                    pairs: pairs.clone(),
                })),
                validation: options.args.validation,
                suffix: options.args.suffix.clone(),
                slice: options.args.slice,
                nulls_equal: options.args.nulls_equal,
                coalesce: options.args.coalesce,
                maintain_order: options.args.maintain_order,
                build_side: options.args.build_side.clone(),
            },
            options: options.options.clone(),
        };

        let fused_schema = det_join_schema(
            &original_left_schema,
            &lp_arena.get(*input_right).schema(lp_arena),
            &prev_left_on
                .iter()
                .cloned()
                .chain(left_on.iter().cloned())
                .collect::<Vec<_>>(),
            &prev_right_on
                .iter()
                .cloned()
                .chain(right_on.iter().cloned())
                .collect::<Vec<_>>(),
            &fused_options,
            expr_arena,
        )?;

        if fused_schema.as_ref() != schema.as_ref() {
            return Ok(None);
        }

        Ok(Some(IR::Join {
            input_left: original_left,
            input_right: *input_right,
            schema: schema.clone(),
            left_on: prev_left_on
                .iter()
                .cloned()
                .chain(left_on.iter().cloned())
                .collect(),
            right_on: prev_right_on
                .iter()
                .cloned()
                .chain(right_on.iter().cloned())
                .collect(),
            options: Arc::new(fused_options),
        }))
    }
}
