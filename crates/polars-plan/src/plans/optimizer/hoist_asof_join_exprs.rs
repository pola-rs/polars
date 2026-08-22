use polars_core::prelude::*;
use polars_ops::frame::JoinType;
use polars_utils::arena::{Arena, Node};

use super::{OptimizationRule, pushdown_maintain_errors};
use crate::plans::{
    AExpr, ExprPushdownGroup, FunctionFlags, FunctionIR, IR, IRBuilder,
    is_inherently_nondeterministic,
};
use crate::prelude::Operator;
use crate::utils::aexpr_to_leaf_names_iter;

pub(super) struct HoistAsOfJoinExpressions;

impl OptimizationRule for HoistAsOfJoinExpressions {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> PolarsResult<Option<IR>> {
        let IR::Join {
            input_left,
            input_right,
            schema: original_schema,
            left_on,
            right_on,
            options: join_options,
        } = lp_arena.get(node)
        else {
            return Ok(None);
        };

        let JoinType::AsOf(asof_options) = &join_options.args.how else {
            return Ok(None);
        };

        let IR::HStack {
            input: right_input,
            exprs,
            options: hstack_options,
            ..
        } = lp_arena.get(*input_right)
        else {
            return Ok(None);
        };

        let input_left = *input_left;
        let right_input = *right_input;
        let original_schema = original_schema.clone();
        let left_on = left_on.clone();
        let right_on = right_on.clone();
        let join_options = join_options.clone();
        let hstack_options = *hstack_options;
        let exprs = exprs.clone();

        let (Some(left_rows), Some(right_rows)) = (
            estimated_row_count(input_left, lp_arena),
            estimated_row_count(right_input, lp_arena),
        ) else {
            return Ok(None);
        };
        if left_rows >= right_rows {
            return Ok(None);
        }

        let left_schema = lp_arena.get(input_left).schema(lp_arena).into_owned();
        let right_input_schema = lp_arena.get(right_input).schema(lp_arena).into_owned();

        let mut required_right_names = PlHashSet::new();
        for expr in &right_on {
            required_right_names.extend(aexpr_to_leaf_names_iter(expr.node(), expr_arena).cloned());
        }
        if let Some(right_by) = asof_options.right_by.as_deref() {
            required_right_names.extend(right_by.iter().cloned());
        }

        let mut output_names = PlHashSet::with_capacity(exprs.len());
        if !exprs
            .iter()
            .all(|expr| output_names.insert(expr.output_name().clone()))
        {
            return Ok(None);
        }

        let maintain_errors = pushdown_maintain_errors();
        let mut hoist = vec![false; exprs.len()];

        for (idx, expr) in exprs.iter().enumerate() {
            let output_name = expr.output_name();
            if required_right_names.contains(output_name)
                || left_schema.contains(output_name)
                || is_inherently_nondeterministic(expr.node(), expr_arena)
                || !is_null_on_unmatched_right(expr.node(), expr_arena)
            {
                continue;
            }

            let mut leaf_names = aexpr_to_leaf_names_iter(expr.node(), expr_arena);
            if leaf_names.any(|name| {
                !right_input_schema.contains(name)
                    || left_schema.contains(name)
                    || required_right_names.contains(name)
            }) {
                continue;
            }

            let mut pushdown_group = ExprPushdownGroup::Pushable;
            pushdown_group.update_with_expr_rec(expr_arena.get(expr.node()), expr_arena, None);
            if !pushdown_group.blocks_pushdown(maintain_errors) {
                hoist[idx] = true;
            }
        }

        // Splitting an HStack changes which version of an overwritten column a moved
        // expression sees. Keep such expressions below the join.
        loop {
            let retained_outputs = exprs
                .iter()
                .zip(&hoist)
                .filter(|(_, hoist)| !**hoist)
                .map(|(expr, _)| expr.output_name().clone())
                .collect::<PlHashSet<_>>();
            let mut changed = false;
            for (expr, hoist) in exprs.iter().zip(&mut hoist) {
                if *hoist
                    && aexpr_to_leaf_names_iter(expr.node(), expr_arena)
                        .any(|name| retained_outputs.contains(name))
                {
                    *hoist = false;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        if !hoist.iter().any(|hoist| *hoist) {
            return Ok(None);
        }

        let mut retained_exprs = Vec::with_capacity(exprs.len());
        let mut hoisted_exprs = Vec::with_capacity(exprs.len());
        for (expr, hoist) in exprs.into_iter().zip(hoist) {
            if hoist {
                hoisted_exprs.push(expr);
            } else {
                retained_exprs.push(expr);
            }
        }

        let new_right_input = if retained_exprs.is_empty() {
            right_input
        } else {
            IRBuilder::new(right_input, expr_arena, lp_arena)
                .with_columns(retained_exprs, hstack_options)
                .node()
        };

        let join = IRBuilder::new(input_left, expr_arena, lp_arena)
            .join(new_right_input, left_on, right_on, join_options)
            .build();
        let builder = IRBuilder::from_lp(join, expr_arena, lp_arena)
            .with_columns(hoisted_exprs, hstack_options);
        let new_schema = builder.schema().into_owned();
        let hoisted = builder.build();

        if new_schema.len() != original_schema.len()
            || original_schema.iter().any(|(name, dtype)| {
                new_schema
                    .get(name)
                    .is_none_or(|new_dtype| new_dtype != dtype)
            })
        {
            return Ok(None);
        }

        if new_schema == original_schema {
            Ok(Some(hoisted))
        } else {
            Ok(Some(IR::SimpleProjection {
                input: lp_arena.add(hoisted),
                columns: original_schema,
            }))
        }
    }
}

fn estimated_row_count(node: Node, lp_arena: &Arena<IR>) -> Option<usize> {
    match lp_arena.get(node) {
        IR::DataFrameScan { df, .. } => Some(df.height()),
        IR::Scan { file_info, .. } if file_info.row_estimation.1 != usize::MAX => {
            Some(file_info.row_estimation.1)
        },
        IR::Slice { input, len, .. } => {
            estimated_row_count(*input, lp_arena).map(|rows| rows.min(*len as usize))
        },
        IR::SimpleProjection { input, .. }
        | IR::HStack { input, .. }
        | IR::Sort { input, .. }
        | IR::Cache { input, .. } => estimated_row_count(*input, lp_arena),
        IR::MapFunction {
            input,
            function: FunctionIR::Hint(_),
        } => estimated_row_count(*input, lp_arena),
        _ => None,
    }
}

fn is_null_on_unmatched_right(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    match expr_arena.get(node) {
        AExpr::Column(_) => true,
        AExpr::Literal(value) => value.is_null(),
        AExpr::BinaryExpr { left, op, right } if binary_operator_preserves_nulls(*op) => {
            is_null_on_unmatched_right(*left, expr_arena)
                || is_null_on_unmatched_right(*right, expr_arena)
        },
        AExpr::Cast { expr, dtype, .. } => {
            !dtype.is_nested() && is_null_on_unmatched_right(*expr, expr_arena)
        },
        AExpr::Function { input, options, .. } => {
            if options
                .flags
                .contains(FunctionFlags::PRESERVES_NULL_FIRST_INPUT)
            {
                input
                    .first()
                    .is_some_and(|expr| is_null_on_unmatched_right(expr.node(), expr_arena))
            } else if options
                .flags
                .contains(FunctionFlags::PRESERVES_NULL_ALL_INPUTS)
            {
                input
                    .iter()
                    .any(|expr| is_null_on_unmatched_right(expr.node(), expr_arena))
            } else {
                false
            }
        },
        _ => false,
    }
}

fn binary_operator_preserves_nulls(op: Operator) -> bool {
    use crate::prelude::Operator::*;

    matches!(
        op,
        Eq | NotEq
            | Lt
            | LtEq
            | Gt
            | GtEq
            | Plus
            | Minus
            | Multiply
            | RustDivide
            | TrueDivide
            | FloorDivide
            | Modulus
            | Xor
    )
}
