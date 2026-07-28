use std::sync::Arc;

use polars_core::prelude::{PlIndexMap, PlIndexSet};
use polars_utils::aliases::InitHashMaps;
use polars_utils::arena::{Arena, Node};
use polars_utils::scratch_vec::ScratchVec;

use super::aexpr::AExpr;
use super::ir::IR;
use super::{PlSmallStr, aexpr_to_leaf_names_iter};
use crate::plans::ExprIR;
use crate::plans::projection_height::aexpr_projection_height_rec;
use crate::prelude::projection_height::ExprProjectionHeight;

enum ExprSource {
    Original { non_scalar_output_height: bool },
    Replaced,
}

pub fn optimize(root: Node, lp_arena: &mut Arena<IR>, expr_arena: &Arena<AExpr>) {
    use ExprProjectionHeight as EH;

    let mut ir_stack = Vec::with_capacity(16);
    ir_stack.push(root);

    // key: output_name, value: (expr, _)
    let mut input_name_to_expr_map: PlIndexMap<PlSmallStr, (ExprIR, ExprSource)> =
        PlIndexMap::new();
    let mut input_names_accessed_by_non_candidates: PlIndexSet<PlSmallStr> = PlIndexSet::new();
    let mut push_candidate_idxs: Vec<usize> = vec![];
    let mut new_current_exprs: Vec<ExprIR> = vec![];
    let mut visited_caches = PlIndexSet::new();

    let mut ae_nodes_stack = ScratchVec::default();
    let mut ae_heights_stack = ScratchVec::default();

    while let Some(current_node) = ir_stack.pop() {
        let current_ir = lp_arena.get(current_node);

        if let IR::Cache { id, .. } = current_ir {
            if !visited_caches.insert(*id) {
                continue;
            }
        }

        current_ir.copy_inputs(&mut ir_stack);

        let IR::HStack { input, .. } = current_ir else {
            continue;
        };

        let input_node = *input;

        let [current_ir, input_ir] = lp_arena.get_disjoint_mut([current_node, input_node]);

        let IR::HStack {
            input: _,
            exprs: current_exprs,
            schema: current_schema,
            options: _,
        } = current_ir
        else {
            unreachable!();
        };

        let IR::HStack {
            input: _,
            exprs: input_exprs,
            schema: input_schema,
            options: _,
        } = input_ir
        else {
            continue;
        };

        input_name_to_expr_map.clear();
        input_names_accessed_by_non_candidates.clear();
        push_candidate_idxs.clear();
        new_current_exprs.clear();
        let mut input_non_scalar_output_height_count: usize = 0;

        input_name_to_expr_map.extend(input_exprs.iter().map(|e| {
            let non_scalar_output_height = !matches!(
                aexpr_projection_height_rec(
                    e.node(),
                    expr_arena,
                    &mut ae_nodes_stack,
                    &mut ae_heights_stack
                ),
                EH::Scalar
            );

            if non_scalar_output_height {
                input_non_scalar_output_height_count += 1;
            }

            (
                e.output_name().clone(),
                (
                    e.clone(),
                    ExprSource::Original {
                        non_scalar_output_height,
                    },
                ),
            )
        }));

        if input_name_to_expr_map.len() != input_exprs.len() {
            if cfg!(debug_assertions) {
                panic!()
            };

            continue;
        }

        for (i, e) in current_exprs.iter().enumerate() {
            // Ignore col()
            if let AExpr::Column(name) = expr_arena.get(e.node())
                && name == e.output_name()
            {
                continue;
            }

            if aexpr_to_leaf_names_iter(e.node(), expr_arena)
                .all(|name| !input_name_to_expr_map.contains_key(name))
            {
                push_candidate_idxs.push(i);
            }
        }

        let mut candidate_idx: usize = 0;

        for (i, e) in current_exprs.iter().enumerate() {
            if push_candidate_idxs.get(candidate_idx) == Some(&i) {
                candidate_idx += 1;
                continue;
            }

            for name in aexpr_to_leaf_names_iter(e.node(), expr_arena) {
                input_names_accessed_by_non_candidates.insert(name.clone());
            }
        }

        push_candidate_idxs.retain(|&i| {
            let e = &current_exprs[i];
            !input_names_accessed_by_non_candidates.contains(e.output_name())
        });

        // E.g. `LazyFrame().with_columns(a=int_range(5)).with_columns(a=1)`, we cannot prune the int_range as
        // otherwise the query may succeed with 1 row instead of erroring.
        let mut last_match_idx: usize = 0;
        if input_non_scalar_output_height_count != 0
            && push_candidate_idxs.len() >= input_non_scalar_output_height_count
            && push_candidate_idxs
                .iter()
                .map(|&i| {
                    let e = &current_exprs[i];

                    let would_replace_non_scalar_with_scalar = matches!(
                        input_name_to_expr_map.get(e.output_name()),
                        Some((
                            _,
                            ExprSource::Original {
                                non_scalar_output_height: true
                            }
                        ))
                    ) && matches!(
                        aexpr_projection_height_rec(
                            e.node(),
                            expr_arena,
                            &mut ae_nodes_stack,
                            &mut ae_heights_stack
                        ),
                        EH::Scalar
                    );

                    if would_replace_non_scalar_with_scalar {
                        last_match_idx = i;
                    }

                    usize::from(would_replace_non_scalar_with_scalar)
                })
                .sum::<usize>()
                == input_non_scalar_output_height_count
        {
            push_candidate_idxs.remove(last_match_idx);
        }

        let mut candidate_idx: usize = 0;

        for (i, e) in current_exprs.iter().enumerate() {
            // Prune col()
            if let AExpr::Column(name) = expr_arena.get(e.node())
                && name == e.output_name()
            {
                continue;
            }

            if push_candidate_idxs.get(candidate_idx) == Some(&i) {
                candidate_idx += 1;
                input_name_to_expr_map
                    .insert(e.output_name().clone(), (e.clone(), ExprSource::Replaced));
                continue;
            }

            new_current_exprs.push(e.clone());
        }

        if new_current_exprs.len() == current_exprs.len() {
            continue;
        }

        input_exprs.clear();

        for (output_name, (e, e_src)) in input_name_to_expr_map.iter().map(|x| (x.0.clone(), x.1)) {
            input_exprs.push(e.clone());

            if !matches!(e_src, ExprSource::Original { .. }) {
                let dtype = current_schema.get(&output_name).unwrap().clone();
                Arc::make_mut(input_schema).insert(output_name, dtype);
            }
        }

        if new_current_exprs.is_empty() {
            let input_ir = input_ir.clone();
            lp_arena.replace(current_node, input_ir);
            *ir_stack.last_mut().unwrap() = current_node;
            continue;
        }

        let fix_output_order = current_exprs.iter().any(|e| {
            input_schema
                .index_of(e.output_name())
                .is_some_and(|i| i != current_schema.index_of(e.output_name()).unwrap())
        });

        current_exprs.clear();
        std::mem::swap(current_exprs, &mut new_current_exprs);

        if fix_output_order {
            let projection = current_schema.clone();

            Arc::make_mut(current_schema)
                .sort_by_key(|name, _| input_schema.index_of(name).unwrap_or(usize::MAX));

            let current_ir = lp_arena.replace(current_node, IR::Invalid);
            let moved_current_node = lp_arena.add(current_ir);
            lp_arena.replace(
                current_node,
                IR::SimpleProjection {
                    input: moved_current_node,
                    columns: projection,
                },
            );
        }
    }
}
