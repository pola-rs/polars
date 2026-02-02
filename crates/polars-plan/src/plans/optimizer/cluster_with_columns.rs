use std::sync::Arc;

use polars_core::prelude::{PlHashSet, PlIndexMap};
use polars_utils::aliases::InitHashMaps;
use polars_utils::arena::{Arena, Node};

use super::aexpr::AExpr;
use super::ir::IR;
use super::{PlSmallStr, aexpr_to_leaf_names_iter};
use crate::plans::ExprIR;

pub fn optimize(root: Node, lp_arena: &mut Arena<IR>, expr_arena: &Arena<AExpr>) {
    let mut ir_stack = Vec::with_capacity(16);
    ir_stack.push(root);

    let mut input_name_to_expr_map: PlIndexMap<PlSmallStr, ExprIR> = PlIndexMap::new();
    let mut accessed_input_names: PlHashSet<PlSmallStr> = PlHashSet::new();
    let mut push_candidate_idxs: Vec<usize> = vec![];
    let mut new_current_exprs: Vec<ExprIR> = vec![];
    let mut visited_caches = PlHashSet::new();

    while let Some(current) = ir_stack.pop() {
        let current_ir = lp_arena.get(current);

        if let IR::Cache { id, .. } = current_ir {
            if !visited_caches.insert(*id) {
                continue;
            }
        }

        current_ir.copy_inputs(&mut ir_stack);

        let IR::HStack { input, .. } = current_ir else {
            continue;
        };

        let input = *input;

        let [current_ir, input_ir] = lp_arena.get_many_mut([current, input]);

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
        accessed_input_names.clear();
        push_candidate_idxs.clear();
        new_current_exprs.clear();

        input_name_to_expr_map.extend(
            input_exprs
                .iter()
                .map(|e| (e.output_name().clone(), e.clone())),
        );

        if input_name_to_expr_map.len() != input_exprs.len() {
            if cfg!(debug_assertions) {
                panic!()
            };

            continue;
        }

        for (i, e) in current_exprs.iter().enumerate() {
            let mut accessed_upper_expr = false;

            for name in aexpr_to_leaf_names_iter(e.node(), expr_arena) {
                if input_name_to_expr_map.contains_key(name) {
                    accessed_upper_expr = true;
                    accessed_input_names.insert(name.clone());
                }
            }

            if !accessed_upper_expr {
                push_candidate_idxs.push(i);
            }
        }

        let mut candidate_idx: usize = 0;

        for (i, e) in current_exprs.iter().enumerate() {
            if push_candidate_idxs.get(candidate_idx) == Some(&i) {
                candidate_idx += 1;

                if !accessed_input_names.contains(e.output_name())
                    && aexpr_to_leaf_names_iter(e.node(), expr_arena)
                        .all(|name| !accessed_input_names.contains(name))
                {
                    input_name_to_expr_map.insert(e.output_name().clone(), e.clone());
                    continue;
                }
            }

            new_current_exprs.push(e.clone());
        }

        if new_current_exprs.len() == current_exprs.len() {
            continue;
        }

        input_exprs.clear();

        for (output_name, e) in input_name_to_expr_map
            .iter()
            .map(|x| (x.0.clone(), x.1.clone()))
        {
            input_exprs.push(e);

            if !input_schema.contains(&output_name) {
                let dtype = current_schema.get(&output_name).unwrap().clone();
                Arc::make_mut(input_schema).insert(output_name, dtype);
            }
        }

        if new_current_exprs.is_empty() {
            let input_ir = input_ir.clone();
            lp_arena.replace(current, input_ir);
            *ir_stack.last_mut().unwrap() = current;
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

            let current_ir = lp_arena.replace(current, IR::Invalid);
            let moved_current_node = lp_arena.add(current_ir);
            lp_arena.replace(
                current,
                IR::SimpleProjection {
                    input: moved_current_node,
                    columns: projection,
                },
            );
        }
    }
}
