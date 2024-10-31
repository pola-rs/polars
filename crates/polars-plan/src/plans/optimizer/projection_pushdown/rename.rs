use std::collections::BTreeSet;

use polars_utils::pl_str::PlSmallStr;

use super::*;

fn iter_and_update_nodes(
    existing: &str,
    new: &str,
    acc_projections: &mut [ColumnNode],
    expr_arena: &mut Arena<AExpr>,
    processed: &mut BTreeSet<usize>,
) {
    for column_node in acc_projections.iter_mut() {
        let node = column_node.0;
        if !processed.contains(&node.0) {
            // We walk the query backwards, so we rename new to existing
            if column_node_to_name(*column_node, expr_arena) == new {
                let new_node = expr_arena.add(AExpr::Column(PlSmallStr::from_str(existing)));
                *column_node = ColumnNode(new_node);
                processed.insert(new_node.0);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn process_rename(
    acc_projections: &mut [ColumnNode],
    projected_names: &mut PlHashSet<PlSmallStr>,
    expr_arena: &mut Arena<AExpr>,
    existing: &[PlSmallStr],
    new: &[PlSmallStr],
    swapping: bool,
) -> PolarsResult<()> {
    if swapping {
        let reverse_map: PlHashMap<_, _> =
            new.iter().cloned().zip(existing.iter().cloned()).collect();
        let mut new_projected_names = PlHashSet::with_capacity(projected_names.len());

        for col in acc_projections {
            let name = column_node_to_name(*col, expr_arena);

            if let Some(previous) = reverse_map.get(name) {
                let new = expr_arena.add(AExpr::Column(previous.clone()));
                *col = ColumnNode(new);
                let _ = new_projected_names.insert(previous.clone());
            } else {
                let _ = new_projected_names.insert(name.clone());
            }
        }
        *projected_names = new_projected_names;
    } else {
        let mut processed = BTreeSet::new();
        for (existing, new) in existing.iter().zip(new.iter()) {
            if projected_names.remove(new.as_str()) {
                let name = existing.clone();
                projected_names.insert(name);
                iter_and_update_nodes(existing, new, acc_projections, expr_arena, &mut processed);
            }
        }
    }
    Ok(())
}
