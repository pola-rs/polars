use std::collections::BTreeSet;

use smartstring::alias::String as SmartString;

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
            if column_node_to_name(*column_node, expr_arena).as_ref() == new {
                let new_node = expr_arena.add(AExpr::Column(ColumnName::from(existing)));
                *column_node = ColumnNode(new_node);
                processed.insert(new_node.0);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn process_rename(
    acc_projections: &mut [ColumnNode],
    projected_names: &mut PlHashSet<Arc<str>>,
    expr_arena: &mut Arena<AExpr>,
    existing: &[SmartString],
    new: &[SmartString],
    swapping: bool,
) -> PolarsResult<()> {
    let mut processed = BTreeSet::new();
    if swapping {
        // We clone otherwise we update a data structure whilst we rename it.
        let mut new_projected_names = projected_names.clone();
        for (existing, new) in existing.iter().zip(new.iter()) {
            let has_existing = projected_names.contains(existing.as_str());
            // Only if the new column name is projected by the upper node we must update the name.
            let has_new = projected_names.contains(new.as_str());
            let has_both = has_existing && has_new;

            if has_new {
                // swapping path
                // this must leave projected names intact, as we only swap
                if has_both {
                    iter_and_update_nodes(
                        existing,
                        new,
                        acc_projections,
                        expr_arena,
                        &mut processed,
                    );
                }
                // simple new name path
                // this must add and remove names
                else {
                    new_projected_names.remove(new.as_str());
                    let name = ColumnName::from(existing.as_str());
                    new_projected_names.insert(name);
                    iter_and_update_nodes(
                        existing,
                        new,
                        acc_projections,
                        expr_arena,
                        &mut processed,
                    );
                }
            }
        }
        *projected_names = new_projected_names;
    } else {
        for (existing, new) in existing.iter().zip(new.iter()) {
            if projected_names.remove(new.as_str()) {
                let name: Arc<str> = ColumnName::from(existing.as_str());
                projected_names.insert(name);
                iter_and_update_nodes(existing, new, acc_projections, expr_arena, &mut processed);
            }
        }
    }
    Ok(())
}
