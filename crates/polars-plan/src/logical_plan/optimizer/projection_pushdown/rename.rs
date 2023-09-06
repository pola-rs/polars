use std::collections::BTreeSet;

use smartstring::alias::String as SmartString;

use super::*;

fn iter_and_update_nodes(
    existing: &str,
    new: &str,
    acc_projections: &mut [Node],
    expr_arena: &mut Arena<AExpr>,
    processed: &mut BTreeSet<usize>,
) {
    for node in acc_projections.iter_mut() {
        if !processed.contains(&node.0) {
            let new_node = rename_matching_aexpr_leaf_names(*node, expr_arena, new, existing);
            if new_node != *node {
                *node = new_node;
                processed.insert(node.0);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn process_rename(
    acc_projections: &mut [Node],
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
                    let name: Arc<str> = Arc::from(existing.as_str());
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
                let name: Arc<str> = Arc::from(existing.as_str());
                projected_names.insert(name);
                iter_and_update_nodes(existing, new, acc_projections, expr_arena, &mut processed);
            }
        }
    }
    Ok(())
}
