use smartstring::alias::String as SmartString;

use super::*;
use crate::prelude::optimizer::predicate_pushdown::keys::{key_has_name, predicate_to_key};

fn remove_any_key_referencing_renamed(
    new: &str,
    acc_predicates: &mut PlHashMap<Arc<str>, ExprIR>,
    local_predicates: &mut Vec<ExprIR>,
) {
    let mut move_to_local = vec![];
    for key in acc_predicates.keys() {
        if key_has_name(key, new) {
            move_to_local.push(key.clone())
        }
    }

    for key in move_to_local {
        local_predicates.push(acc_predicates.remove(&key).unwrap())
    }
}

pub(super) fn process_rename(
    acc_predicates: &mut PlHashMap<Arc<str>, ExprIR>,
    expr_arena: &mut Arena<AExpr>,
    existing: &[SmartString],
    new: &[SmartString],
) -> PolarsResult<Vec<ExprIR>> {
    let mut local_predicates = vec![];
    for (existing, new) in existing.iter().zip(new.iter()) {
        let has_existing = acc_predicates.contains_key(existing.as_str());
        let has_new = acc_predicates.contains_key(new.as_str());
        let has_both = has_existing && has_new;

        // swapping path add to local for now
        if has_both {
            // Search for the key and add it to local because swapping is more complicated
            if let Some(to_local) = acc_predicates.remove(new.as_str()) {
                local_predicates.push(to_local);
            } else {
                // The keys can be combined eg. `a AND b AND c` in this case replacing/finding
                // the key that should be renamed is more complicated, so for now
                // we just move it to local.
                remove_any_key_referencing_renamed(new, acc_predicates, &mut local_predicates)
            }
            continue;
        }
        // simple new name path
        else {
            // Find the key and update the predicate as well as the key
            // This ensure the optimization is pushed down.
            if let Some(mut e) = acc_predicates.remove(new.as_str()) {
                let new_node =
                    rename_matching_aexpr_leaf_names(e.node(), expr_arena, new, existing);
                e.set_node(new_node);
                acc_predicates.insert(predicate_to_key(new_node, expr_arena), e);
            } else {
                // The keys can be combined eg. `a AND b AND c` in this case replacing/finding
                // the key that should be renamed is more complicated, so for now
                // we just move it to local.
                remove_any_key_referencing_renamed(new, acc_predicates, &mut local_predicates)
            }
        }
    }
    Ok(local_predicates)
}
