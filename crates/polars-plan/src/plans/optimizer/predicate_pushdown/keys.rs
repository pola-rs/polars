//! Keys in the `acc_predicates` hashmap.
use super::*;

// an invisible ascii token we use as delimiter
const HIDDEN_DELIMITER: char = '\u{1D17A}';

/// Determine the hashmap key by combining all the leaf column names of a predicate
pub(super) fn predicate_to_key(predicate: Node, expr_arena: &Arena<AExpr>) -> Arc<str> {
    let mut iter = aexpr_to_leaf_names_iter(predicate, expr_arena);
    if let Some(first) = iter.next() {
        if let Some(second) = iter.next() {
            let mut new = String::with_capacity(32 * iter.size_hint().0);
            new.push_str(&first);
            new.push(HIDDEN_DELIMITER);
            new.push_str(&second);

            for name in iter {
                new.push(HIDDEN_DELIMITER);
                new.push_str(&name);
            }
            return Arc::from(new);
        }
        first
    } else {
        let mut s = String::new();
        s.push(HIDDEN_DELIMITER);
        Arc::from(s)
    }
}

pub(super) fn key_has_name(key: &str, name: &str) -> bool {
    if key.contains(HIDDEN_DELIMITER) {
        for root_name in key.split(HIDDEN_DELIMITER) {
            if root_name == name {
                return true;
            }
        }
    }
    key == name
}
