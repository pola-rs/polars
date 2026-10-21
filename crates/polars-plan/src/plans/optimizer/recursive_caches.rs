#[allow(clippy::disallowed_types)]
use polars_utils::aliases::PlHashSet;
use polars_utils::arena::{Arena, Node};
use polars_utils::unique_id::UniqueId;

use crate::plans::IR;

/// The caches in the steps of recursive queries that are computed less often than their
/// readers run: a cache is computed again in every round only if the innermost recursive query
/// around it lists it in its `round_local_caches`.
#[allow(clippy::disallowed_types)] // We don't iterate over it.
pub(super) fn caches_shared_across_rounds(root: Node, ir_arena: &Arena<IR>) -> PlHashSet<UniqueId> {
    let mut out = PlHashSet::default();
    let mut seen_caches = PlHashSet::default();
    let mut stack: Vec<(Node, Option<&[UniqueId]>)> = vec![(root, None)];
    while let Some((node, round_local)) = stack.pop() {
        match ir_arena.get(node) {
            IR::Cache { input, id } => {
                if round_local.is_some_and(|caches| !caches.contains(id)) {
                    out.insert(*id);
                }
                if seen_caches.insert(*id) {
                    stack.push((*input, round_local));
                }
            },
            IR::Recursive {
                anchor,
                step,
                round_local_caches,
                ..
            } => {
                stack.push((*anchor, round_local));
                stack.push((*step, Some(round_local_caches)));
            },
            ir => stack.extend(ir.inputs().map(|input| (input, round_local))),
        }
    }
    out
}
