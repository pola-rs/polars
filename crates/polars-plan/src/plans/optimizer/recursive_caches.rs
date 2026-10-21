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
    visit_step_caches(root, ir_arena, |id, round_local| {
        if !round_local.contains(&id) {
            out.insert(id);
        }
    });
    out
}

/// The caches that are read in the step of a recursive query.
#[cfg(feature = "cse")]
#[allow(clippy::disallowed_types)] // We don't iterate over it.
pub(super) fn caches_in_recursive_steps(root: Node, ir_arena: &Arena<IR>) -> PlHashSet<UniqueId> {
    let mut out = PlHashSet::default();
    visit_step_caches(root, ir_arena, |id, _| {
        out.insert(id);
    });
    out
}

/// The innermost recursive query around a node: its id and its round-local caches.
type Recursion<'a> = (UniqueId, &'a [UniqueId]);

/// Calls `f` for every read of a cache in the step of a recursive query, with the
/// `round_local_caches` of the innermost recursive query around it.
fn visit_step_caches(root: Node, ir_arena: &Arena<IR>, mut f: impl FnMut(UniqueId, &[UniqueId])) {
    #[allow(clippy::disallowed_types)] // We don't iterate over it.
    let mut seen_caches = PlHashSet::default();
    let mut stack: Vec<(Node, Option<Recursion>)> = vec![(root, None)];
    while let Some((node, recursion)) = stack.pop() {
        match ir_arena.get(node) {
            IR::Cache { input, id } => {
                if let Some((_, round_local)) = recursion {
                    f(*id, round_local);
                }
                // The input is walked once per recursive query around it, as each can protect
                // other caches in it.
                if seen_caches.insert((*id, recursion.map(|(rec_id, _)| rec_id))) {
                    stack.push((*input, recursion));
                }
            },
            IR::Recursive {
                anchor,
                step,
                id,
                round_local_caches,
            } => {
                stack.push((*anchor, recursion));
                stack.push((*step, Some((*id, round_local_caches))));
            },
            ir => stack.extend(ir.inputs().map(|input| (input, recursion))),
        }
    }
}
