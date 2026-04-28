mod cache_states;
mod csee;
pub mod cspe;

use cache_states::set_cache_states;
pub(super) use csee::CommonSubExprOptimizer;
pub use csee::NaiveExprMerger;

use super::*;

type Accepted = Option<(VisitRecursion, bool)>;
// Don't allow this node in a cse.
const REFUSE_NO_MEMBER: Accepted = Some((VisitRecursion::Continue, false));
// Don't allow this node, but allow as a member of a cse.
const REFUSE_ALLOW_MEMBER: Accepted = Some((VisitRecursion::Continue, true));
const REFUSE_SKIP: Accepted = Some((VisitRecursion::Skip, false));
// Accept this node.
const ACCEPT: Accepted = None;

pub(super) struct CommonSubPlanOptimizer {}

impl CommonSubPlanOptimizer {
    pub fn new() -> Self {
        Self {}
    }

    #[allow(clippy::too_many_arguments)]
    pub fn optimize(
        &mut self,
        root: Node,
        ir_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
        pushdown_maintain_errors: bool,
        opt_flags: &OptFlags,
        verbose: bool,
        scratch: &mut Vec<Node>,
    ) -> PolarsResult<Node> {
        let inserted_cache = cspe::common_subplan_elimination(root, ir_arena, expr_arena);

        run_projection_predicate_pushdown(
            root,
            ir_arena,
            expr_arena,
            pushdown_maintain_errors,
            opt_flags,
        )?;

        if inserted_cache {
            // We only want to run this on cse inserted caches
            cse::set_cache_states(
                root,
                ir_arena,
                expr_arena,
                scratch,
                verbose,
                pushdown_maintain_errors,
                opt_flags.new_streaming(),
            )?;

            use std::ops::ControlFlow;

            use crate::traversal::tree_traversal::tree_traversal;
            use crate::traversal::visitor::{FnVisitors, SubtreeVisit};

            let mut cache_hits = PlHashMap::new();

            tree_traversal(
                root,
                ir_arena,
                &mut vec![],
                &mut vec![],
                &mut FnVisitors::new(
                    || root,
                    |key, _: &mut Arena<IR>, edges| {
                        edges.inputs().for_each_mut(|e| {
                            *e = key;
                        });

                        ControlFlow::Continue(SubtreeVisit::Visit)
                    },
                    |node, ir_arena, edges| {
                        if let IR::Cache { input, id } = ir_arena.get(node) {
                            use hashbrown::hash_map::Entry;

                            match cache_hits.entry(*id) {
                                Entry::Vacant(e) => {
                                    e.insert(Some((*input, edges.outputs()[0])));
                                },
                                Entry::Occupied(mut e) => {
                                    e.get_mut().take();
                                },
                            }
                        }

                        ControlFlow::<()>::Continue(())
                    },
                ),
            )
            .continue_value()
            .unwrap();

            for (cache_input_node, cache_consumer_node) in cache_hits.into_values().flatten() {
                *ir_arena
                    .get_mut(cache_consumer_node)
                    .inputs_mut()
                    .next()
                    .unwrap() = cache_input_node;
            }
        }

        Ok(root)
    }
}
