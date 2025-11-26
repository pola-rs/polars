mod cache_states;
mod csee;
mod cspe;

use cache_states::set_cache_states;
pub(super) use csee::CommonSubExprOptimizer;
pub use csee::NaiveExprMerger;
use cspe::elim_cmn_subplans;

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
        let (root, inserted_cache, cid2c) = cse::elim_cmn_subplans(root, ir_arena, expr_arena);

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
        }

        // The CSPE only finds the longest trail duplicates, so we must recursively apply the
        // optimization
        // Below the inserted caches, might be more duplicates. So we recurse one time to find
        // inner duplicates as well.
        // Temporarily allow partially recursive CSPE under an env var. Will be removed.
        if std::env::var("POLARS_RECURVIVE_CSPE").is_ok() {
            for (_, (_count, caches_nodes)) in cid2c.iter() {
                // The last node seems the one traversed by the planners. This is validated by tests.
                // We could traverse all nodes, but it would be duplicate work.
                if let Some(cache_or_simple_projection) = caches_nodes.last() {
                    let input = match ir_arena.get(*cache_or_simple_projection) {
                        IR::Cache { input, id: _ } => input,
                        IR::SimpleProjection { input, columns: _ } => {
                            let IR::Cache { input, id: _ } = ir_arena.get(*input) else {
                                continue;
                            };
                            input
                        },
                        _ => continue,
                    };
                    let input = *input;

                    let (sub_root, inserted_cache, _cid2c) =
                        cse::elim_cmn_subplans(input, ir_arena, expr_arena);

                    if inserted_cache {
                        run_projection_predicate_pushdown(
                            sub_root,
                            ir_arena,
                            expr_arena,
                            pushdown_maintain_errors,
                            opt_flags,
                        )?;

                        // We only want to run this on cse inserted caches
                        cse::set_cache_states(
                            sub_root,
                            ir_arena,
                            expr_arena,
                            scratch,
                            verbose,
                            pushdown_maintain_errors,
                            opt_flags.new_streaming(),
                        )?;
                    }
                }
            }
        }

        Ok(root)
    }
}
