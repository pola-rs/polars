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
        members: &mut MemberCollector,
        pushdown_maintain_errors: bool,
        opt_flags: &OptFlags,
        verbose: bool,
        scratch: &mut Vec<Node>,
    ) -> PolarsResult<Node> {
        let (root, inserted_cache) = cse::elim_cmn_subplans(root, ir_arena, expr_arena);

        run_projection_predicate_pushdown(
            root,
            ir_arena,
            expr_arena,
            pushdown_maintain_errors,
            opt_flags,
        )?;

        if (members.has_joins_or_unions | members.has_sink_multiple) && inserted_cache {
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

        Ok(root)
    }
}
