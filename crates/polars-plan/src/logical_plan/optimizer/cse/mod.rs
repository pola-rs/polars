mod cache;
mod cse_expr;
mod cse_lp;

pub(super) use cache::decrement_file_counters_by_cache_hits;
pub(super) use cse_expr::CommonSubExprOptimizer;
pub(super) use cse_lp::elim_cmn_subplans;

use super::*;

type Accepted = Option<(VisitRecursion, bool)>;
// Don't allow this node in a cse.
const REFUSE_NO_MEMBER: Accepted = Some((VisitRecursion::Continue, false));
// Don't allow this node, but allow as a member of a cse.
const REFUSE_ALLOW_MEMBER: Accepted = Some((VisitRecursion::Continue, true));
const REFUSE_SKIP: Accepted = Some((VisitRecursion::Skip, false));
// Accept this node.
const ACCEPT: Accepted = None;
