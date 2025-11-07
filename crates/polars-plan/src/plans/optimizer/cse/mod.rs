mod cache_states;
mod csee;
mod cspe;

pub(super) use cache_states::set_cache_states;
pub(super) use csee::CommonSubExprOptimizer;
pub use csee::NaiveExprMerger;
pub(super) use cspe::elim_cmn_subplans;

use super::*;

type Accepted = Option<(VisitRecursion, bool)>;
// Don't allow this node in a cse.
const REFUSE_NO_MEMBER: Accepted = Some((VisitRecursion::Continue, false));
// Don't allow this node, but allow as a member of a cse.
const REFUSE_ALLOW_MEMBER: Accepted = Some((VisitRecursion::Continue, true));
const REFUSE_SKIP: Accepted = Some((VisitRecursion::Skip, false));
// Accept this node.
const ACCEPT: Accepted = None;
