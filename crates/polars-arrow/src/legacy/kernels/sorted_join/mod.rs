pub mod inner;
pub mod left;

use std::fmt::Debug;

pub use polars_utils::total_ord::{TotalEq, TotalOrd};
use polars_utils::{IdxSize, NullableIdxSize};

/// What the two sides of a sorted-region merge are compared with.
///
/// The keys reach these kernels in the order Polars sorted them, which is a *total* order:
/// every `NaN` sorts alongside every other and after every number, and `-0.0` sorts alongside
/// `0.0`. `PartialOrd` is not that order — `NaN` compares equal to nothing and less than
/// nothing, so a merge driven by it walks past a run of them without ever matching, and a join
/// on a sorted float key silently drops every `NaN` row. [`TotalOrd`] is the order the input is
/// actually in.
pub trait SortedJoinKey: TotalOrd + Copy + Debug {}
impl<T: TotalOrd + Copy + Debug> SortedJoinKey for T {}

type JoinOptIds = Vec<NullableIdxSize>;
type JoinIds = Vec<IdxSize>;
type LeftJoinIds = (JoinIds, JoinOptIds);
type InnerJoinIds = (JoinIds, JoinIds);
