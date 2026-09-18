pub mod inner;
pub mod left;

use std::fmt::Debug;

pub use polars_utils::total_ord::{TotalEq, TotalOrd};
use polars_utils::{IdxSize, NullableIdxSize};

/// What the two sides of a sorted-region merge are compared with.
pub trait SortedJoinKey: TotalOrd + Copy + Debug {}
impl<T: TotalOrd + Copy + Debug> SortedJoinKey for T {}

type JoinOptIds = Vec<NullableIdxSize>;
type JoinIds = Vec<IdxSize>;
type LeftJoinIds = (JoinIds, JoinOptIds);
type InnerJoinIds = (JoinIds, JoinIds);
