pub mod inner;
pub mod left;

use std::fmt::Debug;

use polars_utils::{IdxSize, NullableIdxSize};

type JoinOptIds = Vec<NullableIdxSize>;
type JoinIds = Vec<IdxSize>;
type LeftJoinIds = (JoinIds, JoinOptIds);
type InnerJoinIds = (JoinIds, JoinIds);
