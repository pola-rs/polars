pub mod inner;
pub mod left;

use std::fmt::Debug;

use crate::index::IdxSize;

type JoinOptIds = Vec<Option<IdxSize>>;
type JoinIds = Vec<IdxSize>;
type LeftJoinIds = (JoinIds, JoinOptIds);
type InnerJoinIds = (JoinIds, JoinIds);
