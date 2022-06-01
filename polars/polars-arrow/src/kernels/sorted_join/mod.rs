mod left;
mod right;

use crate::index::IdxSize;
use std::fmt::Debug;

type JoinOptIds = Vec<Option<IdxSize>>;
type JoinIds = Vec<IdxSize>;
type LeftJoinIds = (JoinIds, JoinOptIds);
type InnerJoinIds = (JoinIds, JoinIds);
