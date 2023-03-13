mod convert;
mod tree;

pub(crate) use convert::insert_streaming_nodes;

type IsSink = bool;
// a rhs of a join will be replaced later
type IsRhsJoin = bool;

const IS_SINK: bool = true;
const IS_RHS_JOIN: bool = true;
