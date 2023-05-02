mod convert;
mod plumbing;

pub(crate) use convert::insert_streaming_nodes;

type IsSink = bool;
#[derive(Copy, Clone, Debug)]
enum SplitType {
    None,
    JoinRhs,
    Union,
}

const IS_SINK: bool = true;
