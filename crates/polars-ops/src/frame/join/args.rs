use super::*;

pub(super) type JoinIds = Vec<IdxSize>;
pub type LeftJoinIds = (ChunkJoinIds, ChunkJoinOptIds);
pub type InnerJoinIds = (JoinIds, JoinIds);

#[cfg(feature = "chunked_ids")]
pub(super) type ChunkJoinIds = Either<Vec<IdxSize>, Vec<ChunkId>>;
#[cfg(feature = "chunked_ids")]
pub type ChunkJoinOptIds = Either<Vec<NullableIdxSize>, Vec<ChunkId>>;

#[cfg(not(feature = "chunked_ids"))]
pub type ChunkJoinOptIds = Vec<NullableIdxSize>;

#[cfg(not(feature = "chunked_ids"))]
pub type ChunkJoinIds = Vec<IdxSize>;
