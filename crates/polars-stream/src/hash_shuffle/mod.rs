//! Moving rows to hash partitions: frames that spill together with their hashed keys, row
//! indices grouped by partition, building partitions from batches of frames, and partitions
//! stored as frames. Frames are tagged with the index of the input they came from.
pub(crate) mod batches;
pub(crate) mod frame_with_keys;
pub(crate) mod parallel;
pub(crate) mod split;
pub(crate) mod store;
