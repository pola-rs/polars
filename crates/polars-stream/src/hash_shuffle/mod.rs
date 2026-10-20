//! Moving rows to hash partitions: frames that spill together with their hashed keys, row
//! indices grouped by partition, building partitions from batches of frames, and partitions
//! stored as frames. Frames are tagged with the index of the input they came from.
pub mod batches;
pub mod frame_with_keys;
pub mod split;
pub mod store;
