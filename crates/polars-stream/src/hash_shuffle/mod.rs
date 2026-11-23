//! Moving rows to hash partitions: frames that spill together with their hashed keys, row
//! indices grouped by partition, building partitions from batches of frames, partitions stored
//! as frames, and writing the rows of each partition into frames as they arrive. Frames are
//! tagged with the index of the input they came from.
pub mod batches;
pub mod frame_with_keys;
pub mod split;
pub mod store;
pub mod writer;
