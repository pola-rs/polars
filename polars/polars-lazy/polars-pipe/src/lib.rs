mod executors;
pub mod expressions;
pub mod operators;
pub mod pipeline;


// ideal chunk size we strive to
pub(crate) const CHUNK_SIZE: usize = 5;