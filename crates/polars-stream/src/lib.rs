#![allow(unused)] // TODO: remove.

mod async_executor;
#[allow(unused)] // TODO: remove.
mod async_primitives;
mod skeleton;

pub use skeleton::run_query;

mod execute;
mod graph;
mod morsel;
mod nodes;
mod physical_plan;
mod utils;

// TODO: experiment with these, and make them configurable through environment variables.
const DEFAULT_LINEARIZER_BUFFER_SIZE: usize = 4;
const DEFAULT_DISTRIBUTOR_BUFFER_SIZE: usize = 4;
