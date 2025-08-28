mod async_executor;
mod async_primitives;
mod skeleton;

use std::sync::LazyLock;
use async_primitives::MemoryLimiter;

pub use skeleton::{run_query, visualize_physical_plan};

mod execute;
pub(crate) mod expression;
mod graph;
pub use skeleton::{QueryResult, StreamingQuery};
mod memory_utils;
mod morsel;
mod nodes;
mod physical_plan;
mod pipe;
mod utils;

// TODO: experiment with these.
static DEFAULT_LINEARIZER_BUFFER_SIZE: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("POLARS_DEFAULT_LINEARIZER_BUFFER_SIZE")
        .map(|x| x.parse().unwrap())
        .unwrap_or(4)
});

static DEFAULT_DISTRIBUTOR_BUFFER_SIZE: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("POLARS_DEFAULT_DISTRIBUTOR_BUFFER_SIZE")
        .map(|x| x.parse().unwrap())
        .unwrap_or(4)
});

static DEFAULT_ZIP_HEAD_BUFFER_SIZE: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("POLARS_DEFAULT_ZIP_HEAD_BUFFER_SIZE")
        .map(|x| x.parse().unwrap())
        .unwrap_or(4)
});

static DEFAULT_MEMORY_LIMIT: LazyLock<usize> = LazyLock::new(|| {
    let system_memory = memory_utils::get_system_memory();
    system_memory * 8 / 10 // 80% of system memory by default
});

static MEMORY_LIMITER: LazyLock<MemoryLimiter> = LazyLock::new(|| {
    let limit = std::env::var("POLARS_STREAMING_MAX_MEMORY_BYTES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(*DEFAULT_MEMORY_LIMIT);
        
    MemoryLimiter::new(limit)
});

pub fn get_memory_limiter() -> &'static MemoryLimiter {
    &MEMORY_LIMITER
}
