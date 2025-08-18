mod async_executor;
mod async_primitives;
mod skeleton;

use std::sync::LazyLock;

pub use skeleton::{run_query, visualize_physical_plan};

mod execute;
pub(crate) mod expression;
mod graph;
pub use skeleton::{QueryResult, StreamingQuery};
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
