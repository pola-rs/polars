mod dispatch;
mod skeleton;

use std::sync::LazyLock;

pub use skeleton::{run_query, visualize_physical_plan};

mod execute;
pub use dispatch::build_streaming_query_executor;
pub(crate) mod expression;
mod graph;
pub use graph::{GraphNodeKey, LogicalPipe, LogicalPipeKey};
pub use skeleton::StreamingQuery;
mod metrics;
pub use metrics::{GraphMetrics, NodeMetrics};
mod observer_metrics;
pub use observer_metrics::StreamingQueryMetrics;
pub use polars_observer::{
    QueryMetrics, QueryObserver, QueryObserverFactory, observer, set_query_observer_factory,
};
mod morsel;
mod nodes;
mod physical_plan;
pub use physical_plan::{
    NodeStyle, PhysNode, PhysNodeKey, PhysNodeKind, ZipBehavior, physical_plan_to_description,
};
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
