use polars_descriptions::NodeMetricsDescription;

pub trait QueryMetricsSnapshotter: Send + Sync {
    fn snapshot(&self) -> Vec<NodeMetricsDescription>;
}

pub struct NoopQueryMetrics;

impl QueryMetricsSnapshotter for NoopQueryMetrics {
    fn snapshot(&self) -> Vec<NodeMetricsDescription> {
        Vec::new()
    }
}
