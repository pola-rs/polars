use polars_descriptions::NodeMetricsDescription;

pub trait QueryMetrics: Send + Sync {
    fn snapshot(&self) -> Vec<NodeMetricsDescription>;
}

pub struct NoopQueryMetrics;

impl QueryMetrics for NoopQueryMetrics {
    fn snapshot(&self) -> Vec<NodeMetricsDescription> {
        Vec::new()
    }
}
