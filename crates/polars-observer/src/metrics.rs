use polars_descriptions::MetricsSnapshotDescription;

pub trait QueryMetricsSnapshotter: Send + Sync {
    fn snapshot(&self) -> MetricsSnapshotDescription;
}

pub struct NoopQueryMetrics;

impl QueryMetricsSnapshotter for NoopQueryMetrics {
    fn snapshot(&self) -> MetricsSnapshotDescription {
        MetricsSnapshotDescription::default()
    }
}
