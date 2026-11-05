use polars_descriptions::{IrNodeDescription, PhysicalNodeDescription};

use crate::metrics::QueryMetricsSnapshotter;

pub struct PlannedQuery {
    pub ir: Vec<IrNodeDescription>,
    pub physical: Option<Vec<PhysicalNodeDescription>>,
    pub metrics_snapshotter: Option<Box<dyn QueryMetricsSnapshotter>>,
}

impl PlannedQuery {
    pub fn new(ir: Vec<IrNodeDescription>) -> Self {
        PlannedQuery {
            ir,
            physical: None,
            metrics_snapshotter: None,
        }
    }

    pub fn with_physical(mut self, physical: Vec<PhysicalNodeDescription>) -> Self {
        self.physical = Some(physical);
        self
    }

    pub fn with_metrics_snapshotter(
        mut self,
        snapshotter: Box<dyn QueryMetricsSnapshotter>,
    ) -> Self {
        self.metrics_snapshotter = Some(snapshotter);
        self
    }
}
