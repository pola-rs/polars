use std::any::Any;
use std::sync::Arc;

use parking_lot::RwLock;
use polars_descriptions::{IrNodeDescription, NodeMetricsDescription, PhysicalNodeDescription};
use polars_error::PolarsError;

pub type QueryExecutionGuard = Box<dyn Any + Send>;

pub trait QueryMetrics: Send + Sync {
    fn snapshot(&self) -> Vec<NodeMetricsDescription>;
}

pub struct NoopQueryMetrics;

impl QueryMetrics for NoopQueryMetrics {
    fn snapshot(&self) -> Vec<NodeMetricsDescription> {
        Vec::new()
    }
}

pub struct PlannedQuery {
    pub ir: Vec<IrNodeDescription>,
    pub physical: Option<Vec<PhysicalNodeDescription>>,
    pub metrics: Option<Box<dyn QueryMetrics>>,
}

impl PlannedQuery {
    pub fn builder(ir: Vec<IrNodeDescription>) -> PlannedQueryBuilder {
        PlannedQueryBuilder {
            ir,
            physical: None,
            metrics: None,
        }
    }
}

pub struct PlannedQueryBuilder {
    ir: Vec<IrNodeDescription>,
    physical: Option<Vec<PhysicalNodeDescription>>,
    metrics: Option<Box<dyn QueryMetrics>>,
}

impl PlannedQueryBuilder {
    pub fn physical(mut self, physical: Vec<PhysicalNodeDescription>) -> Self {
        self.physical = Some(physical);
        self
    }

    pub fn metrics(mut self, metrics: Box<dyn QueryMetrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    pub fn build(self) -> PlannedQuery {
        PlannedQuery {
            ir: self.ir,
            physical: self.physical,
            metrics: self.metrics,
        }
    }
}

pub trait QueryObserver: Send {
    /// Signals that the query has been submitted by the user and is about to start planning.
    fn on_query_started(&self);

    /// Signals that the query has finished planning and is about to start executing.
    ///
    /// The returned [`QueryExecutionGuard`] is held for the duration of execution
    /// and dropped once the query finishes, whether it succeeds or fails.
    fn on_query_planned(&self, query: PlannedQuery) -> QueryExecutionGuard;

    /// Signals that the query has failed.
    ///
    /// If the query had already been planned, this is called before its
    /// [`QueryExecutionGuard`] is dropped.
    fn on_query_failed(&self, err: &PolarsError);
}

pub trait QueryObserverFactory: Send + Sync {
    fn new_observer(&self) -> Box<dyn QueryObserver>;
}

static FACTORY: RwLock<Option<Arc<dyn QueryObserverFactory>>> = RwLock::new(None);

pub fn set_query_observer_factory(factory: Option<Arc<dyn QueryObserverFactory>>) {
    *FACTORY.write() = factory;
}

pub fn observer() -> Option<Box<dyn QueryObserver>> {
    FACTORY.read().as_ref().map(|f| f.new_observer())
}
