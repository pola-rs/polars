mod metrics;
mod planned_query;

use std::any::Any;
use std::sync::Arc;

pub use metrics::*;
use parking_lot::RwLock;
pub use planned_query::*;
use polars_error::PolarsError;

pub type QueryExecutionGuard = Box<dyn Any + Send>;

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
    /// If the query had already been planned, this has to be called before its
    /// [`QueryExecutionGuard`] is dropped.
    fn on_query_failed(&self, err: &PolarsError);
}

pub trait QueryObserverFactory: Send + Sync {
    fn new_observer(&self) -> Box<dyn QueryObserver>;
}

static QUERY_OBSERVER_FACTORY: RwLock<Option<Arc<dyn QueryObserverFactory>>> = RwLock::new(None);

pub fn register_query_observer_factory(factory: Option<Arc<dyn QueryObserverFactory>>) {
    *QUERY_OBSERVER_FACTORY.write() = factory;
}

/// Creates a 'unique' instances of the [QueryObserver] for each query.
pub fn new_query_observer() -> Option<Box<dyn QueryObserver>> {
    QUERY_OBSERVER_FACTORY
        .read()
        .as_ref()
        .map(|f| f.new_observer())
}
