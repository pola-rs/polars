use std::any::Any;
use std::sync::{Arc, Mutex};

use polars_core::SINGLE_LOCK;
use polars_core::query_result::QueryResult;
use polars_observer::{
    NoopQueryMetrics, PlannedQuery, QueryMetrics, QueryObserver, QueryObserverFactory,
    set_query_observer_factory,
};

use super::*;

#[derive(Debug, Clone, PartialEq)]
struct Planned {
    ir_len: usize,
    has_physical: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct Snapshot {
    rows: usize,
    total_rows_sent: u64,
    any_done: bool,
}
#[derive(Debug, Clone, PartialEq)]
enum Event {
    Started,
    Planned(Planned),
    Snapshot(Snapshot),
    Failed(String),
    Closed,
}

type Log = Arc<Mutex<Vec<Event>>>;

#[derive(Clone)]
struct TestObserver {
    log: Log,
}

impl QueryObserverFactory for TestObserver {
    fn new_observer(&self) -> Box<dyn QueryObserver> {
        Box::new(self.clone())
    }
}

impl QueryObserver for TestObserver {
    fn on_query_started(&self) {
        self.log.lock().unwrap().push(Event::Started);
    }

    fn on_query_planned(&self, query: PlannedQuery) -> Box<dyn Any + Send> {
        self.log.lock().unwrap().push(Event::Planned(Planned {
            ir_len: query.ir.len(),
            has_physical: query.physical.is_some(),
        }));
        Box::new(CloseGuard {
            log: self.log.clone(),
            metrics: query.metrics,
        })
    }

    fn on_query_failed(&self, err: &PolarsError) {
        self.log
            .lock()
            .unwrap()
            .push(Event::Failed(err.to_string()));
    }
}

struct CloseGuard {
    log: Log,
    metrics: Option<Box<dyn QueryMetrics>>,
}

impl Drop for CloseGuard {
    fn drop(&mut self) {
        if let Some(metrics) = self.metrics.as_ref() {
            let snap = metrics.snapshot();
            self.log.lock().unwrap().push(Event::Snapshot(Snapshot {
                rows: snap.len(),
                total_rows_sent: snap.iter().map(|r| r.rows_sent).sum(),
                any_done: snap.iter().any(|r| r.done),
            }));
        }
        self.log.lock().unwrap().push(Event::Closed);
    }
}

/// Run `collect` on `engine` with a fresh observer registered. `monitor`
/// toggles the `QUERY_MONITORING` opt-flag. Returns the query result and the
/// recorded event log.
fn run_observed_on(
    lf: LazyFrame,
    monitor: bool,
    engine: Engine,
) -> (PolarsResult<QueryResult>, Vec<Event>) {
    let _guard = SINGLE_LOCK.lock().unwrap();
    let log: Log = Arc::new(Mutex::new(Vec::new()));
    set_query_observer_factory(Some(Arc::new(TestObserver { log: log.clone() })));

    let lf = if monitor {
        let flags = lf.get_current_optimizations() | OptFlags::QUERY_MONITORING;
        lf.with_optimizations(flags)
    } else {
        lf
    };
    let res = lf.collect_with_engine(engine);

    set_query_observer_factory(None);
    let events = log.lock().unwrap().clone();
    (res, events)
}

mod tests {
    use super::*;

    #[test]
    fn observer_called_on_successful_query() {
        let lf = load_df().lazy().group_by([col("b")]).agg([col("a").sum()]);
        let (res, events) = run_observed_on(lf, true, Engine::Streaming);

        assert!(res.is_ok());
        assert_eq!(events.len(), 4, "unexpected event log: {events:?}");
        assert_eq!(events[0], Event::Started);
        assert!(matches!(events[1], Event::Planned(_)));
        assert!(matches!(events[2], Event::Snapshot(_)));
        assert_eq!(events[3], Event::Closed);
    }

    #[test]
    fn observer_receives_ir_and_physical() {
        let lf = load_df().lazy().group_by([col("b")]).agg([col("a").sum()]);
        let (_res, events) = run_observed_on(lf, true, Engine::Streaming);

        let planned = events
            .iter()
            .find_map(|e| match e {
                Event::Planned(planned) => Some(planned),
                _ => None,
            })
            .expect("no Planned event");
        assert!(planned.ir_len > 0, "IR description should not be empty");
        assert!(
            planned.has_physical,
            "physical plan description should be present"
        );
    }

    #[test]
    fn observer_metrics_snapshot_nonempty() {
        let lf = load_df().lazy().group_by([col("b")]).agg([col("a").sum()]);
        let (res, events) = run_observed_on(lf, true, Engine::Streaming);
        assert!(res.is_ok());

        let snapshot = events
            .iter()
            .find_map(|e| match e {
                Event::Snapshot(snapshot) => Some(snapshot),
                _ => None,
            })
            .expect("no Snapshot event");
        // One row per physical node, and the query actually moved data.
        assert!(
            snapshot.rows > 0,
            "expected one metrics row per physical node"
        );
        assert!(
            snapshot.total_rows_sent > 0,
            "expected rows to flow through nodes"
        );
        assert!(
            snapshot.any_done,
            "expected at least one node to report done"
        );
    }

    #[test]
    fn observer_failed_on_execution_error() {
        // Strict cast overflow: builds fine, errors during streaming execution.
        let df = df!("a" => [1000i64, 2000, 3000]).unwrap();
        let lf = df.lazy().select([col("a").strict_cast(DataType::Int8)]);
        let (res, events) = run_observed_on(lf, true, Engine::Streaming);

        assert!(res.is_err(), "query was expected to fail at execution");
        assert!(
            events.iter().any(|e| matches!(e, Event::Failed(_))),
            "on_query_failed should have been recorded: {events:?}"
        );
        // The guard still runs, so close fires even on failure.
        assert_eq!(events.last(), Some(&Event::Closed));
    }

    #[test]
    fn observer_failed_on_optimization_error() {
        // Referencing a missing column fails during `to_alp_optimized`, before the
        // plan reaches the streaming engine.
        let lf = load_df().lazy().select([col("does_not_exist")]);
        let (res, events) = run_observed_on(lf, true, Engine::Streaming);

        assert!(res.is_err());
        assert_eq!(events.first(), Some(&Event::Started));
        assert!(
            events.iter().any(|e| matches!(e, Event::Failed(_))),
            "expected on_query_failed on optimization error: {events:?}"
        );
        assert!(
            !events.iter().any(|e| matches!(e, Event::Planned(_))),
            "on_query_planned must not fire when planning fails: {events:?}"
        );
    }

    #[test]
    fn observer_not_called_when_flag_off() {
        let lf = load_df().lazy().group_by([col("b")]).agg([col("a").sum()]);
        let (res, events) = run_observed_on(lf, false, Engine::Streaming);

        assert!(res.is_ok());
        assert!(
            events.is_empty(),
            "observer must not fire without QUERY_MONITORING: {events:?}"
        );
    }

    #[test]
    fn observer_planned_on_in_memory_success() {
        // The in-memory engine is observed with an IR-only planned query: no
        // physical plan, no metrics, but the full started/planned/closed span.
        let lf = load_df().lazy().group_by([col("b")]).agg([col("a").sum()]);
        let (res, events) = run_observed_on(lf, true, Engine::InMemory);

        assert!(res.is_ok());
        assert_eq!(events.len(), 3, "unexpected event log: {events:?}");
        assert_eq!(events[0], Event::Started);
        let Event::Planned(planned) = &events[1] else {
            panic!("events[1] should be Planned, got {:?}", events[1]);
        };
        assert!(
            planned.ir_len > 0,
            "in-memory planned query should carry IR"
        );
        assert!(
            !planned.has_physical,
            "in-memory has no streaming physical plan"
        );
        assert_eq!(events[2], Event::Closed);
    }

    #[test]
    fn observer_failed_on_in_memory_error() {
        let lf = load_df().lazy().select([col("does_not_exist")]);
        let (res, events) = run_observed_on(lf, true, Engine::InMemory);

        assert!(res.is_err());
        assert_eq!(events.first(), Some(&Event::Started));
        assert!(events.iter().any(|e| matches!(e, Event::Failed(_))));
    }

    #[test]
    fn noop_metrics_snapshot_empty() {
        assert!(NoopQueryMetrics.snapshot().is_empty());
    }
}
