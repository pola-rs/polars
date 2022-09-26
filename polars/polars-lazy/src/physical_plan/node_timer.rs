use std::sync::{Arc, Mutex};
use std::time::Instant;

use polars_core::prelude::*;
use polars_core::utils::NoNull;

type StartInstant = Instant;
type EndInstant = Instant;

type Nodes = Vec<String>;
type Ticks = Vec<(StartInstant, EndInstant)>;

#[derive(Clone)]
pub(super) struct NodeTimer {
    query_start: Instant,
    data: Arc<Mutex<(Nodes, Ticks)>>,
}

impl NodeTimer {
    pub(super) fn new() -> Self {
        Self {
            query_start: Instant::now(),
            data: Arc::new(Mutex::new((Vec::with_capacity(16), Vec::with_capacity(16)))),
        }
    }

    pub(super) fn store(&self, start: StartInstant, end: EndInstant, name: String) {
        let mut data = self.data.lock().unwrap();
        let nodes = &mut data.0;
        nodes.push(name);
        let ticks = &mut data.1;
        ticks.push((start, end))
    }

    pub(super) fn finish(self) -> PolarsResult<DataFrame> {
        let mut data = self.data.lock().unwrap();
        let mut nodes = std::mem::take(&mut data.0);
        nodes.push("optimization".to_string());

        let mut ticks = std::mem::take(&mut data.1);
        // first value is end of optimization
        if ticks.is_empty() {
            return Err(PolarsError::ComputeError("no data to time".into()));
        } else {
            let start = ticks[0].0;
            ticks.push((self.query_start, start))
        }
        let nodes_s = Series::new("node", nodes);
        let start: NoNull<UInt64Chunked> = ticks
            .iter()
            .map(|(start, _)| (start.duration_since(self.query_start)).as_micros() as u64)
            .collect();
        let mut start = start.into_inner();
        start.rename("start");

        let end: NoNull<UInt64Chunked> = ticks
            .iter()
            .map(|(_, end)| (end.duration_since(self.query_start)).as_micros() as u64)
            .collect();
        let mut end = end.into_inner();
        end.rename("end");

        DataFrame::new_no_checks(vec![nodes_s, start.into_series(), end.into_series()])
            .sort(vec!["start"], vec![false])
    }
}
