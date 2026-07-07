use std::sync::Mutex;
use std::time::{Duration, Instant};

use polars_core::prelude::*;
use polars_core::utils::NoNull;

type StartInstant = Instant;
type EndInstant = Instant;

type Nodes = Vec<String>;
type Ticks = Vec<(Duration, Duration)>;

#[derive(Clone)]
pub(super) struct NodeTimer {
    query_start: Instant,
    data: Arc<Mutex<(Nodes, Ticks)>>,
}

impl NodeTimer {
    pub(super) fn new(query_start: Instant, optimization_duration: Duration) -> Self {
        let (mut nodes, mut ticks) = (Vec::with_capacity(16), Vec::with_capacity(16));
        nodes.push("optimization".to_string());
        ticks.push((Duration::ZERO, optimization_duration));
        Self {
            query_start,
            data: Arc::new(Mutex::new((nodes, ticks))),
        }
    }

    pub(super) fn store(&self, start: StartInstant, end: EndInstant, name: String) {
        self.store_duration(
            start.duration_since(self.query_start),
            end.duration_since(self.query_start),
            name,
        )
    }

    pub(super) fn store_duration(&self, start: Duration, end: Duration, name: String) {
        let mut data = self.data.lock().unwrap();
        let nodes = &mut data.0;
        nodes.push(name);
        let ticks = &mut data.1;
        ticks.push((start, end))
    }

    pub(super) fn finish(self) -> PolarsResult<DataFrame> {
        let mut data = self.data.lock().unwrap();
        let nodes = std::mem::take(&mut data.0);
        let ticks = std::mem::take(&mut data.1);
        let nodes_s = Column::new(PlSmallStr::from_static("node"), nodes);
        let start: NoNull<UInt64Chunked> = ticks
            .iter()
            .map(|(start, _)| start.as_micros() as u64)
            .collect();
        let mut start = start.into_inner();
        start.rename(PlSmallStr::from_static("start"));

        let end: NoNull<UInt64Chunked> = ticks
            .iter()
            .map(|(_, end)| end.as_micros() as u64)
            .collect();
        let mut end = end.into_inner();
        end.rename(PlSmallStr::from_static("end"));

        let height = nodes_s.len();
        let columns = vec![nodes_s, start.into_column(), end.into_column()];
        let df = unsafe { DataFrame::new_unchecked(height, columns) };
        df.sort(vec!["start"], SortMultipleOptions::default())
    }
}
