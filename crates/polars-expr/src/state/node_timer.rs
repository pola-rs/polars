use std::sync::Mutex;
use std::time::Instant;

use polars_core::prelude::*;
use polars_core::utils::NoNull;

type StartInstant = Instant;
type EndInstant = Instant;

type Nodes = Vec<String>;
type Ticks = Vec<(StartInstant, EndInstant)>;
type RawTicks = Vec<(u64, u64)>;

#[derive(Clone)]
pub struct NodeTimer {
    pub query_start: Instant,
    data: Arc<Mutex<(Nodes, Ticks, RawTicks)>>,
}

impl NodeTimer {
    pub fn new() -> Self {
        Self {
            query_start: Instant::now(),
            data: Arc::new(Mutex::new((
                Vec::with_capacity(16),
                Vec::with_capacity(16),
                Vec::with_capacity(16),
            ))),
        }
    }

    pub fn store(&self, start: StartInstant, end: EndInstant, name: String) {
        let mut data = self.data.lock().unwrap();
        let nodes = &mut data.0;
        nodes.push(name);
        let ticks = &mut data.1;
        ticks.push((start, end))
    }

    pub fn store_raw(&self, start: u64, end: u64, name: String) {
        let mut data = self.data.lock().unwrap();
        let nodes = &mut data.0;
        nodes.push(name);
        let ticks = &mut data.2;
        ticks.push((start, end))
    }

    pub fn finish(self) -> PolarsResult<DataFrame> {
        let mut data = self.data.lock().unwrap();
        let mut nodes = std::mem::take(&mut data.0);
        nodes.push("optimization".to_string());

        let mut ticks = std::mem::take(&mut data.1);
        let mut raw_ticks = std::mem::take(&mut data.2);

        polars_ensure!(
            !ticks.is_empty() || !raw_ticks.is_empty(),
            ComputeError: "no data to time"
        );

        let (start_times, end_times): (Vec<u64>, Vec<u64>) = if !ticks.is_empty() {
            let start = ticks[0].0;
            ticks.push((self.query_start, start));

            let start_times = ticks
                .iter()
                .map(|(start, _)| start.duration_since(self.query_start).as_micros() as u64)
                .collect();

            let end_times = ticks
                .iter()
                .map(|(_, end)| end.duration_since(self.query_start).as_micros() as u64)
                .collect();

            (start_times, end_times)
        } else {
            // Use raw timestamps directly
            let start = raw_ticks[0].0;
            raw_ticks.push((0, start));
            raw_ticks.iter().cloned().unzip()
        };

        let nodes_s = Column::new(PlSmallStr::from_static("node"), nodes);

        let start_col: NoNull<UInt64Chunked> = start_times.into_iter().collect();
        let mut start_col = start_col.into_inner();
        start_col.rename(PlSmallStr::from_static("start"));

        let end_col: NoNull<UInt64Chunked> = end_times.into_iter().collect();
        let mut end_col = end_col.into_inner();
        end_col.rename(PlSmallStr::from_static("end"));

        let height = nodes_s.len();
        let columns = vec![nodes_s, start_col.into_column(), end_col.into_column()];

        let df = unsafe { DataFrame::new_no_checks(height, columns) };
        df.sort(vec!["start"], SortMultipleOptions::default())
    }
}

impl Default for NodeTimer {
    fn default() -> Self {
        Self::new()
    }
}
