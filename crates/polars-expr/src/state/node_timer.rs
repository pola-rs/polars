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
    pub(super) fn new(query_start: Instant) -> Self {
        Self {
            query_start,
            data: Arc::new(Mutex::new((Vec::with_capacity(16), Vec::with_capacity(16)))),
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
        let mut nodes = std::mem::take(&mut data.0);
        let mut ticks = std::mem::take(&mut data.1);

        if ticks.is_empty() {
            let schema = Schema::from_iter(vec![
                Field::new(PlSmallStr::from_static("node"), DataType::String),
                Field::new(PlSmallStr::from_static("start"), DataType::UInt64),
                Field::new(PlSmallStr::from_static("end"), DataType::UInt64),
            ]);
            return PolarsResult::Ok(DataFrame::empty_with_schema(&schema));
        }

        nodes.push("optimization".to_string());

        // first value is end of optimization
        let start = ticks[0].0;
        ticks.push((Duration::from_nanos(0), start));
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

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    #[test]
    fn test_empty_timer() {
        let start_time = Instant::now();
        let timer = NodeTimer::new(start_time);

        let result = timer.finish();
        assert!(result.is_ok());

        let df = result.unwrap();
        assert_eq!(df.height(), 0);
        assert_eq!(df.width(), 3);

        let expected_columns = vec!["node", "start", "end"];
        let actual_columns: Vec<&str> = df.get_column_names().iter().map(|s| s.as_str()).collect();
        assert_eq!(actual_columns, expected_columns);

        assert_eq!(
            df.dtypes(),
            vec![DataType::String, DataType::UInt64, DataType::UInt64]
        );
    }

    #[test]
    fn test_timer_with_data() {
        let start_time = Instant::now();
        let timer = NodeTimer::new(start_time);

        timer.store_duration(
            Duration::from_millis(100),
            Duration::from_millis(200),
            "test_node".to_string(),
        );

        let result = timer.finish();
        assert!(result.is_ok());

        let df = result.unwrap();
        assert_eq!(df.height(), 2); // test_node + optimization
        assert_eq!(df.width(), 3);
    }
}
