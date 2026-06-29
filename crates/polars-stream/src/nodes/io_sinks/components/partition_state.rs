use std::sync::Arc;

use polars_core::frame::DataFrame;

use crate::nodes::io_sinks::components::file_sink::FileSinkTaskData;
use crate::nodes::io_sinks::components::size::RowCountAndSize;

#[derive(Default)]
pub struct PartitionState {
    pub buffered_rows: DataFrame,
    pub total_size: RowCountAndSize,
    /// Must always be <= `total_size`.
    pub sinked_size: RowCountAndSize,
    pub num_sink_opens: usize,
    pub keys_df: Arc<DataFrame>,
    pub file_sink_task_data: Option<FileSinkTaskData>,
}

impl PartitionState {
    pub fn buffered_size(&self) -> RowCountAndSize {
        let num_rows = self
            .total_size
            .num_rows
            .checked_sub(self.sinked_size.num_rows)
            .unwrap();

        if num_rows == 0 {
            return RowCountAndSize::default();
        }

        assert_eq!(
            usize::try_from(num_rows).unwrap(),
            self.buffered_rows.height()
        );

        RowCountAndSize {
            num_rows,
            num_bytes: self
                .total_size
                .num_bytes
                .saturating_sub(self.sinked_size.num_bytes),
        }
    }
}
