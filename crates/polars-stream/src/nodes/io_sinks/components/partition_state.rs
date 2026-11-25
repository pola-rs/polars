use std::sync::Arc;

use polars_arrow::array::Utf8ViewArray;
use polars_core::frame::DataFrame;
use polars_core::frame::column::{Column, DictionaryColumn, SharedDictionary};
use polars_core::prelude::UInt32Chunked;
use polars_utils::pl_str::PlSmallStr;

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

/// An empty frame of `df`'s schema that holds none of its dictionaries, as `df.clear()` would keep
/// them alive (a partition's empty buffer would pin the dictionary of the row group it first saw).
/// An empty dictionary column adopts the dictionary of the first column appended to it.
pub fn empty_without_dictionaries(df: &DataFrame) -> DataFrame {
    if !df
        .columns()
        .iter()
        .any(|c| matches!(c, Column::Dictionary(_)))
    {
        return df.clear();
    }
    let columns = df
        .columns()
        .iter()
        .map(|c| match c {
            Column::Dictionary(d) => {
                let dict = SharedDictionary::new(Utf8ViewArray::new_empty(
                    polars_arrow::datatypes::ArrowDataType::Utf8View,
                ));
                let codes = UInt32Chunked::from_vec(PlSmallStr::EMPTY, Vec::new());
                // SAFETY: there are no rows.
                Column::Dictionary(unsafe {
                    DictionaryColumn::new_unchecked(d.name().clone(), dict, codes)
                })
            },
            c => c.clear(),
        })
        .collect();
    // SAFETY: every column is empty.
    unsafe { DataFrame::new_unchecked(0, columns) }
}
