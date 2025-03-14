use std::collections::VecDeque;

use futures::StreamExt;
use polars_error::PolarsResult;
use polars_io::RowIndex;
use polars_utils::IdxSize;
use polars_utils::slice_enum::Slice;

use crate::async_executor::{self, TaskPriority};
use crate::nodes::io_sources::multi_file_reader::MultiFileReaderConfig;
use crate::nodes::io_sources::multi_file_reader::reader_interface::FileReader;

pub struct ResolvedPositiveSliceInfo {
    pub scan_source_idx: usize,
    pub row_index: Option<RowIndex>,
    /// This should always be positive slice.
    pub pre_slice: Option<Slice>,
    /// This will be in-order - i.e. `pop_front()` corresponds to the next reader.
    pub initialized_readers: Option<(usize, VecDeque<Box<dyn FileReader>>)>,
}

pub async fn resolve_to_positive_slice(
    config: &MultiFileReaderConfig,
) -> PolarsResult<ResolvedPositiveSliceInfo> {
    match config.pre_slice.clone() {
        None => Ok(ResolvedPositiveSliceInfo {
            scan_source_idx: 0,
            row_index: config.row_index.clone(),
            pre_slice: None,
            initialized_readers: None,
        }),

        pre_slice @ Some(Slice::Positive { .. }) => Ok(ResolvedPositiveSliceInfo {
            scan_source_idx: 0,
            row_index: config.row_index.clone(),
            pre_slice,
            initialized_readers: None,
        }),

        Some(_) => resolve_negative_slice(config).await,
    }
}

/// Translates a negative slice to positive slice.
async fn resolve_negative_slice(
    config: &MultiFileReaderConfig,
) -> PolarsResult<ResolvedPositiveSliceInfo> {
    todo!()
}
