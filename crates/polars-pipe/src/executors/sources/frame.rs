use std::iter::Enumerate;
use std::vec::IntoIter;

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::utils::split_df;
use polars_core::POOL;
use polars_utils::IdxSize;

use crate::executors::sources::get_source_index;
use crate::operators::{DataChunk, PExecutionContext, Source, SourceResult};

pub struct DataFrameSource {
    dfs: Enumerate<IntoIter<DataFrame>>,
    n_threads: usize,
}

impl DataFrameSource {
    pub(crate) fn from_df(mut df: DataFrame) -> Self {
        let n_threads = POOL.current_num_threads();
        let dfs = split_df(&mut df, n_threads, false);
        let dfs = dfs.into_iter().enumerate();
        Self { dfs, n_threads }
    }
}

impl Source for DataFrameSource {
    fn get_batches(&mut self, _context: &PExecutionContext) -> PolarsResult<SourceResult> {
        let idx_offset = get_source_index(0);
        let chunks = (&mut self.dfs)
            .map(|(chunk_index, data)| DataChunk {
                chunk_index: (chunk_index as u32 + idx_offset) as IdxSize,
                data,
            })
            .take(self.n_threads)
            .collect::<Vec<_>>();
        get_source_index(chunks.len() as u32);

        if chunks.is_empty() {
            Ok(SourceResult::Finished)
        } else {
            Ok(SourceResult::GotMoreData(chunks))
        }
    }
    fn fmt(&self) -> &str {
        "df"
    }
}
