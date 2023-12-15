use polars_core::utils::accumulate_dataframes_vertical_unchecked;

use super::*;

#[derive(Clone, Debug)]
pub struct DataChunk {
    pub chunk_index: IdxSize,
    pub data: DataFrame,
}

impl DataChunk {
    pub(crate) fn new(chunk_index: IdxSize, data: DataFrame) -> Self {
        // Check the invariant that all columns have a single chunk.
        #[cfg(debug_assertions)]
        {
            for c in data.get_columns() {
                assert_eq!(c.chunks().len(), 1);
            }
        }
        Self { chunk_index, data }
    }
    pub(crate) fn with_data(&self, data: DataFrame) -> Self {
        Self::new(self.chunk_index, data)
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.data.height() == 0
    }
}

pub(crate) fn chunks_to_df_unchecked(chunks: Vec<DataChunk>) -> DataFrame {
    accumulate_dataframes_vertical_unchecked(chunks.into_iter().map(|c| c.data))
}
