use polars_core::utils::accumulate_dataframes_vertical_unchecked;

use super::*;

#[derive(Clone, Debug)]
pub struct DataChunk {
    pub chunk_index: IdxSize,
    pub data: DataFrame,
}

impl DataChunk {
    pub(crate) fn with_data(&self, data: DataFrame) -> Self {
        DataChunk {
            chunk_index: self.chunk_index,
            data,
        }
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.data.height() == 0
    }
}

pub(crate) fn chunks_to_df_unchecked(chunks: Vec<DataChunk>) -> DataFrame {
    accumulate_dataframes_vertical_unchecked(chunks.into_iter().map(|c| c.data))
}
