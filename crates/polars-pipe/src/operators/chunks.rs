use polars_core::utils::accumulate_dataframes_vertical_unchecked;

use crate::executors::sinks::file_sink::SemicontiguousVstacker;

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
    let mut combiner = SemicontiguousVstacker::new();
    let mut frames_iterator = chunks.into_iter().flat_map(|c| combiner.add(c.data)).peekable();
    if frames_iterator.peek().is_some() {
        let mut result = accumulate_dataframes_vertical_unchecked(frames_iterator);
        if let Some(df) = combiner.flush() {
            let _ = result.vstack_mut(&df);
        }
        result
    } else {
        // The presumption is that this function is never called with empty
        // data, cause that'll cause accumulate_dataframes_vertical_unchecked to
        // error, so if we haven't gotten any data we can safely assume it's in
        // the combiner buffer.
        combiner.flush().unwrap()
    }
}
