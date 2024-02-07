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
    let mut combiner = SemicontiguousVstacker::default();
    let mut frames_iterator = chunks
        .into_iter()
        .flat_map(|c| combiner.add(c.data))
        .peekable();
    if frames_iterator.peek().is_some() {
        let mut result = accumulate_dataframes_vertical_unchecked(frames_iterator);
        if let Some(df) = combiner.finish() {
            let _ = result.vstack_mut(&df);
        }
        result
    } else {
        // The presumption is that this function is never called with empty
        // data, cause that'll cause accumulate_dataframes_vertical_unchecked to
        // error, so if we haven't gotten any data we can safely assume it's in
        // the combiner buffer.
        combiner.finish().unwrap()
    }
}

/// Combine `DataFrame`s into sufficiently large contiguous memory allocations
/// that we don't suffer from overhead of many small writes. At the same time,
/// don't use too much memory by creating overly large allocations.
///
/// The assumption is that added `DataFrame`s are already in the correct order,
/// and can therefore be combined.
///
#[derive(Clone)]
pub(crate) struct SemicontiguousVstacker {
    current_dataframe: Option<DataFrame>,
    /// Have we vstack()ed on to the current chunk?
    stacked: bool,
    /// How big should resulting chunks be, if possible?
    output_chunk_size: usize,
}

impl SemicontiguousVstacker {
    /// Create a new instance.
    pub fn new(output_chunk_size: usize) -> Self {
        Self {
            current_dataframe: None,
            stacked: false,
            output_chunk_size,
        }
    }

    /// Add another `DataFrame`, return any (potentially combined) `DataFrame`s
    /// that result.
    pub fn add(&mut self, next_frame: DataFrame) -> impl Iterator<Item = DataFrame> {
        let mut result: [Option<DataFrame>; 2] = [None, None];

        // If the next chunk is too large, we probably don't want make copies of
        // it when we do as_single_chunk() in flush(), so we flush in advance.
        if self.current_dataframe.is_some()
            && next_frame.estimated_size() > self.output_chunk_size / 4
        {
            result[0] = self.flush();
        }

        if let Some(ref mut current_frame) = self.current_dataframe {
            current_frame
                .vstack_mut(&next_frame)
                .expect("These are chunks from the same dataframe");
            self.stacked = true;
        } else {
            self.current_dataframe = Some(next_frame);
        };

        if self.current_dataframe.as_ref().unwrap().estimated_size() > self.output_chunk_size {
            result[1] = self.flush();
        }
        result.into_iter().flatten()
    }

    /// Clear and return any cached `DataFrame` data, making it contiguous if
    /// relevant.
    #[must_use]
    fn flush(&mut self) -> Option<DataFrame> {
        if let Some(mut current_frame) = std::mem::take(&mut self.current_dataframe) {
            // If we've stacked multiple small batches we want to make the data
            // contiguous.
            if self.stacked {
                current_frame.as_single_chunk();
            }
            self.stacked = false;
            Some(current_frame)
        } else {
            None
        }
    }

    /// Finish and return any remaining cached `DataFrame` data. The only way
    /// that `SemicontiguousVstacker` should be cleaned up.
    #[must_use]
    pub fn finish(mut self) -> Option<DataFrame> {
        self.flush()
    }
}

impl Default for SemicontiguousVstacker {
    /// 4 MB was chosen based on some empirical experiments that showed it to
    /// be decently faster than lower or higher values, and it's small enough
    /// it won't impact memory usage significantly.
    fn default() -> Self {
        SemicontiguousVstacker::new(4 * 1024 * 1024)
    }
}

#[cfg(test)]
mod test {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        /// DataFrames get merged into chunks that are bigger than 4MB when
        /// possible.
        #[test]
        #[cfg_attr(miri, ignore)] // miri and proptest do not work well
        fn semicontiguous_vstacker_merges(df_lengths in prop::collection::vec(1..40usize, 1..50)) {
            // Convert the lengths into a series of DataFrames:
            let mut vstacker = SemicontiguousVstacker::new(4096);
            let dfs : Vec<DataFrame> = df_lengths.iter().enumerate().map(|(i, length)| {
                let series = Series::new("val", vec![i as u64; *length]);
                DataFrame::new(vec![series]).unwrap()
            }).collect();

            // Combine the DataFrames using a SemicontiguousVstacker:
            let mut results = vec![];
            for (i, df) in dfs.iter().enumerate() {
                for result_df in vstacker.add(df.clone()) {
                    results.push((i, result_df));
                }
            }
            if let Some(result_df) = vstacker.finish() {
                results.push((df_lengths.len() - 1, result_df));
            }

            // Make sure the lengths are as sufficiently large, and the chunks
            // were merged, the whole point of the exercise:
            for (original_idx, result_df) in &results {
                if result_df.height() < 40 {
                    // This means either this was the last df, or the next one
                    // was big enough we decided not to aggregate.
                    if *original_idx < results.len() - 1 {
                        assert!(dfs[original_idx + 1].height() > 10);
                    }
                }
                // Make sure all result DataFrames only have a single chunk.
                assert_eq!(result_df.get_columns()[0].chunk_lengths().len(), 1);
            }

            // Make sure the data was preserved:
            assert_eq!(
                accumulate_dataframes_vertical_unchecked(dfs.into_iter()),
                accumulate_dataframes_vertical_unchecked(results.into_iter().map(|(_, df)| df)),
            );
        }
    }
}
