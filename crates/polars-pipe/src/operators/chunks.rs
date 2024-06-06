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
        self.data.is_empty()
    }
}

pub(crate) fn chunks_to_df_unchecked(chunks: Vec<DataChunk>) -> DataFrame {
    accumulate_dataframes_vertical_unchecked(chunks.into_iter().map(|c| c.data))
}

/// Combine a series of `DataFrame`s, and if they're small enough, combine them
/// into larger `DataFrame`s using `vstack`. This allows the caller to turn them
/// into contiguous memory allocations so that we don't suffer from overhead of
/// many small writes. The assumption is that added `DataFrame`s are already in
/// the correct order, and can therefore be combined.
///
/// The benefit of having a series of `DataFrame` that are e.g. 4MB each that
/// are then made contiguous is that you're not using a lot of memory (an extra
/// 4MB), but you're still doing better than if you had a series of of 2KB
/// `DataFrame`s.
///
/// Changing the `DataFrame` into contiguous chunks is the caller's
/// responsibility.
#[cfg(any(feature = "parquet", feature = "ipc", feature = "csv"))]
#[derive(Clone)]
pub(crate) struct StreamingVstacker {
    current_dataframe: Option<DataFrame>,
    /// How big should resulting chunks be, if possible?
    output_chunk_size: usize,
}

#[cfg(any(feature = "parquet", feature = "ipc", feature = "csv"))]
impl StreamingVstacker {
    /// Create a new instance.
    pub fn new(output_chunk_size: usize) -> Self {
        Self {
            current_dataframe: None,
            output_chunk_size,
        }
    }

    /// Add another `DataFrame`, return (potentially combined) `DataFrame`s that
    /// result, if any.
    pub fn add(&mut self, next_frame: DataFrame) -> impl Iterator<Item = DataFrame> {
        let mut result: [Option<DataFrame>; 2] = [None, None];

        // If the next chunk is too large, we probably don't want make copies of
        // it if a caller does as_single_chunk(), so we flush in advance.
        if self.current_dataframe.is_some()
            && next_frame.estimated_size() > self.output_chunk_size / 4
        {
            result[0] = self.flush();
        }

        if let Some(ref mut current_frame) = self.current_dataframe {
            current_frame
                .vstack_mut(&next_frame)
                .expect("These are chunks from the same dataframe");
        } else {
            self.current_dataframe = Some(next_frame);
        };

        if self.current_dataframe.as_ref().unwrap().estimated_size() > self.output_chunk_size {
            result[1] = self.flush();
        }
        result.into_iter().flatten()
    }

    /// Clear and return any cached `DataFrame` data.
    #[must_use]
    fn flush(&mut self) -> Option<DataFrame> {
        std::mem::take(&mut self.current_dataframe)
    }

    /// Finish and return any remaining cached `DataFrame` data. The only way
    /// that `SemicontiguousVstacker` should be cleaned up.
    #[must_use]
    pub fn finish(mut self) -> Option<DataFrame> {
        self.flush()
    }
}

#[cfg(any(feature = "parquet", feature = "ipc", feature = "csv"))]
impl Default for StreamingVstacker {
    /// 4 MB was chosen based on some empirical experiments that showed it to
    /// be decently faster than lower or higher values, and it's small enough
    /// it won't impact memory usage significantly.
    fn default() -> Self {
        StreamingVstacker::new(4 * 1024 * 1024)
    }
}

#[cfg(test)]
#[cfg(any(feature = "parquet", feature = "ipc", feature = "csv"))]
mod test {
    use super::*;

    /// DataFrames get merged into chunks that are bigger than the specified
    /// size when possible.
    #[test]
    fn semicontiguous_vstacker_merges() {
        let test = semicontiguous_vstacker_merges_impl;
        test(vec![10]);
        test(vec![10, 10, 10, 10, 10, 10, 10]);
        test(vec![10, 40, 10, 10, 10, 10]);
        test(vec![40, 10, 10, 40, 10, 10, 40]);
        test(vec![50, 50, 50]);
    }

    /// Eventually would be nice to drive this with proptest.
    fn semicontiguous_vstacker_merges_impl(df_lengths: Vec<usize>) {
        // Convert the lengths into a series of DataFrames:
        let mut vstacker = StreamingVstacker::new(4096);
        let dfs: Vec<DataFrame> = df_lengths
            .iter()
            .enumerate()
            .map(|(i, length)| {
                let series = Series::new("val", vec![i as u64; *length]);
                DataFrame::new(vec![series]).unwrap()
            })
            .collect();

        // Combine the DataFrames using a SemicontiguousVstacker:
        let mut results = vec![];
        for (i, df) in dfs.iter().enumerate() {
            for mut result_df in vstacker.add(df.clone()) {
                result_df.as_single_chunk();
                results.push((i, result_df));
            }
        }
        if let Some(mut result_df) = vstacker.finish() {
            result_df.as_single_chunk();
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
