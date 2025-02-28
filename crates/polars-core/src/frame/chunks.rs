use arrow::record_batch::RecordBatch;
use rayon::prelude::*;

use crate::prelude::*;
use crate::utils::{_split_offsets, accumulate_dataframes_vertical_unchecked, split_df_as_ref};
use crate::POOL;

impl TryFrom<(RecordBatch, &ArrowSchema)> for DataFrame {
    type Error = PolarsError;

    fn try_from(arg: (RecordBatch, &ArrowSchema)) -> PolarsResult<DataFrame> {
        let columns: PolarsResult<Vec<Column>> = arg
            .0
            .columns()
            .iter()
            .zip(arg.1.iter_values())
            .map(|(arr, field)| Series::try_from((field, arr.clone())).map(Column::from))
            .collect();

        DataFrame::new(columns?)
    }
}

impl DataFrame {
    pub fn split_chunks(&mut self) -> impl Iterator<Item = DataFrame> + '_ {
        self.align_chunks_par();

        (0..self.first_col_n_chunks()).map(move |i| unsafe {
            let columns = self
                .get_columns()
                .iter()
                .map(|column| column.as_materialized_series().select_chunk(i))
                .map(Column::from)
                .collect::<Vec<_>>();

            let height = Self::infer_height(&columns);
            DataFrame::new_no_checks(height, columns)
        })
    }

    pub fn split_chunks_by_n(self, n: usize, parallel: bool) -> Vec<DataFrame> {
        let split = _split_offsets(self.height(), n);

        let split_fn = |(offset, len)| self.slice(offset as i64, len);

        if parallel {
            // Parallel so that null_counts run in parallel
            POOL.install(|| split.into_par_iter().map(split_fn).collect())
        } else {
            split.into_iter().map(split_fn).collect()
        }
    }
}

/// Split DataFrame into chunks in preparation for writing. The chunks have a
/// maximum number of rows per chunk to ensure reasonable memory efficiency when
/// reading the resulting file, and a minimum size per chunk to ensure
/// reasonable performance when writing.
pub fn chunk_df_for_writing(
    df: &mut DataFrame,
    row_group_size: usize,
) -> PolarsResult<std::borrow::Cow<DataFrame>> {
    // ensures all chunks are aligned.
    df.align_chunks_par();

    // Accumulate many small chunks to the row group size.
    // See: #16403
    if !df.get_columns().is_empty()
        && df.get_columns()[0]
            .as_materialized_series()
            .chunk_lengths()
            .take(5)
            .all(|len| len < row_group_size)
    {
        fn finish(scratch: &mut Vec<DataFrame>, new_chunks: &mut Vec<DataFrame>) {
            let mut new = accumulate_dataframes_vertical_unchecked(scratch.drain(..));
            new.as_single_chunk_par();
            new_chunks.push(new);
        }

        let mut new_chunks = Vec::with_capacity(df.first_col_n_chunks()); // upper limit;
        let mut scratch = vec![];
        let mut remaining = row_group_size;

        for df in df.split_chunks() {
            remaining = remaining.saturating_sub(df.height());
            scratch.push(df);

            if remaining == 0 {
                remaining = row_group_size;
                finish(&mut scratch, &mut new_chunks);
            }
        }
        if !scratch.is_empty() {
            finish(&mut scratch, &mut new_chunks);
        }
        return Ok(std::borrow::Cow::Owned(
            accumulate_dataframes_vertical_unchecked(new_chunks),
        ));
    }

    let n_splits = df.height() / row_group_size;
    let result = if n_splits > 0 {
        let mut splits = split_df_as_ref(df, n_splits, false);

        for df in splits.iter_mut() {
            // If the chunks are small enough, writing many small chunks
            // leads to slow writing performance, so in that case we
            // merge them.
            let n_chunks = df.first_col_n_chunks();
            if n_chunks > 1 && (df.estimated_size() / n_chunks < 128 * 1024) {
                df.as_single_chunk_par();
            }
        }

        std::borrow::Cow::Owned(accumulate_dataframes_vertical_unchecked(splits))
    } else {
        std::borrow::Cow::Borrowed(df)
    };
    Ok(result)
}
