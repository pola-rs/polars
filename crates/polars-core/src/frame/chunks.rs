use std::ops::Range;

use polars_arrow::record_batch::RecordBatch;
use rayon::prelude::*;

use crate::prelude::*;
use crate::runtime::RAYON;
use crate::utils::{_split_offsets, accumulate_dataframes_vertical_unchecked, split_df_as_ref};

impl From<RecordBatch> for DataFrame {
    fn from(rb: RecordBatch) -> DataFrame {
        let height = rb.height();
        let (schema, arrays) = rb.into_schema_and_arrays();

        let columns: Vec<Column> = arrays
            .into_iter()
            .zip(schema.iter())
            .map(|(arr, (name, field))| {
                // SAFETY: Record Batch has the invariant that the schema datatype matches the
                // columns.
                unsafe {
                    Series::_try_from_arrow_unchecked_with_md(
                        name.clone(),
                        vec![arr],
                        field.dtype(),
                        field.metadata.as_deref(),
                    )
                }
                .unwrap()
                .into_column()
            })
            .collect();

        // SAFETY: RecordBatch has the same invariants for names and heights as DataFrame.
        unsafe { DataFrame::new_unchecked(height, columns) }
    }
}

impl DataFrame {
    pub fn split_chunks(&mut self) -> impl Iterator<Item = DataFrame> + '_ {
        self.align_chunks_par();

        // Follows the first column that has chunks (a series, or a dictionary column's codes);
        // scalar columns have none and are sliced.
        let chunk_lengths: Vec<usize> = self.first_col_chunk_lengths().collect();
        let mut offset = 0;
        (0..chunk_lengths.len()).map(move |i| {
            let rows = offset..offset + chunk_lengths[i];
            offset += chunk_lengths[i];
            self.chunk_frame(i, rows)
        })
    }

    /// Chunk `chunk_idx` of this aligned frame, which holds `rows`, as a frame of single chunks.
    pub(crate) fn chunk_frame(&self, chunk_idx: usize, rows: Range<usize>) -> DataFrame {
        let columns = self
            .columns()
            .iter()
            .map(|col| match col {
                // `Column::from` would turn a one-row chunk into a Scalar, which reads the row.
                Column::Series(s) => Column::Series(s.select_chunk(chunk_idx).into()),
                // These are sliced by offset.
                Column::Scalar(_) | Column::Dictionary(_) => col.slice(rows.clone()),
            })
            .collect();

        // SAFETY: each column holds the chunk's rows.
        unsafe { DataFrame::new_unchecked(rows.len(), columns) }
    }

    pub fn split_chunks_by_n(self, n: usize, parallel: bool) -> Vec<DataFrame> {
        let split = _split_offsets(self.height(), n);

        let split_fn = |(offset, len)| self.slice((offset, len));

        if parallel {
            // Parallel so that null_counts run in parallel
            RAYON.install(|| split.into_par_iter().map(split_fn).collect())
        } else {
            split.into_iter().map(split_fn).collect()
        }
    }

    /// Convert the columns of this [DataFrame] to arrow arrays, one per column.
    ///
    /// Use [`DataFrame::try_iter_chunks`] for Map columns that may need several arrays.
    pub fn rechunk_to_arrow(&self, compat_level: CompatLevel) -> Vec<ArrayRef> {
        self.columns()
            .iter()
            .map(|c| c.clone().rechunk_to_arrow(compat_level))
            .collect()
    }

    /// Convert the columns of this [DataFrame] to arrow arrays, one per column.
    ///
    /// Use [`DataFrame::try_iter_chunks`] for Map columns that may need several arrays.
    pub fn rechunk_into_arrow(self, compat_level: CompatLevel) -> Vec<ArrayRef> {
        self.into_columns()
            .into_iter()
            .map(|c| c.rechunk_to_arrow(compat_level))
            .collect()
    }
}

/// Split DataFrame into chunks in preparation for writing. The chunks have a
/// maximum number of rows per chunk to ensure reasonable memory efficiency when
/// reading the resulting file, and a minimum size per chunk to ensure
/// reasonable performance when writing.
///
/// [`DataFrame::iter_chunks`] can still cut a chunk into several record batches, where its
/// arrow MAP or dense Union arrays would overflow their `i32` offsets.
pub fn chunk_df_for_writing(
    df: &mut DataFrame,
    row_group_size: usize,
) -> PolarsResult<std::borrow::Cow<'_, DataFrame>> {
    // ensures all chunks are aligned.
    df.align_chunks_par();

    // Accumulate many small chunks to the row group size.
    // See: #16403
    if !df.columns().is_empty()
        && df
            .first_col_chunk_lengths()
            .take(5)
            .all(|len| len < row_group_size)
    {
        fn finish(scratch: &mut Vec<DataFrame>, new_chunks: &mut Vec<DataFrame>) {
            let mut new = accumulate_dataframes_vertical_unchecked(scratch.drain(..));
            new.rechunk_mut_par();
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
            if n_chunks > 1 && (df.estimated_size(true) / n_chunks < 128 * 1024) {
                df.rechunk_mut_par();
            }
        }

        std::borrow::Cow::Owned(accumulate_dataframes_vertical_unchecked(splits))
    } else {
        std::borrow::Cow::Borrowed(df)
    };
    Ok(result)
}
