use polars_arrow::record_batch::RecordBatch;
use rayon::prelude::*;

use crate::prelude::*;
use crate::runtime::RAYON;
use crate::series::arrow_export::{arrow_export_row_ranges, contains_arrow_map};
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

        let first_series_col_idx = self
            .columns()
            .iter()
            .position(|col| col.as_series().is_some());
        let df_height = self.height();
        (0..self.first_col_n_chunks()).map(move |i| {
            // There might still be scalar/partitioned columns after aligning,
            // so we follow the size of the chunked column, if any.
            let chunk_size = first_series_col_idx
                .map(|c| self.columns()[c].as_series().unwrap().chunks()[i].len())
                .unwrap_or(df_height);
            self.chunk_frame(i, chunk_size)
        })
    }

    /// Chunk `chunk_idx` of this aligned frame, `height` rows long, as a frame of single chunks.
    pub(crate) fn chunk_frame(&self, chunk_idx: usize, height: usize) -> DataFrame {
        let columns = self
            .columns()
            .iter()
            .map(|col| match col {
                Column::Series(s) => Column::from(s.select_chunk(chunk_idx)),
                Column::Scalar(_) => col.slice(0, height),
            })
            .collect();

        // SAFETY: each column holds the chunk's `height` rows.
        unsafe { DataFrame::new_unchecked(height, columns) }
    }

    pub fn split_chunks_by_n(self, n: usize, parallel: bool) -> Vec<DataFrame> {
        let split = _split_offsets(self.height(), n);

        let split_fn = |(offset, len)| self.slice(offset as i64, len);

        if parallel {
            // Parallel so that null_counts run in parallel
            RAYON.install(|| split.into_par_iter().map(split_fn).collect())
        } else {
            split.into_iter().map(split_fn).collect()
        }
    }

    /// Convert the columns of this [DataFrame] to arrow arrays.
    pub fn rechunk_to_arrow(&self, compat_level: CompatLevel) -> Vec<ArrayRef> {
        self.columns()
            .iter()
            .map(|c| c.clone().rechunk_to_arrow(compat_level))
            .collect()
    }

    /// Convert the columns of this [DataFrame] to arrow arrays.
    pub fn rechunk_into_arrow(self, compat_level: CompatLevel) -> Vec<ArrayRef> {
        self.into_columns()
            .into_iter()
            .map(|c| c.rechunk_to_arrow(compat_level))
            .collect()
    }
}

impl DataFrame {
    /// Slices of this frame that each export to `arrow_dtypes` without overflowing the `i32`
    /// offsets of an arrow MAP; just the frame unless one would.
    ///
    /// Only columns whose arrow dtype contains a MAP are rechunked.
    pub fn split_for_arrow_export<'a>(
        mut self,
        arrow_dtypes: impl IntoIterator<Item = &'a ArrowDataType>,
    ) -> PolarsResult<Vec<DataFrame>> {
        let map_columns: Vec<(usize, &ArrowDataType)> = arrow_dtypes
            .into_iter()
            .enumerate()
            .filter(|(_, dtype)| contains_arrow_map(dtype))
            .collect();
        if map_columns.is_empty() {
            return Ok(vec![self]);
        }

        // SAFETY: rechunking keeps the length and name of each column.
        let columns = unsafe { self.columns_mut_retain_schema() };
        for &(i, _) in &map_columns {
            if columns[i].n_chunks() != 1 {
                columns[i] = columns[i].as_materialized_series().rechunk().into_column();
            }
        }
        let arrays = map_columns.iter().map(|&(i, dtype)| {
            let chunk = self.columns()[i].as_materialized_series().chunks()[0].as_ref();
            (chunk, dtype)
        });
        let Some(ranges) = arrow_export_row_ranges(arrays, self.height())? else {
            return Ok(vec![self]);
        };

        Ok(ranges
            .into_iter()
            .map(|r| self.slice(r.start as i64, r.len()))
            .collect())
    }
}

/// Split DataFrame into chunks in preparation for writing. The chunks have a
/// maximum number of rows per chunk to ensure reasonable memory efficiency when
/// reading the resulting file, and a minimum size per chunk to ensure
/// reasonable performance when writing.
///
/// [`DataFrame::iter_chunks`] can still cut a chunk into several record batches, where its
/// arrow MAP arrays would overflow their `i32` offsets.
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
