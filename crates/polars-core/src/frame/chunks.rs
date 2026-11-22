use std::ops::Range;

use polars_arrow::record_batch::RecordBatch;
use rayon::prelude::*;

use crate::frame::column::ScalarColumn;
use crate::prelude::*;
use crate::runtime::RAYON;
use crate::series::arrow_export::{
    MAX_MAP_ENTRIES, arrow_export_cut, contains_arrow_map, map_spans_fit, max_map_span,
};
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

/// The one row a Scalar column repeats, as an array; `None` for an empty column, which has none.
pub(crate) fn scalar_column_row(sc: &ScalarColumn) -> Option<ArrayRef> {
    let row = sc.as_single_value_series().rechunk();
    row.chunks().first().filter(|row| row.len() == 1).cloned()
}

impl DataFrame {
    /// Slices of this frame that each export to `arrow_dtypes`, one per column, without
    /// overflowing the `i32` offsets of an arrow MAP; just the frame unless one would.
    ///
    /// The columns may be chunked differently, and nothing is rechunked: a frame that may
    /// overflow is first cut at the chunk boundaries of its MAP columns, which can give more
    /// slices than needed.
    pub fn split_for_arrow_export<'a>(
        self,
        arrow_dtypes: impl IntoIterator<Item = &'a ArrowDataType>,
    ) -> Vec<DataFrame> {
        let dtypes: Vec<_> = arrow_dtypes.into_iter().collect();
        assert_eq!(dtypes.len(), self.width(), "one arrow dtype per column");
        let height = self.height();
        // Before any row is read, as an empty Scalar column has none.
        if height == 0 || !dtypes.iter().any(|d| contains_arrow_map(d)) {
            return vec![self];
        }
        // Each MAP column, with the one row it repeats if it is a Scalar column.
        let map_columns: Vec<_> = self
            .columns()
            .iter()
            .zip(&dtypes)
            .filter(|(_, dtype)| contains_arrow_map(dtype))
            .map(|(c, dtype)| (c, *dtype, c.as_scalar_column().and_then(scalar_column_row)))
            .collect();

        // The spans of all chunks added up bound every MAP of the frame.
        let fits = map_columns
            .iter()
            .all(|(c, dtype, row)| match (c.as_scalar_column(), row) {
                (Some(_), Some(row)) => map_spans_fit(row.as_ref(), dtype, 0..1, height),
                (Some(_), None) => true,
                (None, _) => {
                    c.as_materialized_series()
                        .chunks()
                        .iter()
                        .map(|arr| max_map_span(arr.as_ref(), 0..arr.len(), dtype))
                        .fold(0u64, u64::saturating_add)
                        <= MAX_MAP_ENTRIES
                },
            });
        if fits {
            return vec![self];
        }

        // Within a candidate, each chunked MAP column is a slice of one of its chunks.
        let mut cuts = vec![0, height];
        for (c, _, _) in &map_columns {
            if c.as_scalar_column().is_none() {
                let lengths = c.as_materialized_series().chunk_lengths();
                cuts.extend(lengths.scan(0, |end, len| {
                    *end += len;
                    Some(*end)
                }));
            }
        }
        cuts.sort_unstable();
        cuts.dedup();

        let mut cursors = vec![(0, 0); map_columns.len()];
        let mut ranges = Vec::new();
        for candidate in cuts.windows(2).map(|w| w[0]..w[1]) {
            let arrays: Vec<(ArrayRef, &ArrowDataType, bool)> = map_columns
                .iter()
                .zip(&mut cursors)
                .filter_map(|((c, dtype, row), (idx, start))| {
                    if c.as_scalar_column().is_some() {
                        return row.as_ref().map(|row| (row.clone(), *dtype, true));
                    }
                    let chunks = c.as_materialized_series().chunks();
                    while *start + chunks[*idx].len() <= candidate.start {
                        *start += chunks[*idx].len();
                        *idx += 1;
                    }
                    let arr = chunks[*idx].sliced(candidate.start - *start, candidate.len());
                    Some((arr, *dtype, false))
                })
                .collect();
            let fits = |r: Range<usize>| {
                arrays.iter().all(|(arr, dtype, repeated)| {
                    if *repeated {
                        map_spans_fit(arr.as_ref(), dtype, 0..1, r.len())
                    } else {
                        map_spans_fit(arr.as_ref(), dtype, r.clone(), 1)
                    }
                })
            };
            let mut start = 0;
            while start < candidate.len() {
                let end = arrow_export_cut(start..candidate.len(), fits);
                ranges.push(candidate.start + start..candidate.start + end);
                start = end;
            }
        }

        ranges
            .into_iter()
            .map(|r| {
                let columns = self
                    .columns()
                    .iter()
                    .map(|c| match c {
                        // `Column::slice` makes a one-row Series a Scalar by reading its row,
                        // which can panic where the export of that row would return an error.
                        Column::Series(s) => Column::Series(s.slice(r.clone()).into()),
                        _ => c.slice(r.clone()),
                    })
                    .collect();
                // SAFETY: every column is sliced to the `r.len()` rows of `r`, and keeps its
                // name and dtype.
                unsafe { DataFrame::new_unchecked(r.len(), columns).with_schema_from(&self) }
            })
            .collect()
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
