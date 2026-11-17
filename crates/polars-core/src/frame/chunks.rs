use std::borrow::Cow;
use std::ops::Range;

use polars_arrow::record_batch::RecordBatch;
use rayon::prelude::*;

use crate::frame::column::ScalarColumn;
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
        let mut offset = 0;
        (0..self.first_col_n_chunks()).map(move |i| {
            // There might still be scalar/partitioned columns after aligning,
            // so we follow the size of the chunked column, if any.
            let chunk_size = first_series_col_idx
                .map(|c| self.columns()[c].as_series().unwrap().chunks()[i].len())
                .unwrap_or(df_height);
            let rows = offset..offset + chunk_size;
            offset += chunk_size;
            self.chunk_frame(i, rows)
        })
    }

    /// Chunk `chunk_idx` of this aligned frame, which holds `rows`, as a frame of single chunks.
    pub(crate) fn chunk_frame(&self, chunk_idx: usize, rows: Range<usize>) -> DataFrame {
        let columns = self
            .columns()
            .iter()
            .map(|col| match col {
                Column::Series(s) => Column::from(s.select_chunk(chunk_idx)),
                // These have no chunks of their own to follow.
                Column::Scalar(_) | Column::Dictionary(_) => {
                    col.slice(rows.start as i64, rows.len())
                },
            })
            .collect();

        // SAFETY: each column holds the chunk's rows.
        unsafe { DataFrame::new_unchecked(rows.len(), columns) }
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
}

impl DataFrame {
    /// Row ranges cutting this frame, or its chunk `chunk_idx` holding `rows` if given, so that
    /// each exports to `arrow_dtypes` without overflowing the `i32` offsets of an arrow MAP;
    /// `None` if no cut is needed.
    pub(crate) fn arrow_export_ranges<'a>(
        &self,
        arrow_dtypes: impl IntoIterator<Item = &'a ArrowDataType>,
        chunk: Option<(usize, Range<usize>)>,
    ) -> PolarsResult<Option<Vec<Range<usize>>>> {
        let rows = chunk
            .as_ref()
            .map_or(0..self.height(), |(_, rows)| rows.clone());
        let columns: Vec<(Cow<'_, [ArrayRef]>, &ArrowDataType, bool)> = self
            .columns()
            .iter()
            .zip(arrow_dtypes)
            .filter(|(_, dtype)| contains_arrow_map(dtype))
            .map(|(c, dtype)| {
                PolarsResult::Ok(match (c, &chunk) {
                    (Column::Series(s), None) => {
                        (Cow::Borrowed(s.chunks().as_slice()), dtype, false)
                    },
                    (Column::Series(s), Some((idx, _))) => (
                        Cow::Borrowed(std::slice::from_ref(&s.chunks()[*idx])),
                        dtype,
                        false,
                    ),
                    (Column::Scalar(sc), _) => (Cow::Owned(single_row_chunks(sc)), dtype, true),
                    (_, _) => (
                        Cow::Owned(
                            c.slice(rows.start as i64, rows.len())
                                .take_materialized_series()
                                .into_chunks(),
                        ),
                        dtype,
                        false,
                    ),
                })
            })
            .collect::<PolarsResult<_>>()?;

        arrow_export_row_ranges(
            columns.iter().map(|(c, d, r)| (c.as_ref(), *d, *r)),
            rows.len(),
        )
    }

    /// Slices of this frame that each export to `arrow_dtypes` without overflowing the `i32`
    /// offsets of an arrow MAP; just the frame unless one would.
    ///
    /// The columns may be chunked differently, and nothing is rechunked.
    pub fn split_for_arrow_export<'a>(
        self,
        arrow_dtypes: impl IntoIterator<Item = &'a ArrowDataType>,
    ) -> PolarsResult<Vec<DataFrame>> {
        let Some(ranges) = self.arrow_export_ranges(arrow_dtypes, None)? else {
            return Ok(vec![self]);
        };

        Ok(ranges
            .into_iter()
            .map(|r| self.slice(r.start as i64, r.len()))
            .collect())
    }
}

/// A single row of `sc` as physical chunks whose rows together make up that row.
fn single_row_chunks(sc: &ScalarColumn) -> Vec<ArrayRef> {
    // Building a Map value's Series deduplicates its keys, which costs a hash per entry, and
    // imploding its entries would copy them unless they are a single chunk.
    #[cfg(feature = "dtype-map")]
    if let AnyValue::Map(entries) = sc.scalar().value() {
        return entries
            .chunks()
            .iter()
            .map(|chunk| {
                let offsets = vec![0, chunk.len() as i64];
                let dtype = ListArray::<i64>::default_datatype(chunk.dtype().clone());
                // SAFETY: the offsets span the whole chunk.
                let offsets =
                    unsafe { polars_arrow::offset::Offsets::new_unchecked(offsets) }.into();
                ListArray::<i64>::new(dtype, offsets, chunk.clone(), None).boxed()
            })
            .collect();
    }
    sc.as_single_value_series().into_chunks()
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
