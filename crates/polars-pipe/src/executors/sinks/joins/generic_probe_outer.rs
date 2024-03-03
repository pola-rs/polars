use std::borrow::Cow;

use arrow::array::{Array, BinaryArray};
use polars_core::export::ahash::RandomState;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_ops::chunked_array::DfTake;
use polars_ops::frame::join::_finish_join;
use polars_ops::prelude::JoinType;
use polars_row::RowsEncoded;
use polars_utils::index::ChunkId;
use polars_utils::slice::GetSaferUnchecked;
use smartstring::alias::String as SmartString;
use arrow::bitmap::{Bitmap, MutableBitmap};
use hashbrown::hash_map::RawEntryMut;

use crate::executors::sinks::joins::generic_build::*;
use crate::executors::sinks::joins::{Key, PartitionedMap};
use crate::executors::sinks::joins::row_values::RowValues;
use crate::executors::sinks::utils::hash_rows;
use crate::expressions::PhysicalPipedExpr;
use crate::operators::{DataChunk, Operator, OperatorResult, PExecutionContext};

#[derive(Clone)]
pub struct GenericOuterJoinProbe {
    /// all chunks are stacked into a single dataframe
    /// the dataframe is not rechunked.
    df_a: Arc<DataFrame>,
    /// The join columns are all tightly packed
    /// the values of a join column(s) can be found
    /// by:
    /// first get the offset of the chunks and multiply that with the number of join
    /// columns
    ///      * chunk_offset = (idx * n_join_keys)
    ///      * end = (offset + n_join_keys)
    materialized_join_cols: Arc<Vec<BinaryArray<i64>>>,
    suffix: Arc<str>,
    hb: RandomState,
    /// partitioned tables that will be used for probing.
    /// stores the key and the chunk_idx, df_idx of the left table.
    hash_tables: Arc<PartitionedMap>,
    /// Bits that indicate if a key has found a match. This keeps track of which values to flush.
    /// Rows that don't find a match need to be appended to the output table.
    /// The chunks are equal to the partitioned hashtable. Every bit is an entry in that table.
    found_match_tracker: Vec<MutableBitmap>,

    // amortize allocations
    // in inner join these are the left table
    // in left join there are the right table
    join_tuples_a: Vec<ChunkId>,
    // in inner join these are the right table
    // in left join there are the left table
    join_tuples_b: Vec<DfIdx>,
    hashes: Vec<u64>,
    // the join order is swapped to ensure we hash the smaller table
    swapped_or_left: bool,
    // location of join columns.
    // these column locations need to be dropped from the rhs
    join_column_idx: Option<Vec<usize>>,
    // cached output names
    output_names: Option<Vec<SmartString>>,
    join_nulls: bool,
    row_values: RowValues,
}


impl GenericOuterJoinProbe {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        mut df_a: DataFrame,
        materialized_join_cols: Arc<Vec<BinaryArray<i64>>>,
        suffix: Arc<str>,
        hb: RandomState,
        hash_tables: Arc<PartitionedMap>,
        join_columns_left: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
        join_columns_right: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
        swapped_or_left: bool,
        // Re-use the hashes allocation of the build side.
        amortized_hashes: Vec<u64>,
        context: &PExecutionContext,
        join_nulls: bool,
    ) -> Self {
        if swapped_or_left {
            let tmp = DataChunk {
                data: df_a.slice(0, 1),
                chunk_index: 0,
            };
            // remove duplicate_names caused by joining
            // on the same column
            let names = join_columns_left
                .iter()
                .flat_map(|phys_e| {
                    phys_e
                        .evaluate(&tmp, context.execution_state.as_any())
                        .ok()
                        .map(|s| s.name().to_string())
                })
                .collect::<Vec<_>>();
            df_a = df_a.drop_many(&names);
        }

        let found_match_tracker = hash_tables.inner().iter().map(|v| MutableBitmap::from_len_zeroed(v.len())).collect();

        GenericOuterJoinProbe {
            df_a: Arc::new(df_a),
            materialized_join_cols,
            suffix,
            hb,
            hash_tables,
            found_match_tracker,
            join_tuples_a: vec![],
            join_tuples_b: vec![],
            hashes: amortized_hashes,
            swapped_or_left,
            join_column_idx: None,
            output_names: None,
            join_nulls,
            row_values: RowValues::new(join_columns_right)
        }
    }

    fn finish_join(
        &mut self,
        mut left_df: DataFrame,
        right_df: DataFrame,
    ) -> PolarsResult<DataFrame> {
        Ok(match &self.output_names {
            None => {
                let out = _finish_join(left_df, right_df, Some(self.suffix.as_ref()))?;
                self.output_names = Some(out.get_column_names_owned());
                out
            },
            Some(names) => unsafe {
                // SAFETY:
                // if we have duplicate names, we overwrite
                // them in the next snippet
                left_df
                    .get_columns_mut()
                    .extend_from_slice(right_df.get_columns());
                left_df
                    .get_columns_mut()
                    .iter_mut()
                    .zip(names)
                    .for_each(|(s, name)| {
                        s.rename(name);
                    });
                left_df
            },
        })
    }

    fn match_outer<'b, I>(&mut self, iter: I)
    where
        I: Iterator<Item = (usize, (&'b u64, &'b [u8]))> + 'b,
    {
        #[inline]
        pub(super) fn compare_fn(
            key: &Key,
            h: u64,
            join_columns_all_chunks: &[BinaryArray<i64>],
            current_row: &[u8],
        ) -> bool {
            let key_hash = key.hash;

            // we check the hash first
            // as that has no indirection
            key_hash == h && {
                // we get the appropriate values from the join columns and compare it with the current row
                let (chunk_idx, df_idx) = key.idx.extract();

                // unset the tracker bit
                let chunk_idx= chunk_idx & !(1 << TRACKER_BIT as IdxSize);
                let chunk_idx = chunk_idx as usize;

                let df_idx = df_idx as usize;

                // get the right columns from the linearly packed buffer
                let other_row = unsafe {
                    join_columns_all_chunks
                        .get_unchecked_release(chunk_idx)
                        .value_unchecked(df_idx)
                };
                current_row == other_row
            }
        }

        for (i, (h, row)) in iter {
            let df_idx_right = i as IdxSize;

            let (entry_builder, partition) = self.hash_tables
                .raw_entry_and_partition_mut(*h);

            let entry = entry_builder
                .from_hash(*h, |key| {
                    compare_fn(key, *h, &self.materialized_join_cols, row)
                });


            match entry {
                RawEntryMut::Occupied(mut entry) => {
                    let key = entry.key_mut();
                    let idx = key.idx.inner_mut();
                    // Set the tracker bit.
                    *idx |= (1 << TRACKER_BIT);

                    todo!()
                    // self.join_tuples_a.extend_from_slice(indexes_left);
                    // self.join_tuples_b
                    //     .extend(std::iter::repeat(df_idx_right).take(indexes_left.len()));
                },
                RawEntryMut::Vacant(_) => {
                    todo!()
                }

            }
        }
    }

    fn execute_outer(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        self.join_tuples_a.clear();
        self.join_tuples_b.clear();
        let mut hashes = std::mem::take(&mut self.hashes);
        let rows = self.row_values.get_values(context, chunk, self.join_nulls)?;
        hash_rows(&rows, &mut hashes, &self.hb);

        if self.join_nulls || rows.null_count() == 0 {
            let iter = hashes.iter().zip(rows.values_iter()).enumerate();
            self.match_outer(iter);
        } else {
            let iter = hashes
                .iter()
                .zip(rows.iter())
                .enumerate()
                .filter_map(|(i, (h, row))| row.map(|row| (i, (h, row))));
            self.match_outer(iter);
        }
        self.hashes = hashes;

        let left_df = unsafe {
            self.df_a
                ._take_chunked_unchecked_seq(&self.join_tuples_a, IsSorted::Not)
        };
        let right_df = unsafe {
            let mut df = Cow::Borrowed(&chunk.data);
            if let Some(ids) = &self.join_column_idx {
                let mut tmp = df.into_owned();
                let cols = tmp.get_columns_mut();
                // we go from higher idx to lower so that lower indices remain untouched
                // by our mutation
                for idx in ids.iter().rev() {
                    let _ = cols.remove(*idx);
                }
                df = Cow::Owned(tmp);
            }
            df._take_unchecked_slice(&self.join_tuples_b, false)
        };

        let (a, b) = if self.swapped_or_left {
            (right_df, left_df)
        } else {
            (left_df, right_df)
        };
        let out = self.finish_join(a, b)?;

        // Clear memory.
        self.row_values.clear();
        self.hashes.clear();

        Ok(OperatorResult::Finished(chunk.with_data(out)))
    }
}

impl Operator for GenericOuterJoinProbe {
    fn execute(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        self.execute_outer(context, chunk)
    }

    fn split(&self, _thread_no: usize) -> Box<dyn Operator> {
        let new = self.clone();
        Box::new(new)
    }
    fn fmt(&self) -> &str {
        "generic_outer_join_probe"
    }
}
