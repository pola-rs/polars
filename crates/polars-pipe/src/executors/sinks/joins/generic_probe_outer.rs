use std::sync::atomic::Ordering;

use arrow::array::{Array, BinaryArray, MutablePrimitiveArray};
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_ops::chunked_array::DfTake;
use polars_ops::frame::join::_finish_join;
use polars_ops::prelude::_coalesce_full_join;
use smartstring::alias::String as SmartString;

use crate::executors::sinks::joins::generic_build::*;
use crate::executors::sinks::joins::row_values::RowValues;
use crate::executors::sinks::joins::PartitionedMap;
use crate::executors::sinks::utils::hash_rows;
use crate::executors::sinks::ExtraPayload;
use crate::expressions::PhysicalPipedExpr;
use crate::operators::{DataChunk, Operator, OperatorResult, PExecutionContext};

#[derive(Clone)]
pub struct GenericFullOuterJoinProbe<K: ExtraPayload> {
    /// all chunks are stacked into a single dataframe
    /// the dataframe is not rechunked.
    df_a: Arc<DataFrame>,
    // Dummy needed for the flush phase.
    df_b_flush_dummy: Option<DataFrame>,
    /// The join columns are all tightly packed
    /// the values of a join column(s) can be found
    /// by:
    /// first get the offset of the chunks and multiply that with the number of join
    /// columns
    ///      * chunk_offset = (idx * n_join_keys)
    ///      * end = (offset + n_join_keys)
    materialized_join_cols: Arc<[BinaryArray<i64>]>,
    suffix: Arc<str>,
    hb: PlRandomState,
    /// partitioned tables that will be used for probing.
    /// stores the key and the chunk_idx, df_idx of the left table.
    hash_tables: Arc<PartitionedMap<K>>,

    // amortize allocations
    // in inner join these are the left table
    // in left join there are the right table
    join_tuples_a: Vec<NullableChunkId>,
    // in inner join these are the right table
    // in left join there are the left table
    join_tuples_b: MutablePrimitiveArray<IdxSize>,
    hashes: Vec<u64>,
    // the join order is swapped to ensure we hash the smaller table
    swapped: bool,
    // cached output names
    output_names: Option<Vec<SmartString>>,
    join_nulls: bool,
    coalesce: bool,
    thread_no: usize,
    row_values: RowValues,
    key_names_left: Arc<[SmartString]>,
    key_names_right: Arc<[SmartString]>,
}

impl<K: ExtraPayload> GenericFullOuterJoinProbe<K> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        df_a: DataFrame,
        materialized_join_cols: Arc<[BinaryArray<i64>]>,
        suffix: Arc<str>,
        hb: PlRandomState,
        hash_tables: Arc<PartitionedMap<K>>,
        join_columns_right: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
        swapped: bool,
        // Re-use the hashes allocation of the build side.
        amortized_hashes: Vec<u64>,
        join_nulls: bool,
        coalesce: bool,
        key_names_left: Arc<[SmartString]>,
        key_names_right: Arc<[SmartString]>,
    ) -> Self {
        GenericFullOuterJoinProbe {
            df_a: Arc::new(df_a),
            df_b_flush_dummy: None,
            materialized_join_cols,
            suffix,
            hb,
            hash_tables,
            join_tuples_a: vec![],
            join_tuples_b: MutablePrimitiveArray::new(),
            hashes: amortized_hashes,
            swapped,
            output_names: None,
            join_nulls,
            coalesce,
            thread_no: 0,
            row_values: RowValues::new(join_columns_right, false),
            key_names_left,
            key_names_right,
        }
    }

    fn finish_join(&mut self, left_df: DataFrame, right_df: DataFrame) -> PolarsResult<DataFrame> {
        fn inner(
            left_df: DataFrame,
            right_df: DataFrame,
            suffix: &str,
            swapped: bool,
            output_names: &mut Option<Vec<SmartString>>,
        ) -> PolarsResult<DataFrame> {
            let (mut left_df, right_df) = if swapped {
                (right_df, left_df)
            } else {
                (left_df, right_df)
            };
            Ok(match output_names {
                None => {
                    let out = _finish_join(left_df, right_df, Some(suffix))?;
                    *output_names = Some(out.get_column_names_owned());
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

        if self.coalesce {
            let out = inner(
                left_df.clone(),
                right_df,
                self.suffix.as_ref(),
                self.swapped,
                &mut self.output_names,
            )?;
            let l = self
                .key_names_left
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>();
            let r = self
                .key_names_right
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>();
            Ok(_coalesce_full_join(
                out,
                &l,
                &r,
                Some(self.suffix.as_ref()),
                &left_df,
            ))
        } else {
            inner(
                left_df.clone(),
                right_df,
                self.suffix.as_ref(),
                self.swapped,
                &mut self.output_names,
            )
        }
    }

    fn match_outer<'b, I>(&mut self, iter: I)
    where
        I: Iterator<Item = (usize, (&'b u64, &'b [u8]))> + 'b,
    {
        for (i, (h, row)) in iter {
            let df_idx_right = i as IdxSize;

            let entry = self
                .hash_tables
                .raw_entry(*h)
                .from_hash(*h, |key| {
                    compare_fn(key, *h, &self.materialized_join_cols, row)
                })
                .map(|key_val| key_val.1);

            if let Some((indexes_left, tracker)) = entry {
                // compiles to normal store: https://rust.godbolt.org/z/331hMo339
                tracker.get_tracker().store(true, Ordering::Relaxed);

                self.join_tuples_a.extend_from_slice(indexes_left);
                self.join_tuples_b
                    .extend_constant(indexes_left.len(), Some(df_idx_right));
            } else {
                self.join_tuples_a.push(ChunkId::null());
                self.join_tuples_b.push_value(df_idx_right);
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

        if self.df_b_flush_dummy.is_none() {
            self.df_b_flush_dummy = Some(chunk.data.clear())
        }

        let mut hashes = std::mem::take(&mut self.hashes);
        let rows = self
            .row_values
            .get_values(context, chunk, self.join_nulls)?;
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
                ._take_opt_chunked_unchecked_seq(&self.join_tuples_a)
        };
        let right_df = unsafe {
            self.join_tuples_b.with_freeze(|idx| {
                let idx = IdxCa::from(idx.clone());
                let out = chunk.data.take_unchecked_impl(&idx, false);
                // Drop so that the freeze context can go back to mutable array.
                drop(idx);
                out
            })
        };
        let out = self.finish_join(left_df, right_df)?;
        Ok(OperatorResult::Finished(chunk.with_data(out)))
    }

    fn execute_flush(&mut self) -> PolarsResult<OperatorResult> {
        let ht = self.hash_tables.inner();
        let n = ht.len();
        self.join_tuples_a.clear();

        ht.iter().enumerate().for_each(|(i, ht)| {
            if i % n == self.thread_no {
                ht.iter().for_each(|(_k, (idx_left, tracker))| {
                    let found_match = tracker.get_tracker().load(Ordering::Relaxed);

                    if !found_match {
                        self.join_tuples_a.extend_from_slice(idx_left);
                    }
                })
            }
        });

        let left_df = unsafe {
            self.df_a
                ._take_chunked_unchecked_seq(&self.join_tuples_a, IsSorted::Not)
        };

        let size = left_df.height();
        let right_df = self.df_b_flush_dummy.as_ref().unwrap();

        let right_df = unsafe {
            DataFrame::new_no_checks(
                right_df
                    .get_columns()
                    .iter()
                    .map(|s| Series::full_null(s.name(), size, s.dtype()))
                    .collect(),
            )
        };

        let out = self.finish_join(left_df, right_df)?;
        Ok(OperatorResult::Finished(DataChunk::new(0, out)))
    }
}

impl<K: ExtraPayload> Operator for GenericFullOuterJoinProbe<K> {
    fn execute(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        self.execute_outer(context, chunk)
    }

    fn flush(&mut self) -> PolarsResult<OperatorResult> {
        self.execute_flush()
    }

    fn must_flush(&self) -> bool {
        self.df_b_flush_dummy.is_some()
    }

    fn split(&self, thread_no: usize) -> Box<dyn Operator> {
        let mut new = self.clone();
        new.thread_no = thread_no;
        Box::new(new)
    }
    fn fmt(&self) -> &str {
        "generic_full_join_probe"
    }
}
