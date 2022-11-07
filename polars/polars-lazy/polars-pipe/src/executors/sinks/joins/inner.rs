use std::borrow::Cow;
use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::export::ahash::RandomState;
use polars_core::frame::hash_join::{ChunkId, _finish_join};
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_utils::hash_to_partition;
use polars_utils::slice::GetSaferUnchecked;
use polars_utils::unwrap::UnwrapUncheckedRelease;

use crate::executors::sinks::joins::generic_build::*;
use crate::executors::sinks::utils::hash_series;
use crate::expressions::PhysicalPipedExpr;
use crate::operators::{DataChunk, Operator, OperatorResult, PExecutionContext};

#[derive(Clone)]
pub struct InnerJoinProbe {
    // all chunks are stacked into a single dataframe
    // the dataframe is not rechunked.
    pub(super) left_df: Arc<DataFrame>,
    // the join columns are all tightly packed
    // the values of a join column(s) can be found
    // by:
    // first get the offset of the chunks and multiply that with the number of join
    // columns
    //      * chunk_offset = (idx * n_join_keys)
    //      * end = (offset + n_join_keys)
    pub(super) materialized_join_cols: Arc<Vec<ArrayRef>>,
    pub(super) suffix: Arc<str>,
    pub(super) hb: RandomState,
    // partitioned tables that will be used for probing
    // stores the key and the chunk_idx, df_idx of the left table
    pub(super) hash_tables: Arc<Vec<PlIdHashMap<Key, Vec<ChunkId>>>>,

    // the columns that will be joined on
    pub(super) join_columns_right: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,

    // amortize allocations
    pub(super) join_series: Vec<Series>,
    pub(super) join_tuples_left: Vec<ChunkId>,
    pub(super) join_tuples_right: Vec<DfIdx>,
    pub(super) hashes: Vec<u64>,
    // the join order is swapped to ensure we hash the smaller table
    pub(super) swapped: bool,
    // location of join columns.
    // these column locations need to be dropped from the rhs
    pub(super) join_column_idx: Option<Vec<usize>>,
}

impl InnerJoinProbe {
    #[inline]
    fn number_of_keys(&self) -> usize {
        self.join_columns_right.len()
    }

    fn set_join_series(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<&[Series]> {
        self.join_series.clear();
        for phys_e in self.join_columns_right.iter() {
            let s = phys_e.evaluate(chunk, context.execution_state.as_ref())?;
            let s = s.to_physical_repr();
            self.join_series.push(s.rechunk());
        }

        if self.join_column_idx.is_none() {
            let mut idx = self
                .join_series
                .iter()
                .filter_map(|s| chunk.data.find_idx_by_name(s.name()))
                .collect::<Vec<_>>();
            // ensure that it is sorted so that we can later remove columns in
            // a predictable order
            idx.sort_unstable();
            self.join_column_idx = Some(idx);
        }

        Ok(&self.join_series)
    }
}

impl Operator for InnerJoinProbe {
    fn execute(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        self.join_tuples_left.clear();
        self.join_tuples_right.clear();
        let mut hashes = std::mem::take(&mut self.hashes);
        self.set_join_series(context, chunk)?;
        hash_series(&self.join_series, &mut hashes, &self.hb);
        self.hashes = hashes;

        // iterators over anyvalues
        let mut key_iters = self
            .join_series
            .iter()
            .map(|s| s.phys_iter())
            .collect::<Vec<_>>();

        // a small buffer that holds the current key values
        // if we join by 2 keys, this holds 2 anyvalues.
        let mut current_tuple_buf = Vec::with_capacity(self.number_of_keys());

        for (i, h) in self.hashes.iter().enumerate() {
            let df_idx_right = i as IdxSize;

            // load the keys in the buffer
            current_tuple_buf.clear();
            for key_iter in key_iters.iter_mut() {
                unsafe { current_tuple_buf.push(key_iter.next().unwrap_unchecked_release()) }
            }
            // get the hashtable belonging by this hash partition
            let partition = hash_to_partition(*h, self.hash_tables.len());
            let current_table = unsafe { self.hash_tables.get_unchecked_release(partition) };

            let entry = current_table
                .raw_entry()
                .from_hash(*h, |key| {
                    compare_fn(
                        key,
                        *h,
                        &self.materialized_join_cols,
                        &current_tuple_buf,
                        current_tuple_buf.len(),
                    )
                })
                .map(|key_val| key_val.1);

            if let Some(indexes_left) = entry {
                self.join_tuples_left.extend_from_slice(indexes_left);
                self.join_tuples_right
                    .extend(std::iter::repeat(df_idx_right).take(indexes_left.len()));
            }
        }

        let left_df = unsafe {
            self.left_df
                ._take_chunked_unchecked_seq(&self.join_tuples_left, IsSorted::Not)
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
            df._take_unchecked_slice(&self.join_tuples_right, false)
        };

        let (a, b) = if self.swapped {
            (right_df, left_df)
        } else {
            (left_df, right_df)
        };
        let out = _finish_join(a, b, Some(self.suffix.as_ref()))?;

        Ok(OperatorResult::Finished(chunk.with_data(out)))
    }

    fn split(&self, _thread_no: usize) -> Box<dyn Operator> {
        let new = self.clone();
        Box::new(new)
    }
}
