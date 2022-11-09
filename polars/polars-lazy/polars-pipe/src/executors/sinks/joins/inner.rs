use std::borrow::Cow;
use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::export::ahash::RandomState;
use polars_core::frame::hash_join::{ChunkId, _finish_join};
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_utils::hash_to_partition;
use polars_utils::slice::GetSaferUnchecked;

use crate::executors::sinks::joins::generic_build::*;
use crate::executors::sinks::utils::hash_series;
use crate::expressions::PhysicalPipedExpr;
use crate::operators::{DataChunk, Operator, OperatorResult, PExecutionContext};

#[derive(Clone)]
pub struct InnerJoinProbe {
    // all chunks are stacked into a single dataframe
    // the dataframe is not rechunked.
    left_df: Arc<DataFrame>,
    // the join columns are all tightly packed
    // the values of a join column(s) can be found
    // by:
    // first get the offset of the chunks and multiply that with the number of join
    // columns
    //      * chunk_offset = (idx * n_join_keys)
    //      * end = (offset + n_join_keys)
    materialized_join_cols: Arc<Vec<ArrayRef>>,
    suffix: Arc<str>,
    hb: RandomState,
    // partitioned tables that will be used for probing
    // stores the key and the chunk_idx, df_idx of the left table
    hash_tables: Arc<Vec<PlIdHashMap<Key, Vec<ChunkId>>>>,

    // the columns that will be joined on
    join_columns_right: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,

    // amortize allocations
    join_series: Vec<Series>,
    join_tuples_left: Vec<ChunkId>,
    join_tuples_right: Vec<DfIdx>,
    hashes: Vec<u64>,
    // the join order is swapped to ensure we hash the smaller table
    swapped: bool,
    // location of join columns.
    // these column locations need to be dropped from the rhs
    join_column_idx: Option<Vec<usize>>,
    // cached output names
    output_names: Option<Vec<String>>,
}

impl InnerJoinProbe {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        mut left_df: DataFrame,
        materialized_join_cols: Arc<Vec<ArrayRef>>,
        suffix: Arc<str>,
        hb: RandomState,
        hash_tables: Arc<Vec<PlIdHashMap<Key, Vec<ChunkId>>>>,
        join_columns_left: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
        join_columns_right: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
        swapped: bool,
        join_series: Vec<Series>,
        hashes: Vec<u64>,
        context: &PExecutionContext,
    ) -> Self {
        if swapped {
            let tmp = DataChunk {
                data: left_df.slice(0, 1),
                chunk_index: 0,
            };
            let names = join_columns_left
                .iter()
                .map(|phys_e| {
                    let s = phys_e
                        .evaluate(&tmp, context.execution_state.as_ref())
                        .unwrap();
                    s.name().to_string()
                })
                .collect::<Vec<_>>();
            left_df = left_df.drop_many(&names)
        }

        InnerJoinProbe {
            left_df: Arc::new(left_df),
            materialized_join_cols,
            suffix,
            hb,
            hash_tables,
            join_columns_right,
            join_series,
            join_tuples_left: vec![],
            join_tuples_right: vec![],
            hashes,
            swapped,
            join_column_idx: None,
            output_names: None,
        }
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

        // we determine the indices of the columns that have to be removed
        // if swapped the join column is already removed from the `build_df` as that will
        // be the rhs one.
        if !self.swapped && self.join_column_idx.is_none() {
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
        let mut keys_iter = KeysIter::new(&self.join_series);

        for (i, h) in self.hashes.iter().enumerate() {
            let df_idx_right = i as IdxSize;
            let current_tuple = unsafe { keys_iter.lend_next() };
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
                        current_tuple,
                        current_tuple.len(),
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

        let (mut a, b) = if self.swapped {
            (right_df, left_df)
        } else {
            (left_df, right_df)
        };
        let out = match &self.output_names {
            None => {
                let out = _finish_join(a, b, Some(self.suffix.as_ref()))?;
                self.output_names = Some(out.get_column_names_owned());
                out
            }
            Some(names) => {
                a.hstack_mut(b.get_columns()).unwrap();
                a.get_columns_mut()
                    .iter_mut()
                    .zip(names)
                    .for_each(|(s, name)| {
                        s.rename(name);
                    });
                a
            }
        };

        Ok(OperatorResult::Finished(chunk.with_data(out)))
    }

    fn split(&self, _thread_no: usize) -> Box<dyn Operator> {
        let new = self.clone();
        Box::new(new)
    }
}
