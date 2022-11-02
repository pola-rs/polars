use std::any::Any;
use std::sync::Arc;

use hashbrown::hash_map::RawEntryMut;
use polars_core::error::PolarsResult;
use polars_core::export::ahash::RandomState;
use polars_core::frame::hash_join::ChunkId;
use polars_core::prelude::*;
use polars_utils::hash_to_partition;
use polars_utils::slice::GetSaferUnchecked;
use polars_utils::unwrap::UnwrapUncheckedRelease;

use crate::executors::sinks::utils::hash_series;
use crate::expressions::PhysicalPipedExpr;
use crate::operators::{DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult};

type ChunkIdx = IdxSize;
type DfIdx = IdxSize;
// This is the hash and the Index offset in the chunks and the index offset in the dataframe
type Key = (u64, ChunkIdx, DfIdx);

pub struct GenericBuild {
    chunks: Vec<DataChunk>,
    // the join columns are all tightly packed
    // the values of a join column(s) can be found
    // by:
    // first get the offset of the chunks and multiply that with the number of join
    // columns
    //      * chunk_offset = (idx * n_join_keys)
    //      * end = (offset + n_join_keys)
    materialized_join_cols: Vec<ArrayRef>,
    suffix: String,
    hb: RandomState,
    // partitioned tables that will be used for probing
    // stores the key and the chunk_idx, df_idx of the left table
    hash_tables: Vec<PlHashMap<Key, Vec<(ChunkIdx, DfIdx)>>>,

    // the columns that will be joined on
    join_columns: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
    join_names: Arc<Vec<String>>,

    // amortize allocations
    join_series: Vec<Series>,
    hashes: Vec<u64>,
}

fn compare_fn(
    key: &Key,
    h: u64,
    join_columns_all_chunks: &[ArrayRef],
    current_row: &[AnyValue],
    n_join_cols: usize,
) -> bool {
    let key_hash = key.0;

    let chunk_idx = key.1 as usize * n_join_cols;
    let df_idx = key.2 as usize;

    // get the right columns from the linearly packed buffer
    let join_cols = unsafe {
        join_columns_all_chunks.get_unchecked_release(chunk_idx..chunk_idx + n_join_cols)
    };

    // we check the hash and
    // we get the appropriate values from the join columns and compare it with the current row
    key_hash == h && {
        join_cols
            .iter()
            .zip(current_row)
            .all(|(column, value)| unsafe { &column.get_unchecked(df_idx) == value })
    }
}

impl GenericBuild {
    #[inline]
    fn number_of_keys(&self) -> usize {
        self.join_columns.len()
    }

    fn set_join_series(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<&[Series]> {
        self.join_series.clear();
        for phys_e in self.join_columns.iter() {
            let s = phys_e.evaluate(chunk, context.execution_state.as_ref())?;
            let s = s.to_physical_repr();
            self.join_series.push(s.rechunk());
        }
        Ok(&self.join_series)
    }
    unsafe fn get_tuple<'a>(
        &'a self,
        chunk_idx: ChunkIdx,
        df_idx: DfIdx,
        buf: &mut Vec<AnyValue<'a>>,
    ) {
        buf.clear();
        // get the right columns from the linearly packed buffer
        let join_cols = self
            .materialized_join_cols
            .get_unchecked_release(chunk_idx as usize..chunk_idx as usize + self.number_of_keys());
        buf.extend(
            join_cols
                .iter()
                .map(|arr| arr.get_unchecked(df_idx as usize)),
        )
    }
}

impl Sink for GenericBuild {
    fn sink(&mut self, context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult> {
        let mut hashes = std::mem::take(&mut self.hashes);
        self.set_join_series(context, &chunk)?;
        hash_series(&self.join_series, &mut hashes, &self.hb);
        self.hashes = hashes;

        let current_chunk_offset = self.chunks.len() as ChunkIdx;

        // iterators over anyvalues
        let mut key_iters = self
            .join_series
            .iter()
            .map(|s| s.phys_iter())
            .collect::<Vec<_>>();

        // a small buffer that holds the current key values
        // if we join by 2 keys, this holds 2 anyvalues.
        let mut current_keys_buf = Vec::with_capacity(self.number_of_keys());
        let n_join_cols = self.number_of_keys();

        // row offset in the chunk belonging to the hash
        let mut current_df_idx = 0 as IdxSize;
        for h in &self.hashes {
            // load the keys in the buffer
            current_keys_buf.clear();
            for key_iter in key_iters.iter_mut() {
                unsafe { current_keys_buf.push(key_iter.next().unwrap_unchecked_release()) }
            }

            // get the hashtable belonging by this hash partition
            let partition = hash_to_partition(*h, self.hash_tables.len());
            let current_table = unsafe { self.hash_tables.get_unchecked_release_mut(partition) };

            let entry = current_table.raw_entry_mut().from_hash(*h, |key| {
                compare_fn(
                    key,
                    *h,
                    &self.materialized_join_cols,
                    &current_keys_buf,
                    n_join_cols,
                )
            });

            let payload = (current_chunk_offset, current_df_idx);
            match entry {
                RawEntryMut::Vacant(entry) => {
                    let key = (*h, current_chunk_offset, current_df_idx);
                    entry.insert(key, vec![payload]);
                }
                RawEntryMut::Occupied(mut entry) => {
                    entry.get_mut().push(payload);
                }
            };

            current_df_idx += 1;
        }
        self.chunks.push(chunk);
        Ok(SinkResult::CanHaveMoreInput)
    }

    fn combine(&mut self, mut other: Box<dyn Sink>) {
        let other = other.as_any().downcast_ref::<Self>().unwrap();
        let mut tuple_buf = Vec::with_capacity(self.number_of_keys());

        let chunks_offset = self.chunks.len() as IdxSize;
        self.chunks.extend_from_slice(&other.chunks);
        self.materialized_join_cols
            .extend_from_slice(&other.materialized_join_cols);

        // we combine the other hashtable with ours, but we must offset the chunk_idx
        // values by the number of chunks we already got.
        for (ht, other_ht) in self.hash_tables.iter_mut().zip(&other.hash_tables) {
            for (k, val) in other_ht.iter() {
                // use the indexes to materialize the row
                for (chunk_idx, df_idx) in val {
                    unsafe { other.get_tuple(*chunk_idx, *df_idx, &mut tuple_buf) };
                }

                let h = k.0;
                let entry = ht.raw_entry_mut().from_hash(h, |key| {
                    compare_fn(
                        key,
                        h,
                        &self.materialized_join_cols,
                        &tuple_buf,
                        tuple_buf.len(),
                    )
                });

                match entry {
                    RawEntryMut::Vacant(entry) => {
                        let (chunk_idx, df_idx) = unsafe { val.get_unchecked_release(0) };
                        let new_chunk_idx = chunk_idx + chunks_offset;
                        let key = (h, new_chunk_idx, *df_idx);
                        let mut payload = vec![(new_chunk_idx, *df_idx)];
                        if val.len() > 1 {
                            let iter = val[1..]
                                .iter()
                                .map(|(chunk_idx, val_idx)| (*chunk_idx + chunks_offset, *val_idx));
                            payload.extend(iter);
                        }
                        entry.insert(key, payload);
                    }
                    RawEntryMut::Occupied(mut entry) => {
                        let iter = val
                            .iter()
                            .map(|(chunk_idx, val_idx)| (*chunk_idx + 1, *val_idx));
                        entry.get_mut().extend(iter);
                    }
                }
            }
        }
    }

    fn split(&self, thread_no: usize) -> Box<dyn Sink> {
        todo!()
    }

    fn finalize(&mut self) -> PolarsResult<FinalizedSink> {
        todo!()
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}
