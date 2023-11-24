use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow::array::{
    BinaryArray, ArrayRef
};
use hashbrown::hash_map::RawEntryMut;
use polars_core::datatypes::ChunkId;
use polars_core::error::PolarsResult;
use polars_core::export::ahash::RandomState;
use polars_core::prelude::*;
use polars_core::utils::{_set_partition_size, accumulate_dataframes_vertical_unchecked};
use polars_utils::hashing::hash_to_partition;
use polars_utils::slice::GetSaferUnchecked;

use super::*;
use crate::executors::sinks::joins::inner_left::GenericJoinProbe;
use crate::executors::sinks::utils::{hash_rows, load_vec};
use crate::executors::sinks::HASHMAP_INIT_SIZE;
use crate::expressions::PhysicalPipedExpr;
use crate::operators::{DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult};

pub(super) type ChunkIdx = IdxSize;
pub(super) type DfIdx = IdxSize;

// This is the hash and the Index offset in the chunks and the index offset in the dataframe
#[derive(Copy, Clone, Debug)]
pub(super) struct Key {
    pub(super) hash: u64,
    chunk_idx: IdxSize,
    df_idx: IdxSize,
}

impl Key {
    #[inline]
    fn new(hash: u64, chunk_idx: IdxSize, df_idx: IdxSize) -> Self {
        Key {
            hash,
            chunk_idx,
            df_idx,
        }
    }
}

impl Hash for Key {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash)
    }
}

pub struct GenericBuild {
    chunks: Vec<DataChunk>,
    // the join columns are all tightly packed
    // the values of a join column(s) can be found
    // by:
    // first get the offset of the chunks and multiply that with the number of join
    // columns
    //      * chunk_offset = (idx * n_join_keys)
    //      * end = (offset + n_join_keys)
    materialized_join_cols: Vec<BinaryArray<i64>>,
    suffix: Arc<str>,
    hb: RandomState,
    // partitioned tables that will be used for probing
    // stores the key and the chunk_idx, df_idx of the left table
    hash_tables: Vec<PlIdHashMap<Key, Vec<ChunkId>>>,

    // the columns that will be joined on
    join_columns_left: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
    join_columns_right: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,

    // amortize allocations
    join_columns: Vec<ArrayRef>,
    hashes: Vec<u64>,
    join_type: JoinType,
    // the join order is swapped to ensure we hash the smaller table
    swapped: bool,
}

impl GenericBuild {
    pub(crate) fn new(
        suffix: Arc<str>,
        join_type: JoinType,
        swapped: bool,
        join_columns_left: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
        join_columns_right: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
    ) -> Self {
        let hb: RandomState = Default::default();
        let partitions = _set_partition_size();
        let hash_tables = load_vec(partitions, || PlIdHashMap::with_capacity(HASHMAP_INIT_SIZE));
        GenericBuild {
            chunks: vec![],
            join_type,
            suffix,
            hb,
            swapped,
            join_columns_left,
            join_columns_right,
            join_columns: vec![],
            materialized_join_cols: vec![],
            hash_tables,
            hashes: vec![],
        }
    }
}

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
        let chunk_idx = key.chunk_idx as usize;
        let df_idx = key.df_idx as usize;

        // get the right columns from the linearly packed buffer
        let other_row = unsafe {
            join_columns_all_chunks
                .get_unchecked_release(chunk_idx)
                .value_unchecked(df_idx)
        };
        current_row == other_row
    }
}

impl GenericBuild {
    fn is_empty(&self) -> bool {
        match self.chunks.len() {
            0 => true,
            1 => self.chunks[0].is_empty(),
            _ => false,
        }
    }

    fn set_join_series(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<&BinaryArray<i64>> {
        debug_assert!(self.join_columns.is_empty());
        for phys_e in self.join_columns_left.iter() {
            let s = phys_e.evaluate(chunk, context.execution_state.as_any())?;
            let arr = s.to_physical_repr().rechunk().array_ref(0).clone();
            self.join_columns.push(arr);
        }
        let rows_encoded = polars_row::convert_columns_no_order(&self.join_columns).into_array();
        self.materialized_join_cols.push(rows_encoded);
        Ok(self.materialized_join_cols.last().unwrap())
    }
    unsafe fn get_row(&self, chunk_idx: ChunkIdx, df_idx: DfIdx) -> &[u8] {
        self.materialized_join_cols
            .get_unchecked_release(chunk_idx as usize)
            .value_unchecked(df_idx as usize)
    }
}

impl Sink for GenericBuild {
    fn sink(&mut self, context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult> {
        // we do some juggling here so that we don't
        // end up with empty chunks
        // But we always want one empty chunk if all is empty as we need
        // to finish the join
        if self.chunks.len() == 1 && self.chunks[0].is_empty() {
            self.chunks.pop().unwrap();
        }
        if chunk.is_empty() {
            if self.chunks.is_empty() {
                self.chunks.push(chunk)
            }
            return Ok(SinkResult::CanHaveMoreInput);
        }
        let mut hashes = std::mem::take(&mut self.hashes);
        let rows = self.set_join_series(context, &chunk)?.clone();
        hash_rows(&rows, &mut hashes, &self.hb);
        self.hashes = hashes;

        let current_chunk_offset = self.chunks.len() as ChunkIdx;

        // row offset in the chunk belonging to the hash
        let mut current_df_idx = 0 as IdxSize;
        for (row, h) in rows.values_iter().zip(&self.hashes) {
            // get the hashtable belonging to this hash partition
            let partition = hash_to_partition(*h, self.hash_tables.len());
            let current_table = unsafe { self.hash_tables.get_unchecked_release_mut(partition) };

            let entry = current_table.raw_entry_mut().from_hash(*h, |key| {
                compare_fn(key, *h, &self.materialized_join_cols, row)
            });

            let payload = [current_chunk_offset, current_df_idx];
            match entry {
                RawEntryMut::Vacant(entry) => {
                    let key = Key::new(*h, current_chunk_offset, current_df_idx);
                    entry.insert(key, vec![payload]);
                },
                RawEntryMut::Occupied(mut entry) => {
                    entry.get_mut().push(payload);
                },
            };

            current_df_idx += 1;
        }

        // clear memory
        self.hashes.clear();
        self.join_columns.clear();

        self.chunks.push(chunk);
        Ok(SinkResult::CanHaveMoreInput)
    }

    fn combine(&mut self, other: &mut dyn Sink) {
        if self.is_empty() {
            let other = other.as_any().downcast_mut::<Self>().unwrap();
            if !other.is_empty() {
                std::mem::swap(self, other);
            }
            return;
        }
        let other = other.as_any().downcast_ref::<Self>().unwrap();
        if other.is_empty() {
            return;
        }

        let chunks_offset = self.chunks.len() as IdxSize;
        self.chunks.extend_from_slice(&other.chunks);
        self.materialized_join_cols
            .extend_from_slice(&other.materialized_join_cols);

        // we combine the other hashtable with ours, but we must offset the chunk_idx
        // values by the number of chunks we already got.
        self.hash_tables
            .iter_mut()
            .zip(&other.hash_tables)
            .for_each(|(ht, other_ht)| {
                for (k, val) in other_ht.iter() {
                    // use the indexes to materialize the row
                    let other_row = unsafe { other.get_row(k.chunk_idx, k.df_idx) };

                    let h = k.hash;
                    let entry = ht.raw_entry_mut().from_hash(h, |key| {
                        compare_fn(key, h, &self.materialized_join_cols, other_row)
                    });

                    match entry {
                        RawEntryMut::Vacant(entry) => {
                            let [chunk_idx, df_idx] = unsafe { val.get_unchecked_release(0) };
                            let new_chunk_idx = chunk_idx + chunks_offset;
                            let key = Key::new(h, new_chunk_idx, *df_idx);
                            let mut payload = vec![[new_chunk_idx, *df_idx]];
                            if val.len() > 1 {
                                let iter = val[1..].iter().map(|[chunk_idx, val_idx]| {
                                    [*chunk_idx + chunks_offset, *val_idx]
                                });
                                payload.extend(iter);
                            }
                            entry.insert(key, payload);
                        },
                        RawEntryMut::Occupied(mut entry) => {
                            let iter = val
                                .iter()
                                .map(|[chunk_idx, val_idx]| [*chunk_idx + chunks_offset, *val_idx]);
                            entry.get_mut().extend(iter);
                        },
                    }
                }
            })
    }

    fn split(&self, _thread_no: usize) -> Box<dyn Sink> {
        let mut new = Self::new(
            self.suffix.clone(),
            self.join_type.clone(),
            self.swapped,
            self.join_columns_left.clone(),
            self.join_columns_right.clone(),
        );
        new.hb = self.hb.clone();
        Box::new(new)
    }

    fn finalize(&mut self, context: &PExecutionContext) -> PolarsResult<FinalizedSink> {
        match self.join_type {
            JoinType::Inner | JoinType::Left => {
                let chunks_len = self.chunks.len();
                let left_df = accumulate_dataframes_vertical_unchecked(
                    std::mem::take(&mut self.chunks)
                        .into_iter()
                        .map(|chunk| chunk.data),
                );
                if left_df.height() > 0 {
                    assert_eq!(left_df.n_chunks(), chunks_len);
                }
                let materialized_join_cols =
                    Arc::new(std::mem::take(&mut self.materialized_join_cols));
                let suffix = self.suffix.clone();
                let hb = self.hb.clone();
                let hash_tables = Arc::new(std::mem::take(&mut self.hash_tables));
                let join_columns_left = self.join_columns_left.clone();
                let join_columns_right = self.join_columns_right.clone();

                // take the buffers, this saves one allocation
                let mut join_series = std::mem::take(&mut self.join_columns);
                join_series.clear();
                let mut hashes = std::mem::take(&mut self.hashes);
                hashes.clear();

                let probe_operator = GenericJoinProbe::new(
                    left_df,
                    materialized_join_cols,
                    suffix,
                    hb,
                    hash_tables,
                    join_columns_left,
                    join_columns_right,
                    self.swapped,
                    join_series,
                    hashes,
                    context,
                    self.join_type.clone(),
                );
                Ok(FinalizedSink::Operator(Box::new(probe_operator)))
            },
            _ => unimplemented!(),
        }
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
    fn fmt(&self) -> &str {
        "generic_join_build"
    }
}
