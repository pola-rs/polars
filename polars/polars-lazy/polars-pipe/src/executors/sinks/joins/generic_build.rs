use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use hashbrown::hash_map::RawEntryMut;
use polars_arrow::trusted_len::PushUnchecked;
use polars_core::error::PolarsResult;
use polars_core::export::ahash::RandomState;
use polars_core::frame::hash_join::ChunkId;
use polars_core::prelude::*;
use polars_core::utils::{_set_partition_size, accumulate_dataframes_vertical_unchecked};
use polars_utils::hash_to_partition;
use polars_utils::slice::GetSaferUnchecked;
use polars_utils::unwrap::UnwrapUncheckedRelease;

use crate::executors::sinks::joins::inner_left::GenericJoinProbe;
use crate::executors::sinks::utils::{hash_series, load_vec};
use crate::executors::sinks::HASHMAP_INIT_SIZE;
use crate::expressions::PhysicalPipedExpr;
use crate::operators::{DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult};

pub(super) type ChunkIdx = IdxSize;
pub(super) type DfIdx = IdxSize;

// This is the hash and the Index offset in the chunks and the index offset in the dataframe
#[derive(Copy, Clone, Debug)]
pub(super) struct Key {
    pub(super) hash: u64,
    chunk_idx: ChunkIdx,
    df_idx: DfIdx,
}

impl Key {
    #[inline]
    fn new(hash: u64, chunk_idx: ChunkIdx, df_idx: DfIdx) -> Self {
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
    materialized_join_cols: Vec<ArrayRef>,
    suffix: Arc<str>,
    hb: RandomState,
    // partitioned tables that will be used for probing
    // stores the key and the chunk_idx, df_idx of the left table
    hash_tables: Vec<PlIdHashMap<Key, Vec<ChunkId>>>,

    // the columns that will be joined on
    join_columns_left: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
    join_columns_right: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,

    // amortize allocations
    join_series: Vec<Series>,
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
            join_series: vec![],
            materialized_join_cols: vec![],
            hash_tables,
            hashes: vec![],
        }
    }
}

pub(super) fn compare_fn(
    key: &Key,
    h: u64,
    join_columns_all_chunks: &[ArrayRef],
    current_row: &[AnyValue],
    n_join_cols: usize,
) -> bool {
    let key_hash = key.hash;

    // we check the hash first
    // as that has no indirection
    key_hash == h && {
        // we get the appropriate values from the join columns and compare it with the current row
        let chunk_idx = key.chunk_idx as usize * n_join_cols;
        let df_idx = key.df_idx as usize;

        // get the right columns from the linearly packed buffer
        let join_cols = unsafe {
            join_columns_all_chunks.get_unchecked_release(chunk_idx..chunk_idx + n_join_cols)
        };

        join_cols
            .iter()
            .zip(current_row)
            .all(|(column, value)| unsafe { &column.get_unchecked(df_idx) == value })
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

    #[inline]
    fn number_of_keys(&self) -> usize {
        self.join_columns_left.len()
    }

    fn set_join_series(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<&[Series]> {
        self.join_series.clear();

        for phys_e in self.join_columns_left.iter() {
            let s = phys_e.evaluate(chunk, context.execution_state.as_any())?;
            let s = s.to_physical_repr();
            let s = s.rechunk();
            self.materialized_join_cols.push(s.array_ref(0).clone());
            self.join_series.push(s);
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
        let n_keys = self.number_of_keys();
        let chunk_offset = chunk_idx as usize * n_keys;
        let chunk_end = chunk_offset + n_keys;
        let join_cols = self
            .materialized_join_cols
            .get_unchecked_release(chunk_offset..chunk_end);
        buf.extend(
            join_cols
                .iter()
                .map(|arr| arr.get_unchecked(df_idx as usize)),
        )
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
        self.set_join_series(context, &chunk)?;
        hash_series(&self.join_series, &mut hashes, &self.hb);
        self.hashes = hashes;

        let current_chunk_offset = self.chunks.len() as ChunkIdx;

        let mut keys_iter = KeysIter::new(&self.join_series);
        let n_join_cols = self.number_of_keys();

        // row offset in the chunk belonging to the hash
        let mut current_df_idx = 0 as IdxSize;
        for h in &self.hashes {
            let current_tuple = unsafe { keys_iter.lend_next() };

            // get the hashtable belonging by this hash partition
            let partition = hash_to_partition(*h, self.hash_tables.len());
            let current_table = unsafe { self.hash_tables.get_unchecked_release_mut(partition) };

            let entry = current_table.raw_entry_mut().from_hash(*h, |key| {
                compare_fn(
                    key,
                    *h,
                    &self.materialized_join_cols,
                    current_tuple,
                    n_join_cols,
                )
            });

            let payload = [current_chunk_offset, current_df_idx];
            match entry {
                RawEntryMut::Vacant(entry) => {
                    let key = Key::new(*h, current_chunk_offset, current_df_idx);
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
        let mut tuple_buf = Vec::with_capacity(self.number_of_keys());

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
                    for [chunk_idx, df_idx] in val {
                        unsafe { other.get_tuple(*chunk_idx, *df_idx, &mut tuple_buf) };
                    }

                    let h = k.hash;
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
                        }
                        RawEntryMut::Occupied(mut entry) => {
                            let iter = val
                                .iter()
                                .map(|[chunk_idx, val_idx]| [*chunk_idx + chunks_offset, *val_idx]);
                            entry.get_mut().extend(iter);
                        }
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
                let mut join_series = std::mem::take(&mut self.join_series);
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
            }
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

pub(super) struct KeysIter<'a> {
    key_iters: Vec<Box<dyn ExactSizeIterator<Item = AnyValue<'a>> + 'a>>,
    // a small buffer that holds the current key values
    // if we join by 2 keys, this holds 2 anyvalues.
    buf: Vec<AnyValue<'a>>,
}

impl<'a> KeysIter<'a> {
    pub(super) fn new(join_series: &'a [Series]) -> Self {
        // iterators over anyvalues
        let key_iters = join_series
            .iter()
            .map(|s| s.phys_iter())
            .collect::<Vec<_>>();

        // ensure that they have the appriate size as we will not bound check on push
        let buf = Vec::with_capacity(key_iters.len());
        Self { key_iters, buf }
    }

    /// # Safety
    /// will not check any bounds on iterators. `lend_next` should not be called more often
    /// than items in the given iterators.
    pub(super) unsafe fn lend_next<'b>(&'b mut self) -> &'b [AnyValue<'a>] {
        self.buf.clear();
        for key_iter in self.key_iters.iter_mut() {
            // safety: we allocated up front
            self.buf
                .push_unchecked(key_iter.next().unwrap_unchecked_release())
        }
        &self.buf
    }
}
