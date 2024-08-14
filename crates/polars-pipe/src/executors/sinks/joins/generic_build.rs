use std::any::Any;

use arrow::array::BinaryArray;
use hashbrown::hash_map::RawEntryMut;
use polars_core::prelude::*;
use polars_core::utils::{_set_partition_size, accumulate_dataframes_vertical_unchecked};
use polars_ops::prelude::JoinArgs;
use polars_utils::arena::Node;
use polars_utils::slice::GetSaferUnchecked;
use polars_utils::unitvec;
use smartstring::alias::String as SmartString;

use super::*;
use crate::executors::operators::PlaceHolder;
use crate::executors::sinks::joins::generic_probe_inner_left::GenericJoinProbe;
use crate::executors::sinks::joins::generic_probe_outer::GenericFullOuterJoinProbe;
use crate::executors::sinks::utils::{hash_rows, load_vec};
use crate::executors::sinks::HASHMAP_INIT_SIZE;
use crate::expressions::PhysicalPipedExpr;
use crate::operators::{DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult};

pub(super) type ChunkIdx = IdxSize;
pub(super) type DfIdx = IdxSize;

pub struct GenericBuild<K: ExtraPayload> {
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
    hb: PlRandomState,
    join_args: JoinArgs,
    // partitioned tables that will be used for probing
    // stores the key and the chunk_idx, df_idx of the left table
    hash_tables: PartitionedMap<K>,

    // the columns that will be joined on
    join_columns_left: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
    join_columns_right: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,

    // amortize allocations
    join_columns: Vec<ArrayRef>,
    hashes: Vec<u64>,
    // the join order is swapped to ensure we hash the smaller table
    swapped: bool,
    join_nulls: bool,
    node: Node,
    key_names_left: Arc<[SmartString]>,
    key_names_right: Arc<[SmartString]>,
    placeholder: PlaceHolder,
}

impl<K: ExtraPayload> GenericBuild<K> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        suffix: Arc<str>,
        join_args: JoinArgs,
        swapped: bool,
        join_columns_left: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
        join_columns_right: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
        join_nulls: bool,
        node: Node,
        key_names_left: Arc<[SmartString]>,
        key_names_right: Arc<[SmartString]>,
        placeholder: PlaceHolder,
    ) -> Self {
        let hb: PlRandomState = Default::default();
        let partitions = _set_partition_size();
        let hash_tables = PartitionedHashMap::new(load_vec(partitions, || {
            PlIdHashMap::with_capacity(HASHMAP_INIT_SIZE)
        }));
        GenericBuild {
            chunks: vec![],
            join_args,
            suffix,
            hb,
            swapped,
            join_columns_left,
            join_columns_right,
            join_columns: vec![],
            materialized_join_cols: vec![],
            hash_tables,
            hashes: vec![],
            join_nulls,
            node,
            key_names_left,
            key_names_right,
            placeholder,
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
        let (chunk_idx, df_idx) = key.idx.extract();
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

impl<K: ExtraPayload> GenericBuild<K> {
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
            let s = phys_e.evaluate(chunk, &context.execution_state)?;
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

impl<K: ExtraPayload> Sink for GenericBuild<K> {
    fn node(&self) -> Node {
        self.node
    }
    fn is_join_build(&self) -> bool {
        true
    }

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
            let entry = self.hash_tables.raw_entry_mut(*h).from_hash(*h, |key| {
                compare_fn(key, *h, &self.materialized_join_cols, row)
            });

            let payload = ChunkId::store(current_chunk_offset, current_df_idx);
            match entry {
                RawEntryMut::Vacant(entry) => {
                    let key = Key::new(*h, current_chunk_offset, current_df_idx);
                    entry.insert(key, (unitvec![payload], Default::default()));
                },
                RawEntryMut::Occupied(mut entry) => {
                    entry.get_mut().0.push(payload);
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
            .inner_mut()
            .iter_mut()
            .zip(other.hash_tables.inner())
            .for_each(|(ht, other_ht)| {
                for (k, val) in other_ht.iter() {
                    let val = &val.0;
                    let (chunk_idx, df_idx) = k.idx.extract();
                    // Use the indexes to materialize the row.
                    let other_row = unsafe { other.get_row(chunk_idx, df_idx) };

                    let h = k.hash;
                    let entry = ht.raw_entry_mut().from_hash(h, |key| {
                        compare_fn(key, h, &self.materialized_join_cols, other_row)
                    });

                    match entry {
                        RawEntryMut::Vacant(entry) => {
                            let chunk_id = unsafe { val.get_unchecked_release(0) };
                            let (chunk_idx, df_idx) = chunk_id.extract();
                            let new_chunk_idx = chunk_idx + chunks_offset;
                            let key = Key::new(h, new_chunk_idx, df_idx);
                            let mut payload = unitvec![ChunkId::store(new_chunk_idx, df_idx)];
                            if val.len() > 1 {
                                let iter = val[1..].iter().map(|chunk_id| {
                                    let (chunk_idx, val_idx) = chunk_id.extract();
                                    ChunkId::store(chunk_idx + chunks_offset, val_idx)
                                });
                                payload.extend(iter);
                            }
                            entry.insert(key, (payload, Default::default()));
                        },
                        RawEntryMut::Occupied(mut entry) => {
                            let iter = val.iter().map(|chunk_id| {
                                let (chunk_idx, val_idx) = chunk_id.extract();
                                ChunkId::store(chunk_idx + chunks_offset, val_idx)
                            });
                            entry.get_mut().0.extend(iter);
                        },
                    }
                }
            })
    }

    fn split(&self, _thread_no: usize) -> Box<dyn Sink> {
        let mut new = Self::new(
            self.suffix.clone(),
            self.join_args.clone(),
            self.swapped,
            self.join_columns_left.clone(),
            self.join_columns_right.clone(),
            self.join_nulls,
            self.node,
            self.key_names_left.clone(),
            self.key_names_right.clone(),
            self.placeholder.clone(),
        );
        new.hb = self.hb.clone();
        Box::new(new)
    }

    fn finalize(&mut self, context: &PExecutionContext) -> PolarsResult<FinalizedSink> {
        let chunks_len = self.chunks.len();
        let left_df = accumulate_dataframes_vertical_unchecked(
            std::mem::take(&mut self.chunks)
                .into_iter()
                .map(|chunk| chunk.data),
        );
        if left_df.height() > 0 {
            assert_eq!(left_df.n_chunks(), chunks_len);
        }
        // Reallocate to Arc<[]> to get rid of double indirection as this is accessed on every
        // hashtable cmp.
        let materialized_join_cols = Arc::from(std::mem::take(&mut self.materialized_join_cols));
        let suffix = self.suffix.clone();
        let hb = self.hb.clone();
        let hash_tables = Arc::new(PartitionedHashMap::new(std::mem::take(
            self.hash_tables.inner_mut(),
        )));
        let join_columns_left = self.join_columns_left.clone();
        let join_columns_right = self.join_columns_right.clone();

        // take the buffers, this saves one allocation
        let mut hashes = std::mem::take(&mut self.hashes);
        hashes.clear();

        match self.join_args.how {
            JoinType::Inner | JoinType::Left => {
                let probe_operator = GenericJoinProbe::new(
                    left_df,
                    materialized_join_cols,
                    suffix,
                    hb,
                    hash_tables,
                    join_columns_left,
                    join_columns_right,
                    self.swapped,
                    hashes,
                    context,
                    self.join_args.clone(),
                    self.join_nulls,
                );
                self.placeholder.replace(Box::new(probe_operator));
                Ok(FinalizedSink::Operator)
            },
            JoinType::Full => {
                let coalesce = self.join_args.coalesce.coalesce(&JoinType::Full);
                let probe_operator = GenericFullOuterJoinProbe::new(
                    left_df,
                    materialized_join_cols,
                    suffix,
                    hb,
                    hash_tables,
                    join_columns_left,
                    self.swapped,
                    hashes,
                    self.join_nulls,
                    coalesce,
                    self.key_names_left.clone(),
                    self.key_names_right.clone(),
                );
                self.placeholder.replace(Box::new(probe_operator));
                Ok(FinalizedSink::Operator)
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
