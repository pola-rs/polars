use std::any::Any;
use std::fmt::Debug;
use std::hash::{BuildHasher, Hash, Hasher};

use hashbrown::hash_map::RawEntryMut;
use polars_core::export::ahash::RandomState;
use polars_core::export::arrow;
use polars_core::frame::row::AnyValueBuffer;
use polars_core::prelude::*;
use polars_core::utils::{_set_partition_size, accumulate_dataframes_vertical, accumulate_dataframes_vertical_unchecked};
use polars_core::utils::arrow::types::NativeType;
use polars_utils::slice::GetSaferUnchecked;
use polars_utils::{debug_unwrap, get_hash, hash_to_partition};

use super::aggregates::AggregateFn;
use crate::operators::{DataChunk, PExecutionContext, Sink, SinkResult};

// We must strike a balance between cache coherence and resizing costs.
// Overallocation seems a lot more expensive than resizing so we start reasonable small.
pub(crate) const HASHMAP_INIT_SIZE: usize = 128;

pub struct PrimitiveGroupbySink<K: PolarsNumericType> {
    thread_no: usize,
    // idx is the offset in the array with aggregators
    pre_agg_partitions: Vec<PlHashMap<Option<K::Native>, usize>>,
    // the aggregations are all tightly packed
    // the aggregation function of a group can be found
    // by:
    // first get the correct vec by the partition index
    //      * offset = (idx)
    //      * end = (offset + n_aggs)
    aggregators: Vec<Vec<Box<dyn AggregateFn>>>,
    key: Arc<str>,
    // index of the columns that will be aggregated
    aggregation_columns: Vec<usize>,
    hb: RandomState,
    // Aggregation functions
    agg_fns: Vec<Box<dyn AggregateFn>>,
}

impl<K: PolarsNumericType> PrimitiveGroupbySink<K> {
    pub fn new(
        key: Arc<str>,
        aggregation_columns: Vec<usize>,
        agg_fns: Vec<Box<dyn AggregateFn>>,
    ) -> Self {
        let hb = RandomState::default();
        let partitions = _set_partition_size();

        let mut pre_agg = Vec::with_capacity(partitions);
        let mut aggregators = Vec::with_capacity(partitions);

        for _ in 0..partitions {
            pre_agg.push(PlHashMap::with_capacity_and_hasher(
                HASHMAP_INIT_SIZE,
                hb.clone(),
            ));
            aggregators.push(Vec::with_capacity(
                HASHMAP_INIT_SIZE * aggregation_columns.len(),
            ));
        }

        Self {
            thread_no: 0,
            pre_agg_partitions: pre_agg,
            aggregators,
            key,
            aggregation_columns,
            hb,
            agg_fns,
        }
    }

    fn number_of_aggs(&self) -> usize {
        self.aggregation_columns.len()
    }
}

impl<K: PolarsNumericType> Sink for PrimitiveGroupbySink<K>
where
    K::Native: Hash + Eq + Debug,
    ChunkedArray<K>: IntoSeries
{
    fn sink(&mut self, context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult> {
        let s = chunk.data.column(self.key.as_ref())?.to_physical_repr();
        // cow -> &series -> &dyn series_trait -> &chunkedarray
        let ca: &ChunkedArray<K> = s.as_ref().as_ref().as_ref();

        // todo! ammortize allocation
        let agg_s = self
            .aggregation_columns
            .iter()
            .map(|i| chunk.data.select_at_idx(*i).unwrap().rechunk())
            .collect::<Vec<_>>();

        let mut agg_iters = agg_s.iter().map(|s| s.iter()).collect::<Vec<_>>();

        for arr in ca.downcast_iter() {
            for opt_v in arr {
                let opt_v = opt_v.copied();
                let h = get_hash(opt_v, &self.hb);
                let part = hash_to_partition(h, self.pre_agg_partitions.len());
                let mut current_part =
                    unsafe { self.pre_agg_partitions.get_unchecked_release_mut(part) };
                let mut current_aggregators =
                    unsafe { self.aggregators.get_unchecked_release_mut(part) };

                let entry = current_part
                    .raw_entry_mut()
                    .from_key_hashed_nocheck(h, &opt_v);
                let agg_idx = match entry {
                    RawEntryMut::Vacant(entry) => {
                        entry.insert_with_hasher(h, opt_v, current_aggregators.len(), |_| h);
                        // initialize the aggregators
                        for agg_fn in &self.agg_fns {
                            current_aggregators.push(agg_fn.split())
                        }
                        0 as usize
                    }
                    RawEntryMut::Occupied(entry) => *entry.get(),
                };
                for (i, agg_iter) in
                    (agg_idx..agg_idx + self.aggregation_columns.len()).zip(agg_iters.iter_mut())
                {
                    let agg_fn = unsafe { current_aggregators.get_unchecked_release_mut(i) };

                    let value = unsafe { debug_unwrap(agg_iter.next()) };
                    agg_fn.pre_agg(chunk.chunk_index, value)
                }
            }
        }
        Ok(SinkResult::CanHaveMoreInput)
    }

    fn combine(&mut self, other: Box<dyn Sink>) {
        // TODO! parallel
        let other = other.as_any().downcast_ref::<Self>().unwrap();

        self.pre_agg_partitions
            .iter_mut()
            .zip(other.pre_agg_partitions.iter())
            .zip(self.aggregators.iter_mut())
            .zip(other.aggregators.iter())
            .for_each(|(((map_self, map_other), agg_fn_self), agg_fn_other)| {
                for (k, agg_idx_other) in map_other.iter() {
                    let opt_v = *k;
                    unsafe {
                        let agg_fn_other = agg_fn_other.get_unchecked_release(*agg_idx_other);

                        let h = get_hash(opt_v, &self.hb);
                        let entry = map_self.raw_entry_mut().from_key_hashed_nocheck(h, &opt_v);

                        let agg_idx_self = match entry {
                            RawEntryMut::Vacant(entry) => {
                                entry.insert_with_hasher(h, opt_v, agg_fn_self.len(), |_| h);
                                // initialize the aggregators
                                for agg_fn in &self.agg_fns {
                                    agg_fn_self.push(agg_fn.split())
                                }
                                0 as usize
                            }
                            RawEntryMut::Occupied(entry) => *entry.get(),
                        };
                        for i in agg_idx_self..agg_idx_self + self.aggregation_columns.len() {
                            let agg_fn_self = agg_fn_self.get_unchecked_release_mut(i);
                            agg_fn_self.combine(agg_fn_other)
                        }
                    }
                }
            });
    }

    fn split(&self, thread_no: usize) -> Box<dyn Sink> {
        let mut new = Self::new(
            self.key.clone(),
            self.aggregation_columns.clone(),
            self.agg_fns.iter().map(|func| func.split()).collect(),
        );
        new.thread_no = thread_no;
        Box::new(new)
    }

    fn finalize(&mut self) -> PolarsResult<DataFrame> {
        // TODO! parallel
        let mut aggregators = std::mem::take(&mut self.aggregators);
        let dfs = self.pre_agg_partitions
            .iter()
            .zip(aggregators.iter_mut())
            .filter_map(|(agg_map, agg_fns)| {
                if agg_map.is_empty() {
                    return None
                }
                let mut key_builder = PrimitiveChunkedBuilder::<K>::new(&self.key, agg_map.len());
                let dtypes = agg_fns
                    .iter()
                    .take(self.number_of_aggs()).map(|func| func.dtype()).collect::<Vec<_>>();

                let mut buffers = dtypes.iter()
                    .map(|dtype| AnyValueBuffer::new(dtype, agg_map.len()))
                    .collect::<Vec<_>>();

                agg_map.into_iter().for_each(|(k, &offset)| {
                    key_builder.append_option(*k);

                    for (i, buffer) in
                        (offset..offset + self.aggregation_columns.len()).zip(buffers.iter_mut())
                    {
                        unsafe {
                            let agg_fn = agg_fns.get_unchecked_release_mut(i);
                            let av = agg_fn.finalize();
                            buffer.add(av);
                        }
                    }
                });

                let mut cols = Vec::with_capacity(1 + self.number_of_aggs());
                cols.push(key_builder.finish().into_series());
                cols.extend(
                    buffers.into_iter().map(|buf| buf.into_series())
                );
                Some(DataFrame::new(cols))
            }).collect::<PolarsResult<Vec<_>>>()?;
        Ok(accumulate_dataframes_vertical_unchecked(dfs))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

unsafe impl<K: PolarsNumericType> Sync for PrimitiveGroupbySink<K> {}
