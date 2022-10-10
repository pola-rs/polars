use std::any::Any;
use std::fmt::Debug;
use std::hash::Hash;

use hashbrown::hash_map::RawEntryMut;
use num::NumCast;
use polars_core::export::ahash::RandomState;
use polars_core::frame::row::AnyValueBuffer;
use polars_core::prelude::*;
use polars_core::utils::{_set_partition_size, accumulate_dataframes_vertical_unchecked};
use polars_core::POOL;
use polars_utils::option::UnwrapUncheckedRelease;
use polars_utils::slice::GetSaferUnchecked;
use polars_utils::{debug_unwrap, get_hash, hash_to_partition};
use rayon::prelude::*;

use super::aggregates::AggregateFn;
use crate::expressions::PhysicalPipedExpr;
use crate::operators::{DataChunk, PExecutionContext, Sink, SinkResult};

// We must strike a balance between cache coherence and resizing costs.
// Overallocation seems a lot more expensive than resizing so we start reasonable small.
pub(crate) const HASHMAP_INIT_SIZE: usize = 128;

// This is the hash and the Index offset in the linear buffer
type Key = (u64, IdxSize);

pub struct GenericGroupbySink {
    thread_no: usize,
    // idx is the offset in the array with keys
    // idx is the offset in the array with aggregators
    pre_agg_partitions: Vec<PlHashMap<Key, IdxSize>>,
    // the aggregations/keys are all tightly packed
    // the aggregation function of a group can be found
    // by:
    // first get the correct vec by the partition index
    //      * offset = (idx)
    //      * end = (offset + n_aggs)
    keys: Vec<Vec<AnyValue<'static>>>,
    aggregators: Vec<Vec<Box<dyn AggregateFn>>>,
    // the keys that will be aggregated on
    key_columns: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
    // the columns that will be aggregated
    aggregation_columns: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
    hb: RandomState,
    // Aggregation functions
    agg_fns: Vec<Box<dyn AggregateFn>>,
    output_schema: SchemaRef,
    // amortize allocations
    aggregation_series: Vec<Series>,
    keys_series: Vec<Series>,
    hashes: Vec<u64>,
}

impl GenericGroupbySink {
    pub fn new(
        key_columns: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
        aggregation_columns: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
        agg_fns: Vec<Box<dyn AggregateFn>>,
        output_schema: SchemaRef,
    ) -> Self {
        let hb = RandomState::default();
        let partitions = _set_partition_size();

        let mut pre_agg = Vec::with_capacity(partitions);
        let mut keys = Vec::with_capacity(partitions);
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
            keys,
            aggregators,
            key_columns,
            aggregation_columns,
            hb,
            agg_fns,
            output_schema,
            aggregation_series: vec![],
            keys_series: vec![],
            hashes: vec![],
        }
    }

    #[inline]
    fn number_of_aggs(&self) -> usize {
        self.aggregation_columns.len()
    }
}

impl Sink for GenericGroupbySink {
    fn sink(&mut self, context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult> {
        let num_aggs = self.number_of_aggs();

        // todo! ammortize allocation
        for phys_e in self.aggregation_columns.iter() {
            let s = phys_e.evaluate(&chunk, context.execution_state.as_ref())?;
            let s = s.to_physical_repr();
            self.aggregation_series.push(s.rechunk());
        }

        let mut agg_iters = self
            .aggregation_series
            .iter()
            .map(|s| s.phys_iter())
            .collect::<Vec<_>>();

        for phys_e in self.key_columns.iter() {
            let s = phys_e.evaluate(&chunk, context.execution_state.as_ref())?;
            let s = s.to_physical_repr();
            self.keys_series.push(s.rechunk());
        }

        let mut key_iters = self
            .keys_series
            .iter()
            .map(|s| s.phys_iter())
            .collect::<Vec<_>>();

        let mut iter_keys = self.keys_series.iter();
        let first_key = iter_keys.next().unwrap();
        first_key.vec_hash(self.hb.clone(), &mut self.hashes).unwrap();
        for other_key in iter_keys {
            other_key.vec_hash_combine(self.hb.clone(), &mut self.hashes).unwrap();
        }

        let mut current_keys_buf = Vec::with_capacity(self.keys.len());

        for &h in &self.hashes {
            // load the keys in the buffer
            current_keys_buf.clear();
            for key_iter in key_iters.iter_mut() {
                unsafe { current_keys_buf.push(key_iter.next().unwrap_unchecked_release()) }
            }

            let partition = hash_to_partition(h, self.pre_agg_partitions.len());
            let current_partition =
                unsafe { self.pre_agg_partitions.get_unchecked_release_mut(partition) };
            let current_aggregators =
                unsafe { self.aggregators.get_unchecked_release_mut(partition) };
            let current_key_values = unsafe { self.keys.get_unchecked_release_mut(partition) };

            let mut entry = current_partition.raw_entry_mut().from_hash(h, |key| {
                let idx = key.1 as usize;
                if self.keys_series.len() > 1 {
                    current_keys_buf.iter().enumerate().all(|(i, key)| unsafe {
                        current_key_values.get_unchecked_release(i + idx) == key
                    })
                } else {
                    unsafe {
                        current_key_values.get_unchecked_release(idx)
                            == current_keys_buf.get_unchecked_release(0)
                    }
                }
            });

            let agg_idx = match entry {
                RawEntryMut::Vacant(entry) => {
                    let value_offset = unsafe {
                        NumCast::from(current_aggregators.len()).unwrap_unchecked_release()
                    };
                    let keys_offset = unsafe {
                        (h, NumCast::from(current_key_values.len()).unwrap_unchecked_release())
                    };
                    entry.insert_with_hasher(h, keys_offset, value_offset, |_| h);

                    unsafe {
                        current_key_values.extend(
                            current_keys_buf
                                .iter()
                                .map(|av| av.clone().into_static().unwrap_unchecked_release()),
                        )
                    };
                    // initialize the aggregators
                    for agg_fn in &self.agg_fns {
                        current_aggregators.push(agg_fn.split())
                    }
                    value_offset
                }
                RawEntryMut::Occupied(entry) => *entry.get(),
            };
            for (i, agg_iter) in (0..num_aggs).zip(agg_iters.iter_mut()) {
                let i = agg_idx as usize + i;
                let agg_fn = unsafe { current_aggregators.get_unchecked_release_mut(i) };

                let value = unsafe { agg_iter.next().unwrap_unchecked_release() };
                agg_fn.pre_agg(chunk.chunk_index, value)
            }
        }
        drop(agg_iters);
        drop(key_iters);
        self.aggregation_series.clear();
        self.keys_series.clear();
        self.hashes.clear();
        Ok(SinkResult::CanHaveMoreInput)
    }

    fn combine(&mut self, other: Box<dyn Sink>) {
        // don't parallel this as this is already done in parallel.

        let other = other.as_any().downcast_ref::<Self>().unwrap();
        let n_partitions = self.pre_agg_partitions.len();

        self.pre_agg_partitions
            .iter_mut()
            .zip(other.pre_agg_partitions.iter())
            .zip(self.aggregators.iter_mut())
            .zip(other.aggregators.iter())
            .for_each(
                |(((map_self, map_other), aggregators_self), aggregators_other)| {
                    for (k_other, &agg_idx_other) in map_other.iter() {
                        unsafe {
                            // the hash value
                            let h = k_other.0;
                            // the partition where all keys and maps are located
                            let partition = hash_to_partition(h, n_partitions);
                            // get the key buffers
                            let keys_buffer_self = self.keys.get_unchecked_release_mut(partition);
                            let keys_buffer_other = other.keys.get_unchecked_release(partition);

                            // the offset in the keys of other
                            let idx_other = k_other.1 as usize;
                            // slice to the keys of other
                            let keys_other = keys_buffer_other
                                .get_unchecked_release(idx_other..idx_other + self.keys_series.len());

                            let entry = map_self.raw_entry_mut().from_hash(h, |k_self| {
                                // the offset in the keys of self
                                let idx_self = k_self.1 as usize;
                                // slice to the keys of self
                                let keys_self = keys_buffer_self
                                    .get_unchecked_release(idx_self..idx_self + self.keys_series.len());
                                // compare the keys
                                keys_self == keys_other
                            });

                            let agg_idx_self = match entry {
                                // the keys of other are not in this table, so we must update this table
                                RawEntryMut::Vacant(entry) => {
                                    // get the current offset in the values buffer
                                    let values_offset = NumCast::from(aggregators_self.len()).unwrap_unchecked_release();
                                    // get the key, comprised of the hash and the current offset in the keys buffer
                                    let key = unsafe {
                                        (h, NumCast::from(keys_buffer_self.len()).unwrap_unchecked_release())
                                    };

                                    // extend the keys buffer with the new keys from othger
                                    unsafe {
                                        keys_buffer_self.extend_from_slice(
                                            keys_other
                                        )
                                    };

                                    // inser the keys and values_offset
                                    entry.insert_with_hasher(h, key, values_offset, |_| h);
                                    // initialize the new aggregators
                                    for agg_fn in &self.agg_fns {
                                        aggregators_self.push(agg_fn.split())
                                    }
                                    values_offset
                                }
                                RawEntryMut::Occupied(entry) => *entry.get(),
                            };

                            // combine the aggregation functions
                            for i in 0..self.aggregation_columns.len() {
                                let agg_fn_other = aggregators_other
                                    .get_unchecked_release(agg_idx_other as usize + i);
                                let agg_fn_self = aggregators_self
                                    .get_unchecked_release_mut(agg_idx_self as usize + i);
                                agg_fn_self.combine(agg_fn_other.as_any())
                            }
                        }
                    }
                },
            );
    }

    fn split(&self, thread_no: usize) -> Box<dyn Sink> {
        let mut new = Self::new(
            self.key_columns.clone(),
            self.aggregation_columns.clone(),
            self.agg_fns.iter().map(|func| func.split()).collect(),
            self.output_schema.clone(),
        );
        new.hb = self.hb.clone();
        new.thread_no = thread_no;
        Box::new(new)
    }

    fn finalize(&mut self) -> PolarsResult<DataFrame> {
        // TODO! parallel
        let mut aggregators = std::mem::take(&mut self.aggregators);
        let n_keys = self.keys_series.len();

        POOL.install(|| {
            let dfs = self
                .pre_agg_partitions
                .par_iter()
                .zip(aggregators.par_iter_mut())
                .zip(self.keys.par_iter())
                .filter_map(|((agg_map, agg_fns), current_keys)| {
                    if agg_map.is_empty() {
                        return None;
                    }
                    let mut key_builders = self.keys_series.iter().map(|s|AnyValueBuffer::new(s.dtype(), agg_map.len())).collect::<Vec<_>>();
                    let dtypes = agg_fns
                        .iter()
                        .take(self.number_of_aggs())
                        .map(|func| func.dtype())
                        .collect::<Vec<_>>();

                    let mut buffers = dtypes
                        .iter()
                        .map(|dtype| AnyValueBuffer::new(dtype, agg_map.len()))
                        .collect::<Vec<_>>();

                    agg_map.into_iter().for_each(|(k, &offset)| {
                        let keys_offset = k.1 as usize;
                        let keys = unsafe { current_keys.get_unchecked_release(keys_offset..keys_offset + n_keys) };

                        for (key, key_builder) in keys.iter().zip(key_builders.iter_mut()) {
                            key_builder.add(key.as_borrowed());
                        }

                        for (i, buffer) in (offset as usize
                            ..offset as usize + self.aggregation_columns.len())
                            .zip(buffers.iter_mut())
                        {
                            unsafe {
                                let agg_fn = agg_fns.get_unchecked_release_mut(i);
                                let av = agg_fn.finalize();
                                buffer.add(av);
                            }
                        }
                    });

                    let mut cols = Vec::with_capacity(n_keys + self.number_of_aggs());
                    for key_builder in key_builders {
                        cols.push(key_builder.into_series());
                    }
                    cols.extend(buffers.into_iter().map(|buf| buf.into_series()));
                    for (s, (name, dtype)) in cols.iter_mut().zip(self.output_schema.iter()) {
                        if s.name() != name {
                            s.rename(name);
                        }
                        if s.dtype() != dtype {
                            *s = s.cast(dtype).unwrap()
                        }
                    }
                    Some(DataFrame::new_no_checks(cols))
                })
                .collect::<Vec<_>>();
            Ok(accumulate_dataframes_vertical_unchecked(dfs))
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
