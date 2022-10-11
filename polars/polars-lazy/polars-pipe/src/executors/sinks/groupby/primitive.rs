use std::any::Any;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

use hashbrown::hash_map::RawEntryMut;
use num::NumCast;
use polars_core::export::ahash::RandomState;
use polars_core::frame::row::AnyValueBuffer;
use polars_core::prelude::*;
use polars_core::utils::{_set_partition_size, accumulate_dataframes_vertical_unchecked};
use polars_core::POOL;
use polars_utils::hash_to_partition;
use polars_utils::slice::GetSaferUnchecked;
use polars_utils::unwrap::UnwrapUncheckedRelease;
use rayon::prelude::*;

use super::aggregates::AggregateFn;
use crate::expressions::PhysicalPipedExpr;
use crate::operators::{DataChunk, PExecutionContext, Sink, SinkResult};

// We must strike a balance between cache coherence and resizing costs.
// Overallocation seems a lot more expensive than resizing so we start reasonable small.
pub(crate) const HASHMAP_INIT_SIZE: usize = 128;
// hash + value
#[derive(Eq, Copy, Clone)]
struct Key<T: Copy> {
    hash: u64,
    value: T,
}

impl<T: Copy> Hash for Key<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash)
    }
}

impl<T: Copy + PartialEq> PartialEq for Key<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

pub struct PrimitiveGroupbySink<K: PolarsNumericType> {
    thread_no: usize,
    // idx is the offset in the array with aggregators
    pre_agg_partitions: Vec<PlHashMap<Key<Option<K::Native>>, IdxSize>>,
    // the aggregations are all tightly packed
    // the aggregation function of a group can be found
    // by:
    // first get the correct vec by the partition index
    //      * offset = (idx)
    //      * end = (offset + n_aggs)
    aggregators: Vec<Vec<Box<dyn AggregateFn>>>,
    key: Arc<dyn PhysicalPipedExpr>,
    // the columns that will be aggregated
    aggregation_columns: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
    hb: RandomState,
    // Aggregation functions
    agg_fns: Vec<Box<dyn AggregateFn>>,
    output_schema: SchemaRef,
    // amortize allocations
    aggregation_series: Vec<Series>,
    hashes: Vec<u64>,
}

impl<K: PolarsNumericType> PrimitiveGroupbySink<K> {
    pub fn new(
        key: Arc<dyn PhysicalPipedExpr>,
        aggregation_columns: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
        agg_fns: Vec<Box<dyn AggregateFn>>,
        output_schema: SchemaRef,
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
            output_schema,
            aggregation_series: vec![],
            hashes: vec![],
        }
    }

    #[inline]
    fn number_of_aggs(&self) -> usize {
        self.aggregation_columns.len()
    }
}

impl<K: PolarsNumericType> Sink for PrimitiveGroupbySink<K>
where
    K::Native: Hash + Eq + Debug,
    ChunkedArray<K>: IntoSeries,
{
    fn sink(&mut self, context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult> {
        let num_aggs = self.number_of_aggs();

        let s = self
            .key
            .evaluate(&chunk, context.execution_state.as_ref())?;
        let s = s.to_physical_repr();
        let s = s.rechunk();

        s.vec_hash(self.hb.clone(), &mut self.hashes).unwrap();

        // cow -> &series -> &dyn series_trait -> &chunkedarray
        let ca: &ChunkedArray<K> = s.as_ref().as_ref();

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

        let arr = ca.downcast_iter().next().unwrap();
        for (opt_v, &h) in arr.iter().zip(self.hashes.iter()) {
            let opt_v = opt_v.copied();
            let part = hash_to_partition(h, self.pre_agg_partitions.len());
            let current_partition =
                unsafe { self.pre_agg_partitions.get_unchecked_release_mut(part) };
            let current_aggregators = unsafe { self.aggregators.get_unchecked_release_mut(part) };

            let key = Key {
                hash: h,
                value: opt_v,
            };
            let entry = current_partition.raw_entry_mut().from_key(&key);
            let agg_idx = match entry {
                RawEntryMut::Vacant(entry) => {
                    let offset = unsafe {
                        NumCast::from(current_aggregators.len()).unwrap_unchecked_release()
                    };
                    entry.insert(key, offset);
                    // initialize the aggregators
                    for agg_fn in &self.agg_fns {
                        current_aggregators.push(agg_fn.split())
                    }
                    offset
                }
                RawEntryMut::Occupied(entry) => *entry.get(),
            };
            for (i, agg_iter) in (0 as IdxSize..num_aggs as IdxSize).zip(agg_iters.iter_mut()) {
                let i = (agg_idx + i) as usize;
                let agg_fn = unsafe { current_aggregators.get_unchecked_release_mut(i) };

                let value = unsafe { agg_iter.next().unwrap_unchecked_release() };
                agg_fn.pre_agg(chunk.chunk_index, value)
            }
        }
        drop(agg_iters);
        self.aggregation_series.clear();
        Ok(SinkResult::CanHaveMoreInput)
    }

    fn combine(&mut self, other: Box<dyn Sink>) {
        // don't parallel this as this is already done in parallel.
        let other = other.as_any().downcast_ref::<Self>().unwrap();

        self.pre_agg_partitions
            .iter_mut()
            .zip(other.pre_agg_partitions.iter())
            .zip(self.aggregators.iter_mut())
            .zip(other.aggregators.iter())
            .for_each(
                |(((map_self, map_other), aggregators_self), aggregators_other)| {
                    for (key, &agg_idx_other) in map_other.iter() {
                        unsafe {
                            let entry = map_self.raw_entry_mut().from_key(key);

                            let agg_idx_self = match entry {
                                RawEntryMut::Vacant(entry) => {
                                    let offset = NumCast::from(aggregators_self.len()).unwrap();
                                    entry.insert(*key, offset);
                                    // initialize the aggregators
                                    for agg_fn in &self.agg_fns {
                                        aggregators_self.push(agg_fn.split())
                                    }
                                    offset
                                }
                                RawEntryMut::Occupied(entry) => *entry.get(),
                            };
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
            self.key.clone(),
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

        POOL.install(|| {
            let dfs = self
                .pre_agg_partitions
                .par_iter()
                .zip(aggregators.par_iter_mut())
                .filter_map(|(agg_map, agg_fns)| {
                    if agg_map.is_empty() {
                        return None;
                    }
                    let mut key_builder = PrimitiveChunkedBuilder::<K>::new(
                        self.output_schema.get_index(0).unwrap().0,
                        agg_map.len(),
                    );
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
                        key_builder.append_option(k.value);

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

                    let mut cols = Vec::with_capacity(1 + self.number_of_aggs());
                    cols.push(key_builder.finish().into_series());
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
            let mut df = accumulate_dataframes_vertical_unchecked(dfs);
            DataFrame::new(std::mem::take(df.get_columns_mut()))
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
