use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

use hashbrown::hash_map::RawEntryMut;
use num_traits::NumCast;
use polars_arrow::trusted_len::PushUnchecked;
use polars_core::export::ahash::RandomState;
use polars_core::frame::row::AnyValueBuffer;
use polars_core::prelude::*;
use polars_core::series::SeriesPhysIter;
use polars_core::utils::_set_partition_size;
use polars_core::{IdBuildHasher, POOL};
use polars_utils::hash_to_partition;
use polars_utils::slice::GetSaferUnchecked;
use polars_utils::unwrap::UnwrapUncheckedRelease;
use rayon::prelude::*;

use super::aggregates::AggregateFn;
use crate::executors::sinks::groupby::aggregates::AggregateFunction;
use crate::executors::sinks::groupby::ooc_state::OocState;
use crate::executors::sinks::groupby::physical_agg_to_logical;
use crate::executors::sinks::groupby::utils::{compute_slices, finalize_groupby};
use crate::executors::sinks::io::IOThread;
use crate::executors::sinks::utils::{hash_series, load_vec};
use crate::executors::sinks::HASHMAP_INIT_SIZE;
use crate::expressions::PhysicalPipedExpr;
use crate::operators::{DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult};
use crate::pipeline::FORCE_OOC_GROUPBY;

// This is the hash and the Index offset in the linear buffer
#[derive(Copy, Clone)]
pub(super) struct Key {
    pub(super) hash: u64,
    pub(super) idx: IdxSize,
}

impl Key {
    #[inline]
    pub(super) fn new(hash: u64, idx: IdxSize) -> Self {
        Self { hash, idx }
    }
}

impl Hash for Key {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash)
    }
}

// we store a hashmap per partition (partitioned by hash)
// the hashmap contains indexes as keys and as values
// those indexes point into the keys buffer and the values buffer
// the keys buffer are buffers of AnyValue per partition
// and the values are buffer of Aggregation functions per partition
pub struct GenericGroupbySink {
    thread_no: usize,
    // idx is the offset in the array with keys
    // idx is the offset in the array with aggregators
    pre_agg_partitions: Vec<PlIdHashMap<Key, IdxSize>>,
    // the aggregations/keys are all tightly packed
    // the aggregation function of a group can be found
    // by:
    // first get the correct vec by the partition index
    //      * offset = (idx)
    //      * end = (offset + n_aggs)
    keys: Vec<Vec<AnyValue<'static>>>,
    aggregators: Vec<Vec<AggregateFunction>>,
    // the keys that will be aggregated on
    key_columns: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
    // the columns that will be aggregated
    aggregation_columns: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
    hb: RandomState,
    // Initializing Aggregation functions. If we aggregate by 2 columns
    // this vec will have two functions. We will use these functions
    // to populate the buffer where the hashmap points to
    agg_fns: Vec<AggregateFunction>,
    input_schema: SchemaRef,
    output_schema: SchemaRef,
    // amortize allocations
    aggregation_series: Vec<Series>,
    keys_series: Vec<Series>,
    hashes: Vec<u64>,
    slice: Option<(i64, usize)>,
    ooc_state: OocState,
}

impl GenericGroupbySink {
    pub(crate) fn new(
        key_columns: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
        aggregation_columns: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
        agg_fns: Vec<AggregateFunction>,
        input_schema: SchemaRef,
        output_schema: SchemaRef,
        slice: Option<(i64, usize)>,
    ) -> Self {
        let ooc = std::env::var(FORCE_OOC_GROUPBY).is_ok();
        Self::new_inner(
            key_columns,
            aggregation_columns,
            agg_fns,
            input_schema,
            output_schema,
            slice,
            None,
            ooc,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_inner(
        key_columns: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
        aggregation_columns: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
        agg_fns: Vec<AggregateFunction>,
        input_schema: SchemaRef,
        output_schema: SchemaRef,
        slice: Option<(i64, usize)>,
        io_thread: Option<Arc<Mutex<Option<IOThread>>>>,
        ooc: bool,
    ) -> Self {
        let hb = RandomState::default();
        let partitions = _set_partition_size();

        let pre_agg = load_vec(partitions, || PlIdHashMap::with_capacity(HASHMAP_INIT_SIZE));
        let keys = load_vec(partitions, || {
            Vec::with_capacity(HASHMAP_INIT_SIZE * key_columns.len())
        });
        let aggregators = load_vec(partitions, || {
            Vec::with_capacity(HASHMAP_INIT_SIZE * aggregation_columns.len())
        });

        let mut out = Self {
            thread_no: 0,
            pre_agg_partitions: pre_agg,
            keys,
            aggregators,
            key_columns,
            aggregation_columns,
            hb,
            agg_fns,
            input_schema,
            output_schema,
            aggregation_series: vec![],
            keys_series: vec![],
            hashes: vec![],
            slice,
            ooc_state: OocState::new(io_thread, ooc),
        };
        if ooc {
            out.ooc_state.init_ooc(out.input_schema.clone()).unwrap();
        }
        out
    }

    #[inline]
    fn number_of_aggs(&self) -> usize {
        self.aggregation_columns.len()
    }
    #[inline]
    fn number_of_keys(&self) -> usize {
        self.key_columns.len()
    }

    fn pre_finalize(&mut self) -> PolarsResult<Vec<DataFrame>> {
        let mut aggregators = std::mem::take(&mut self.aggregators);
        let n_keys = self.number_of_keys();
        let slices = compute_slices(&self.pre_agg_partitions, self.slice);

        POOL.install(|| {
            let dfs =
                self.pre_agg_partitions
                    .par_iter()
                    .zip(aggregators.par_iter_mut())
                    .zip(self.keys.par_iter())
                    .zip(slices.par_iter())
                    .filter_map(|(((agg_map, agg_fns), current_keys), slice)| {
                        let (offset, slice_len) = (*slice)?;
                        if agg_map.is_empty() {
                            return None;
                        }
                        let mut key_builders = self
                            .output_schema
                            .iter_dtypes()
                            .take(n_keys)
                            .map(|dtype| AnyValueBuffer::new(&dtype.to_physical(), agg_map.len()))
                            .collect::<Vec<_>>();
                        let dtypes = agg_fns
                            .iter()
                            .take(self.number_of_aggs())
                            .map(|func| func.dtype())
                            .collect::<Vec<_>>();

                        let mut buffers = dtypes
                            .iter()
                            .map(|dtype| AnyValueBuffer::new(dtype, slice_len))
                            .collect::<Vec<_>>();

                        agg_map.into_iter().skip(offset).take(slice_len).for_each(
                            |(k, &offset)| {
                                let keys_offset = k.idx as usize;
                                let keys = unsafe {
                                    current_keys
                                        .get_unchecked_release(keys_offset..keys_offset + n_keys)
                                };

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
                            },
                        );

                        let mut cols = Vec::with_capacity(n_keys + self.number_of_aggs());
                        for key_builder in key_builders {
                            cols.push(key_builder.into_series());
                        }
                        cols.extend(buffers.into_iter().map(|buf| buf.into_series()));
                        physical_agg_to_logical(&mut cols, &self.output_schema);
                        Some(DataFrame::new_no_checks(cols))
                    })
                    .collect::<Vec<_>>();

            Ok(dfs)
        })
    }

    fn evaluate_keys_aggs_and_hashes(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<()> {
        // todo! amortize allocation
        for phys_e in self.aggregation_columns.iter() {
            let s = phys_e.evaluate(chunk, context.execution_state.as_any())?;
            let s = s.to_physical_repr();
            self.aggregation_series.push(s.rechunk());
        }
        for phys_e in self.key_columns.iter() {
            let s = phys_e.evaluate(chunk, context.execution_state.as_any())?;
            let s = s.to_physical_repr();
            self.keys_series.push(s.rechunk());
        }

        // write the hashes to self.hashes buffer
        hash_series(&self.keys_series, &mut self.hashes, &self.hb);
        Ok(())
    }

    #[inline]
    fn get_partitions(
        &mut self,
        h: u64,
    ) -> (
        &mut PlIdHashMap<Key, IdxSize>,
        &mut Vec<AggregateFunction>,
        &mut Vec<AnyValue<'static>>,
    ) {
        let partition = hash_to_partition(h, self.pre_agg_partitions.len());
        let current_partition =
            unsafe { self.pre_agg_partitions.get_unchecked_release_mut(partition) };
        let current_aggregators = unsafe { self.aggregators.get_unchecked_release_mut(partition) };
        let current_key_values = unsafe { self.keys.get_unchecked_release_mut(partition) };

        (current_partition, current_aggregators, current_key_values)
    }

    fn sink_ooc(
        &mut self,
        context: &PExecutionContext,
        chunk: DataChunk,
    ) -> PolarsResult<SinkResult> {
        let num_aggs = self.number_of_aggs();
        let num_keys = self.number_of_keys();
        self.evaluate_keys_aggs_and_hashes(context, &chunk)?;

        // take containers to please bchk
        // we put them back once done
        let keys_series = std::mem::take(&mut self.keys_series);
        let aggregation_series = std::mem::take(&mut self.aggregation_series);
        let mut hashes = std::mem::take(&mut self.hashes);
        let agg_fns = std::mem::take(&mut self.agg_fns);

        let (mut key_iters, mut agg_iters) = get_iters(&keys_series, &aggregation_series);

        // a small buffer that holds the current key values
        // if we groupby 2 keys, this holds 2 anyvalues.
        let mut current_tuple = Vec::with_capacity(num_keys);

        // set all bits to false
        self.ooc_state.reset_ooc_filter_rows(chunk.data.height());

        for (iteration_idx, &h) in hashes.iter().enumerate() {
            // load the keys in the buffer
            current_tuple.clear();
            for key_iter in key_iters.iter_mut() {
                unsafe { current_tuple.push_unchecked(key_iter.next().unwrap_unchecked_release()) }
            }

            let (current_partition, current_aggregators, current_key_values) =
                self.get_partitions(h);
            let entry = get_entry(
                h,
                &mut current_tuple,
                current_partition,
                current_key_values,
                num_keys,
            );

            match entry {
                // if the slot does not exist, we do not add it but instead
                // we sink these rows to disk
                RawEntryMut::Vacant(_) => {
                    // set this row to true: e.g. processed ooc
                    // safety: we correctly set the length with `reset_ooc_filter_rows`
                    unsafe {
                        self.ooc_state.set_row_as_ooc(iteration_idx);
                    }
                }
                RawEntryMut::Occupied(entry) => {
                    let agg_idx = *entry.get();

                    apply_aggregation(
                        agg_idx,
                        num_aggs,
                        chunk.chunk_index,
                        &mut agg_iters,
                        current_aggregators,
                    );
                }
            };
        }
        self.ooc_state.dump(chunk.data, &mut hashes);
        drop(agg_iters);
        drop(key_iters);
        self.aggregation_series = aggregation_series;
        self.keys_series = keys_series;
        self.hashes = hashes;
        self.agg_fns = agg_fns;
        self.aggregation_series.clear();
        self.keys_series.clear();
        self.hashes.clear();
        Ok(SinkResult::CanHaveMoreInput)
    }
}

impl Sink for GenericGroupbySink {
    fn sink(&mut self, context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult> {
        if self.ooc_state.ooc {
            return self.sink_ooc(context, chunk);
        }
        let num_aggs = self.number_of_aggs();
        let num_keys = self.number_of_keys();
        self.evaluate_keys_aggs_and_hashes(context, &chunk)?;

        // take containers to please bchk
        // we put them back once done
        let keys_series = std::mem::take(&mut self.keys_series);
        let aggregation_series = std::mem::take(&mut self.aggregation_series);
        let hashes = std::mem::take(&mut self.hashes);
        let agg_fns = std::mem::take(&mut self.agg_fns);

        let (mut key_iters, mut agg_iters) = get_iters(&keys_series, &aggregation_series);

        // a small buffer that holds the current key values
        // if we groupby 2 keys, this holds 2 anyvalues.
        let mut current_tuple = Vec::with_capacity(self.key_columns.len());

        for &h in &hashes {
            // load the keys in the buffer
            current_tuple.clear();
            for key_iter in key_iters.iter_mut() {
                unsafe { current_tuple.push_unchecked(key_iter.next().unwrap_unchecked_release()) }
            }

            let (current_partition, current_aggregators, current_key_values) =
                self.get_partitions(h);
            let entry = get_entry(
                h,
                &mut current_tuple,
                current_partition,
                current_key_values,
                num_keys,
            );

            let agg_idx = match entry {
                RawEntryMut::Vacant(entry) => {
                    let value_offset = unsafe {
                        NumCast::from(current_aggregators.len()).unwrap_unchecked_release()
                    };
                    let keys_offset = unsafe {
                        Key::new(
                            h,
                            NumCast::from(current_key_values.len()).unwrap_unchecked_release(),
                        )
                    };
                    entry.insert(keys_offset, value_offset);

                    unsafe {
                        current_key_values.extend(
                            current_tuple
                                .iter()
                                .map(|av| av.clone().into_static().unwrap_unchecked_release()),
                        )
                    };
                    // initialize the aggregators
                    for agg_fn in &agg_fns {
                        current_aggregators.push(agg_fn.split2())
                    }
                    value_offset
                }
                RawEntryMut::Occupied(entry) => *entry.get(),
            };
            apply_aggregation(
                agg_idx,
                num_aggs,
                chunk.chunk_index,
                &mut agg_iters,
                current_aggregators,
            );
        }
        drop(agg_iters);
        drop(key_iters);
        self.aggregation_series = aggregation_series;
        self.keys_series = keys_series;
        self.hashes = hashes;
        self.agg_fns = agg_fns;
        self.aggregation_series.clear();
        self.keys_series.clear();
        self.hashes.clear();
        self.ooc_state.check_memory_usage(&self.input_schema)?;
        Ok(SinkResult::CanHaveMoreInput)
    }

    fn combine(&mut self, other: &mut dyn Sink) {
        // don't parallelize this as this is already done in parallel.

        let other = other.as_any().downcast_ref::<Self>().unwrap();
        let n_partitions = self.pre_agg_partitions.len();
        debug_assert_eq!(n_partitions, other.pre_agg_partitions.len());
        let n_keys = self.number_of_keys();

        self.pre_agg_partitions
            .iter_mut()
            .zip(other.pre_agg_partitions.iter())
            .zip(self.aggregators.iter_mut())
            .zip(other.aggregators.iter())
            .for_each(
                |(((map_self, map_other), aggregators_self), aggregators_other)| {
                    for (k_other, &agg_idx_other) in map_other.iter() {
                        // the hash value
                        let h = k_other.hash;
                        // the partition where all keys and maps are located
                        let partition = hash_to_partition(h, n_partitions);
                        // get the key buffers
                        let keys_buffer_self =
                            unsafe { self.keys.get_unchecked_release_mut(partition) };
                        let keys_buffer_other =
                            unsafe { other.keys.get_unchecked_release(partition) };

                        // the offset in the keys of other
                        let idx_other = k_other.idx as usize;
                        // slice to the keys of other
                        let keys_other = unsafe {
                            keys_buffer_other.get_unchecked_release(idx_other..idx_other + n_keys)
                        };

                        let entry = map_self.raw_entry_mut().from_hash(h, |k_self| {
                            // the offset in the keys of self
                            let idx_self = k_self.idx as usize;
                            // slice to the keys of self
                            // safety:
                            // in bounds
                            let keys_self = unsafe {
                                keys_buffer_self.get_unchecked_release(idx_self..idx_self + n_keys)
                            };
                            // compare the keys
                            keys_self == keys_other
                        });

                        let agg_idx_self = match entry {
                            // the keys of other are not in this table, so we must update this table
                            RawEntryMut::Vacant(entry) => {
                                // get the current offset in the values buffer
                                let values_offset = unsafe {
                                    NumCast::from(aggregators_self.len()).unwrap_unchecked_release()
                                };
                                // get the key, comprised of the hash and the current offset in the keys buffer
                                let key = unsafe {
                                    Key::new(
                                        h,
                                        NumCast::from(keys_buffer_self.len())
                                            .unwrap_unchecked_release(),
                                    )
                                };

                                // extend the keys buffer with the new keys from other
                                keys_buffer_self.extend_from_slice(keys_other);

                                // insert the keys and values_offset
                                entry.insert(key, values_offset);
                                // initialize the new aggregators
                                for agg_fn in &self.agg_fns {
                                    aggregators_self.push(agg_fn.split2())
                                }
                                values_offset
                            }
                            RawEntryMut::Occupied(entry) => *entry.get(),
                        };

                        // combine the aggregation functions
                        for i in 0..self.aggregation_columns.len() {
                            unsafe {
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
        let mut new = Self::new_inner(
            self.key_columns.clone(),
            self.aggregation_columns.clone(),
            self.agg_fns.iter().map(|func| func.split2()).collect(),
            self.input_schema.clone(),
            self.output_schema.clone(),
            self.slice,
            Some(self.ooc_state.io_thread.clone()),
            self.ooc_state.ooc,
        );
        new.hb = self.hb.clone();
        new.thread_no = thread_no;
        Box::new(new)
    }

    fn finalize(&mut self, _context: &PExecutionContext) -> PolarsResult<FinalizedSink> {
        let dfs = self.pre_finalize()?;
        let payload = if self.ooc_state.ooc {
            let mut iot = self.ooc_state.io_thread.lock().unwrap();
            // make sure that we reset the shared states
            // the OOC groupby will call split as well and it should
            // not send continue spilling to disk
            let iot = iot.take().unwrap();
            self.ooc_state.ooc = false;

            Some((iot, self.split(0)))
        } else {
            None
        };
        finalize_groupby(dfs, &self.output_schema, self.slice, payload)
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
    fn fmt(&self) -> &str {
        "generic_groupby"
    }
}

#[inline]
fn get_entry<'a>(
    h: u64,
    current_tuple: &mut Vec<AnyValue>,
    current_partition: &'a mut PlIdHashMap<Key, IdxSize>,
    current_key_values: &Vec<AnyValue>,
    n_keys: usize,
) -> RawEntryMut<'a, Key, IdxSize, IdBuildHasher> {
    current_partition.raw_entry_mut().from_hash(h, |key| {
        key.hash == h && {
            let idx = key.idx as usize;
            if n_keys > 1 {
                current_tuple.iter().enumerate().all(|(i, key)| unsafe {
                    current_key_values.get_unchecked_release(i + idx) == key
                })
            } else {
                unsafe {
                    current_key_values.get_unchecked_release(idx)
                        == current_tuple.get_unchecked_release(0)
                }
            }
        }
    })
}
fn get_iters<'a>(
    key_series: &'a [Series],
    aggregation_series: &'a [Series],
) -> (Vec<SeriesPhysIter<'a>>, Vec<SeriesPhysIter<'a>>) {
    (
        key_series.iter().map(|s| s.phys_iter()).collect::<Vec<_>>(),
        aggregation_series
            .iter()
            .map(|s| s.phys_iter())
            .collect::<Vec<_>>(),
    )
}

#[inline]
fn apply_aggregation(
    agg_idx: IdxSize,
    num_aggs: usize,
    chunk_index: IdxSize,
    agg_iters: &mut [SeriesPhysIter],
    current_aggregators: &mut [AggregateFunction],
) {
    for (i, agg_iter) in (0..num_aggs).zip(agg_iters.iter_mut()) {
        let i = agg_idx as usize + i;
        let agg_fn = unsafe { current_aggregators.get_unchecked_release_mut(i) };

        agg_fn.pre_agg(chunk_index, agg_iter.as_mut())
    }
}
