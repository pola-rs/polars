use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

use hashbrown::hash_map::RawEntryMut;
use num_traits::NumCast;
use polars_core::frame::row::AnyValueBuffer;
use polars_core::prelude::*;
use polars_core::utils::_set_partition_size;
use polars_core::{IdBuildHasher, POOL};
use polars_utils::hashing::hash_to_partition;
use polars_utils::slice::GetSaferUnchecked;
use polars_utils::unwrap::UnwrapUncheckedRelease;
use rayon::prelude::*;

use super::aggregates::AggregateFn;
use crate::executors::sinks::group_by::aggregates::AggregateFunction;
use crate::executors::sinks::group_by::ooc_state::OocState;
use crate::executors::sinks::group_by::physical_agg_to_logical;
use crate::executors::sinks::group_by::primitive::apply_aggregation;
use crate::executors::sinks::group_by::utils::{compute_slices, finalize_group_by, prepare_key};
use crate::executors::sinks::io::IOThread;
use crate::executors::sinks::utils::load_vec;
use crate::executors::sinks::HASHMAP_INIT_SIZE;
use crate::expressions::PhysicalPipedExpr;
use crate::operators::{DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult};

// This is the hash and the Index offset in the linear buffer
#[derive(Copy, Clone)]
struct Key {
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
pub struct StringGroupbySink {
    thread_no: usize,
    // idx is the offset in the array with keys
    // idx is the offset in the array with aggregators
    pre_agg_partitions: Vec<PlIdHashMap<Key, IdxSize>>,
    // the aggregations/keys are all tightly packed
    // the aggregation function of a group can be found
    // by:
    //      * offset = (idx)
    //      * end = (offset + 1)
    keys: Vec<Option<smartstring::alias::String>>,
    aggregators: Vec<AggregateFunction>,
    // the key that will be aggregated on
    key_column: Arc<dyn PhysicalPipedExpr>,
    // the columns that will be aggregated
    aggregation_columns: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
    hb: PlRandomState,
    // Initializing Aggregation functions. If we aggregate by 2 columns
    // this vec will have two functions. We will use these functions
    // to populate the buffer where the hashmap points to
    agg_fns: Vec<AggregateFunction>,
    input_schema: SchemaRef,
    output_schema: SchemaRef,
    // amortize allocations
    aggregation_series: Vec<Series>,
    hashes: Vec<u64>,
    slice: Option<(i64, usize)>,

    ooc_state: OocState,
}

impl StringGroupbySink {
    pub(crate) fn new(
        key_column: Arc<dyn PhysicalPipedExpr>,
        aggregation_columns: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
        agg_fns: Vec<AggregateFunction>,
        input_schema: SchemaRef,
        output_schema: SchemaRef,
        slice: Option<(i64, usize)>,
    ) -> Self {
        Self::new_inner(
            key_column,
            aggregation_columns,
            agg_fns,
            input_schema,
            output_schema,
            slice,
            None,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_inner(
        key_column: Arc<dyn PhysicalPipedExpr>,
        aggregation_columns: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
        agg_fns: Vec<AggregateFunction>,
        input_schema: SchemaRef,
        output_schema: SchemaRef,
        slice: Option<(i64, usize)>,
        io_thread: Option<Arc<Mutex<Option<IOThread>>>>,
        ooc: bool,
    ) -> Self {
        let hb = Default::default();
        let partitions = _set_partition_size();

        let pre_agg = load_vec(partitions, || PlIdHashMap::with_capacity(HASHMAP_INIT_SIZE));
        let keys = Vec::with_capacity(HASHMAP_INIT_SIZE * partitions);
        let aggregators =
            Vec::with_capacity(HASHMAP_INIT_SIZE * aggregation_columns.len() * partitions);

        let mut out = Self {
            thread_no: 0,
            pre_agg_partitions: pre_agg,
            keys,
            aggregators,
            key_column,
            aggregation_columns,
            hb,
            agg_fns,
            input_schema,
            output_schema,
            aggregation_series: vec![],
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

    fn pre_finalize(&mut self) -> PolarsResult<Vec<DataFrame>> {
        // we create a pointer to the aggregation functions buffer
        // we will deref *mut on every partition thread
        // this will be safe, as the partitions guarantee that access don't alias.
        let aggregators = self.aggregators.as_ptr() as usize;
        let aggregators_len = self.aggregators.len();

        let slices = compute_slices(&self.pre_agg_partitions, self.slice);

        POOL.install(|| {
            let dfs =
                self.pre_agg_partitions
                    .par_iter()
                    .zip(slices.par_iter())
                    .filter_map(|(agg_map, slice)| {
                        let ptr = aggregators as *mut AggregateFunction;
                        // SAFETY:
                        // we will not alias.
                        let aggregators =
                            unsafe { std::slice::from_raw_parts_mut(ptr, aggregators_len) };

                        let (offset, slice_len) = (*slice)?;
                        if agg_map.is_empty() {
                            return None;
                        }
                        let dtypes = aggregators
                            .iter()
                            .take(self.number_of_aggs())
                            .map(|func| func.dtype())
                            .collect::<Vec<_>>();

                        let mut buffers = dtypes
                            .iter()
                            .map(|dtype| AnyValueBuffer::new(dtype, slice_len))
                            .collect::<Vec<_>>();

                        let cap = std::cmp::min(slice_len, agg_map.len());
                        let mut key_builder = StringChunkedBuilder::new("", cap);
                        agg_map.into_iter().skip(offset).take(slice_len).for_each(
                            |(k, &offset)| {
                                let key_offset = k.idx as usize;
                                let key = unsafe {
                                    self.keys.get_unchecked_release(key_offset).as_deref()
                                };
                                key_builder.append_option(key);

                                for (i, buffer) in (offset as usize
                                    ..offset as usize + self.aggregation_columns.len())
                                    .zip(buffers.iter_mut())
                                {
                                    unsafe {
                                        let agg_fn = aggregators.get_unchecked_release_mut(i);
                                        let av = agg_fn.finalize();
                                        buffer.add(av);
                                    }
                                }
                            },
                        );

                        let mut cols = Vec::with_capacity(1 + self.number_of_aggs());
                        cols.push(key_builder.finish().into_series());
                        cols.extend(buffers.into_iter().map(|buf| buf.into_series()));
                        physical_agg_to_logical(&mut cols, &self.output_schema);
                        Some(unsafe { DataFrame::new_no_checks(cols) })
                    })
                    .collect::<Vec<_>>();

            Ok(dfs)
        })
    }
    fn prepare_key_and_aggregation_series(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<Series> {
        let s = self.key_column.evaluate(chunk, &context.execution_state)?;
        let s = s.to_physical_repr();
        let s = prepare_key(&s, chunk);

        // todo! ammortize allocation
        for phys_e in self.aggregation_columns.iter() {
            let s = phys_e.evaluate(chunk, &context.execution_state)?;
            let s = s.to_physical_repr();
            self.aggregation_series.push(s.rechunk());
        }
        s.vec_hash(self.hb.clone(), &mut self.hashes).unwrap();
        Ok(s)
    }
    #[inline]
    fn get_partitions(&mut self, h: u64) -> &mut PlIdHashMap<Key, IdxSize> {
        let partition = hash_to_partition(h, self.pre_agg_partitions.len());
        let current_partition =
            unsafe { self.pre_agg_partitions.get_unchecked_release_mut(partition) };

        current_partition
    }

    fn sink_ooc(
        &mut self,
        context: &PExecutionContext,
        chunk: DataChunk,
    ) -> PolarsResult<SinkResult> {
        let s = self.prepare_key_and_aggregation_series(context, &chunk)?;

        // take containers to please bchk
        // we put them back once done
        let mut hashes = std::mem::take(&mut self.hashes);
        let keys = std::mem::take(&mut self.keys);
        let agg_fns = std::mem::take(&mut self.agg_fns);
        let mut aggregators = std::mem::take(&mut self.aggregators);

        // write the hashes to self.hashes buffer
        // s.vec_hash(self.hb.clone(), &mut self.hashes).unwrap();
        // now we have written hashes, we take the pointer to this buffer
        // we will write the aggregation_function indexes in the same buffer
        // this is unsafe and we must check that we only write the hashes that
        // already read/taken. So we write on the slots we just read
        let agg_idx_ptr = hashes.as_ptr() as *mut u64 as *mut IdxSize;
        // array of the keys
        let keys_arr = s.str().unwrap().downcast_iter().next().unwrap().clone();

        // set all bits to false
        self.ooc_state.reset_ooc_filter_rows(chunk.data.height());

        let mut processed = 0;
        for (iteration_idx, (key_val, &h)) in keys_arr.iter().zip(&hashes).enumerate() {
            let current_partition = self.get_partitions(h);
            let entry = get_entry(key_val, h, current_partition, &keys);

            match entry {
                RawEntryMut::Vacant(_) => {
                    // set this row to true: e.g. processed ooc
                    // SAFETY: we correctly set the length with `reset_ooc_filter_rows`
                    unsafe {
                        self.ooc_state.set_row_as_ooc(iteration_idx);
                    }
                },
                RawEntryMut::Occupied(entry) => {
                    let agg_idx = *entry.get();
                    // # Safety
                    // we write to the hashes buffer we iterate over at the moment.
                    // this is sound because we writes are trailing from iteration
                    unsafe { write_agg_idx(agg_idx_ptr, processed, agg_idx) };
                    processed += 1;
                },
            };
        }

        // note that this slice looks into the self.hashes buffer
        let agg_idxs = unsafe { std::slice::from_raw_parts(agg_idx_ptr, processed) };

        apply_aggregation(
            agg_idxs,
            &chunk,
            self.number_of_aggs(),
            &self.aggregation_series,
            &agg_fns,
            &mut aggregators,
        );
        self.ooc_state.dump(chunk.data, &mut hashes);

        self.aggregation_series.clear();
        self.hashes = hashes;
        self.keys = keys;
        self.agg_fns = agg_fns;
        self.aggregators = aggregators;
        self.hashes.clear();
        self.ooc_state.check_memory_usage(&self.input_schema)?;
        Ok(SinkResult::CanHaveMoreInput)
    }
}

impl Sink for StringGroupbySink {
    fn sink(&mut self, context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult> {
        if chunk.is_empty() {
            return Ok(SinkResult::CanHaveMoreInput);
        }
        if self.ooc_state.ooc {
            return self.sink_ooc(context, chunk);
        }
        let s = self.prepare_key_and_aggregation_series(context, &chunk)?;

        // take containers to please bchk
        // we put them back once done
        let hashes = std::mem::take(&mut self.hashes);
        let mut keys = std::mem::take(&mut self.keys);
        let agg_fns = std::mem::take(&mut self.agg_fns);
        let mut aggregators = std::mem::take(&mut self.aggregators);

        // write the hashes to self.hashes buffer
        // s.vec_hash(self.hb.clone(), &mut self.hashes).unwrap();
        // now we have written hashes, we take the pointer to this buffer
        // we will write the aggregation_function indexes in the same buffer
        // this is unsafe and we must check that we only write the hashes that
        // already read/taken. So we write on the slots we just read
        let agg_idx_ptr = hashes.as_ptr() as *mut u64 as *mut IdxSize;
        // array of the keys
        let keys_arr = s.str().unwrap().downcast_iter().next().unwrap().clone();

        for (iteration_idx, (key_val, &h)) in keys_arr.iter().zip(&hashes).enumerate() {
            let current_partition = self.get_partitions(h);
            let entry = get_entry(key_val, h, current_partition, &keys);

            let agg_idx = match entry {
                RawEntryMut::Vacant(entry) => {
                    let value_offset =
                        unsafe { NumCast::from(aggregators.len()).unwrap_unchecked_release() };
                    let keys_offset = unsafe {
                        Key::new(h, NumCast::from(keys.len()).unwrap_unchecked_release())
                    };
                    entry.insert(keys_offset, value_offset);

                    keys.push(key_val.map(|s| s.into()));

                    // initialize the aggregators
                    for agg_fn in &agg_fns {
                        aggregators.push(agg_fn.split())
                    }
                    value_offset
                },
                RawEntryMut::Occupied(entry) => *entry.get(),
            };
            // # Safety
            // we write to the hashes buffer we iterate over at the moment.
            // this is sound because we writes are trailing from iteration
            unsafe { write_agg_idx(agg_idx_ptr, iteration_idx, agg_idx) };
        }

        // note that this slice looks into the self.hashes buffer
        let agg_idxs = unsafe { std::slice::from_raw_parts(agg_idx_ptr, keys_arr.len()) };

        apply_aggregation(
            agg_idxs,
            &chunk,
            self.number_of_aggs(),
            &self.aggregation_series,
            &agg_fns,
            &mut aggregators,
        );
        self.aggregation_series.clear();
        self.hashes = hashes;
        self.keys = keys;
        self.agg_fns = agg_fns;
        self.aggregators = aggregators;
        self.hashes.clear();
        self.ooc_state.check_memory_usage(&self.input_schema)?;
        Ok(SinkResult::CanHaveMoreInput)
    }

    fn combine(&mut self, other: &mut dyn Sink) {
        // don't parallelize this as this is already done in parallel.

        let other = other.as_any().downcast_ref::<Self>().unwrap();
        let n_partitions = self.pre_agg_partitions.len();
        debug_assert_eq!(n_partitions, other.pre_agg_partitions.len());

        self.pre_agg_partitions
            .iter_mut()
            .zip(other.pre_agg_partitions.iter())
            .for_each(|(map_self, map_other)| {
                for (k_other, &agg_idx_other) in map_other.iter() {
                    // the hash value
                    let h = k_other.hash;

                    // the offset in the keys of other
                    let idx_other = k_other.idx as usize;
                    // slice to the keys of other
                    let key_other = unsafe { other.keys.get_unchecked_release(idx_other) };

                    let entry = map_self.raw_entry_mut().from_hash(h, |k_self| {
                        h == k_self.hash && {
                            // the offset in the keys of self
                            let idx_self = k_self.idx as usize;
                            // slice to the keys of self
                            // SAFETY:
                            // in bounds
                            let key_self = unsafe { self.keys.get_unchecked_release(idx_self) };
                            // compare the keys
                            key_self == key_other
                        }
                    });

                    let agg_idx_self = match entry {
                        // the keys of other are not in this table, so we must update this table
                        RawEntryMut::Vacant(entry) => {
                            // get the current offset in the values buffer
                            let values_offset = unsafe {
                                NumCast::from(self.aggregators.len()).unwrap_unchecked_release()
                            };
                            // get the key, comprised of the hash and the current offset in the keys buffer
                            let key = unsafe {
                                Key::new(
                                    h,
                                    NumCast::from(self.keys.len()).unwrap_unchecked_release(),
                                )
                            };

                            // extend the keys buffer with the new key from other
                            self.keys.push(key_other.clone());

                            // insert the keys and values_offset
                            entry.insert(key, values_offset);
                            // initialize the new aggregators
                            for agg_fn in &self.agg_fns {
                                self.aggregators.push(agg_fn.split())
                            }
                            values_offset
                        },
                        RawEntryMut::Occupied(entry) => *entry.get(),
                    };

                    // combine the aggregation functions
                    for i in 0..self.aggregation_columns.len() {
                        unsafe {
                            let agg_fn_other = other
                                .aggregators
                                .get_unchecked_release(agg_idx_other as usize + i);
                            let agg_fn_self = self
                                .aggregators
                                .get_unchecked_release_mut(agg_idx_self as usize + i);
                            agg_fn_self.combine(agg_fn_other.as_any())
                        }
                    }
                }
            });
    }

    fn split(&self, thread_no: usize) -> Box<dyn Sink> {
        let mut new = Self::new_inner(
            self.key_column.clone(),
            self.aggregation_columns.clone(),
            self.agg_fns.iter().map(|func| func.split()).collect(),
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
            // the OOC group_by will call split as well and it should
            // not send continue spilling to disk
            let iot = iot.take().unwrap();
            self.ooc_state.ooc = false;

            Some((iot, self.split(0)))
        } else {
            None
        };
        finalize_group_by(dfs, &self.output_schema, self.slice, payload)
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
    fn fmt(&self) -> &str {
        "string_group_by"
    }
}

// write agg_idx to the hashes buffer.
pub(super) unsafe fn write_agg_idx(h: *mut IdxSize, i: usize, agg_idx: IdxSize) {
    h.add(i).write(agg_idx)
}

pub(super) fn apply_aggregate(
    agg_i: usize,
    chunk_idx: IdxSize,
    agg_idxs: &[IdxSize],
    aggregation_s: &Series,
    has_physical_agg: bool,
    aggregators: &mut [AggregateFunction],
) {
    macro_rules! apply_agg {
                ($self:expr, $macro:ident $(, $opt_args:expr)*) => {{
                    match $self.dtype() {
                        #[cfg(feature = "dtype-u8")]
                        DataType::UInt8 => $macro!($self.u8().unwrap(), pre_agg_primitive $(, $opt_args)*),
                        #[cfg(feature = "dtype-u16")]
                        DataType::UInt16 => $macro!($self.u16().unwrap(), pre_agg_primitive $(, $opt_args)*),
                        DataType::UInt32 => $macro!($self.u32().unwrap(), pre_agg_primitive $(, $opt_args)*),
                        DataType::UInt64 => $macro!($self.u64().unwrap(), pre_agg_primitive $(, $opt_args)*),
                        #[cfg(feature = "dtype-i8")]
                        DataType::Int8 => $macro!($self.i8().unwrap(), pre_agg_primitive $(, $opt_args)*),
                        #[cfg(feature = "dtype-i16")]
                        DataType::Int16 => $macro!($self.i16().unwrap(), pre_agg_primitive $(, $opt_args)*),
                        DataType::Int32 => $macro!($self.i32().unwrap(), pre_agg_primitive $(, $opt_args)*),
                        DataType::Int64 => $macro!($self.i64().unwrap(), pre_agg_primitive $(, $opt_args)*),
                        DataType::Float32 => $macro!($self.f32().unwrap(), pre_agg_primitive $(, $opt_args)*),
                        DataType::Float64 => $macro!($self.f64().unwrap(), pre_agg_primitive $(, $opt_args)*),
                        dt => panic!("not implemented for {:?}", dt),
                    }
                }};
            }

    if has_physical_agg && aggregation_s.dtype().is_numeric() {
        macro_rules! dispatch {
            ($ca:expr, $name:ident) => {{
                let arr = $ca.downcast_iter().next().unwrap();

                for (&agg_idx, av) in agg_idxs.iter().zip(arr.into_iter()) {
                    let i = agg_idx as usize + agg_i;
                    let agg_fn = unsafe { aggregators.get_unchecked_release_mut(i) };

                    agg_fn.$name(chunk_idx, av.copied())
                }
            }};
        }

        apply_agg!(aggregation_s, dispatch);
    } else {
        let mut iter = aggregation_s.phys_iter();
        for &agg_idx in agg_idxs.iter() {
            let i = agg_idx as usize + agg_i;
            let agg_fn = unsafe { aggregators.get_unchecked_release_mut(i) };
            agg_fn.pre_agg(chunk_idx, &mut iter)
        }
    }
}

#[inline]
fn get_entry<'a>(
    key_val: Option<&str>,
    h: u64,
    current_partition: &'a mut PlIdHashMap<Key, IdxSize>,
    keys: &[Option<smartstring::alias::String>],
) -> RawEntryMut<'a, Key, IdxSize, IdBuildHasher> {
    current_partition.raw_entry_mut().from_hash(h, |key| {
        // first compare the hash before we incur the cache miss
        key.hash == h && {
            let idx = key.idx as usize;
            unsafe { keys.get_unchecked_release(idx).as_deref() == key_val }
        }
    })
}
