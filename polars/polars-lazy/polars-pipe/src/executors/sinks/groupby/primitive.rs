use std::any::Any;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

use hashbrown::hash_map::RawEntryMut;
use num_traits::NumCast;
use polars_arrow::kernels::sort_partition::partition_to_groups_amortized;
use polars_core::config::verbose;
use polars_core::export::ahash::RandomState;
use polars_core::frame::row::AnyValueBuffer;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::utils::arrow::bitmap::utils::set_bit_unchecked;
use polars_core::utils::{
    _set_partition_size, accumulate_dataframes_vertical_unchecked, split_df_as_ref,
};
use polars_core::POOL;
use polars_utils::hash_to_partition;
use polars_utils::slice::GetSaferUnchecked;
use polars_utils::unwrap::UnwrapUncheckedRelease;
use rayon::prelude::*;

use super::aggregates::AggregateFn;
use crate::executors::sinks::groupby::aggregates::AggregateFunction;
use crate::executors::sinks::groupby::ooc::GroupBySource;
use crate::executors::sinks::groupby::physical_agg_to_logical;
use crate::executors::sinks::groupby::string::{apply_aggregate, write_agg_idx};
use crate::executors::sinks::groupby::utils::compute_slices;
use crate::executors::sinks::io::{block_thread_until_io_thread_done, IOThread};
use crate::executors::sinks::memory::MemTracker;
use crate::executors::sinks::utils::load_vec;
use crate::executors::sinks::HASHMAP_INIT_SIZE;
use crate::expressions::PhysicalPipedExpr;
use crate::operators::{DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult};
use crate::pipeline::{morsels_per_sink, PARTITION_SIZE};

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
    pre_agg_partitions: Vec<PlIdHashMap<Key<Option<K::Native>>, IdxSize>>,
    // the aggregations are all tightly packed
    // the aggregation function of a group can be found
    // by:
    //      * offset = (idx)
    //      * end = (offset + n_aggs)
    aggregators: Vec<AggregateFunction>,
    key: Arc<dyn PhysicalPipedExpr>,
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
    hashes: Vec<u64>,
    slice: Option<(i64, usize)>,
    // for sorted fast paths
    sort_partitions: Vec<[IdxSize; 2]>,

    // Stores available memory in the system at the start of this sink.
    // and stores the memory used by this this sink.
    mem_track: MemTracker,
    // sort in-memory or out-of-core
    ooc: bool,
    // bitmap that indicates the rows that are processed ooc
    // will be mmap converted to `BooleanArray`.
    ooc_filter: Vec<u8>,
    agg_idx_ooc: Vec<IdxSize>,
    // when ooc, we write to disk using an IO thread
    io_thread: Arc<Mutex<Option<IOThread>>>,
    partitions: Option<Arc<[IdxSize]>>,
}

impl<K: PolarsNumericType> PrimitiveGroupbySink<K>
where
    ChunkedArray<K>: IntoSeries,
    K::Native: FxHash,
{
    pub(crate) fn new(
        key: Arc<dyn PhysicalPipedExpr>,
        aggregation_columns: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
        agg_fns: Vec<AggregateFunction>,
        input_schema: SchemaRef,
        output_schema: SchemaRef,
        slice: Option<(i64, usize)>,
        io_thread: Option<Arc<Mutex<Option<IOThread>>>>,
    ) -> Self {
        let ooc = std::env::var("POLARS_FORCE_OOC_GROUPBY").is_ok();
        Self::new_inner(
            key,
            aggregation_columns,
            agg_fns,
            input_schema,
            output_schema,
            slice,
            io_thread,
            ooc,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_inner(
        key: Arc<dyn PhysicalPipedExpr>,
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
        let n_morsels_per_sink = morsels_per_sink();

        let pre_agg = load_vec(partitions, || PlIdHashMap::with_capacity(HASHMAP_INIT_SIZE));
        let aggregators =
            Vec::with_capacity(HASHMAP_INIT_SIZE * aggregation_columns.len() * partitions);

        let mut out = Self {
            thread_no: 0,
            pre_agg_partitions: pre_agg,
            aggregators,
            key,
            aggregation_columns,
            hb,
            agg_fns,
            input_schema,
            output_schema,
            aggregation_series: vec![],
            hashes: vec![],
            slice,
            sort_partitions: vec![],
            mem_track: MemTracker::new(n_morsels_per_sink),
            ooc,
            ooc_filter: vec![],
            agg_idx_ooc: vec![],
            io_thread: io_thread.unwrap_or_default(),
            partitions: None,
        };
        if ooc {
            out.init_ooc().unwrap();
        }
        out
    }

    #[inline]
    fn number_of_aggs(&self) -> usize {
        self.aggregation_columns.len()
    }

    fn check_memory_usage(&mut self) -> PolarsResult<()> {
        if self.mem_track.free_memory_fraction_since_start() < 0.25 {
            self.init_ooc()?
        }
        Ok(())
    }

    fn reset_ooc_filter_rows(&mut self, len: usize) {
        // todo! single pass
        self.ooc_filter.fill(0);
        self.ooc_filter.resize_with(len / 8 + 1, || 0)
    }

    fn get_ooc_filter(&self, len: usize) -> BooleanChunked {
        unsafe { BooleanChunked::mmap_slice("", &self.ooc_filter, 0, len) }
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
                        let (offset, slice_len) = (*slice)?;
                        if agg_map.is_empty() {
                            return None;
                        }
                        // safety:
                        // we will not alias.
                        let ptr = aggregators as *mut AggregateFunction;
                        let agg_fns =
                            unsafe { std::slice::from_raw_parts_mut(ptr, aggregators_len) };
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
                            .map(|dtype| AnyValueBuffer::new(dtype, slice_len))
                            .collect::<Vec<_>>();

                        agg_map.into_iter().skip(offset).take(slice_len).for_each(
                            |(k, &offset)| {
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
                            },
                        );

                        let mut cols = Vec::with_capacity(1 + self.number_of_aggs());
                        cols.push(key_builder.finish().into_series());
                        cols.extend(buffers.into_iter().map(|buf| buf.into_series()));
                        physical_agg_to_logical(&mut cols, &self.output_schema);
                        Some(DataFrame::new_no_checks(cols))
                    })
                    .collect::<Vec<_>>();
            Ok(dfs)
        })
    }

    fn sink_sorted(&mut self, ca: &ChunkedArray<K>, chunk: DataChunk) -> PolarsResult<SinkResult> {
        let arr = ca.downcast_iter().next().unwrap();
        let values = arr.values().as_slice();
        partition_to_groups_amortized(values, 0, false, 0, &mut self.sort_partitions);

        let k = K::Native::get_k(self.hb.clone());
        let pre_agg_len = self.pre_agg_partitions.len();

        for group in &self.sort_partitions {
            let [offset, length] = group;
            let first_g_value = unsafe { *values.get_unchecked_release(*offset as usize) };
            let h = first_g_value._fx_hash(k);

            let agg_idx = insert_and_get(
                h,
                Some(first_g_value),
                pre_agg_len,
                &mut self.pre_agg_partitions,
                &mut self.aggregators,
                &self.agg_fns,
            );

            for (i, aggregation_s) in
                (0..self.number_of_aggs() as IdxSize).zip(&self.aggregation_series)
            {
                let agg_fn = unsafe {
                    self.aggregators
                        .get_unchecked_release_mut((agg_idx + i) as usize)
                };
                agg_fn.pre_agg_ordered(chunk.chunk_index, *offset, *length, aggregation_s)
            }
        }
        self.aggregation_series.clear();
        self.check_memory_usage()?;
        Ok(SinkResult::CanHaveMoreInput)
    }

    fn prepare_key_and_aggregation_series(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<Series> {
        let s = self.key.evaluate(chunk, context.execution_state.as_any())?;
        let s = s.to_physical_repr();
        let s = s.rechunk();

        // todo! ammortize allocation
        for phys_e in self.aggregation_columns.iter() {
            let s = phys_e.evaluate(chunk, context.execution_state.as_any())?;
            let s = s.to_physical_repr();
            self.aggregation_series.push(s.rechunk());
        }
        Ok(s)
    }

    fn aggregate(&mut self, agg_idxs: &[IdxSize], chunk: &DataChunk) {
        let chunk_idx = chunk.chunk_index;
        for (agg_i, aggregation_s) in (0..self.number_of_aggs()).zip(&self.aggregation_series) {
            let has_physical_agg = self.agg_fns[agg_i].has_physical_agg();
            apply_aggregate(
                agg_i,
                chunk_idx,
                agg_idxs,
                aggregation_s,
                has_physical_agg,
                &mut self.aggregators,
            );
        }

        self.aggregation_series.clear();
    }

    fn init_ooc(&mut self) -> PolarsResult<()> {
        if verbose() {
            eprintln!("OOC groupby started");
        }
        self.ooc = true;
        self.partitions = Some(Arc::from_iter((0 as IdxSize)..(PARTITION_SIZE as IdxSize)));

        // start IO thread
        let mut iot = self.io_thread.lock().unwrap();
        if iot.is_none() {
            *iot = Some(IOThread::try_new(self.input_schema.clone(), "groupby")?)
        }
        Ok(())
    }

    fn sink_ooc(
        &mut self,
        context: &PExecutionContext,
        chunk: DataChunk,
    ) -> PolarsResult<SinkResult> {
        let s = self.prepare_key_and_aggregation_series(context, &chunk)?;
        // cow -> &series -> &dyn series_trait -> &chunkedarray
        let ca: &ChunkedArray<K> = s.as_ref().as_ref();

        // ensure the hashes are set
        s.vec_hash(self.hb.clone(), &mut self.hashes).unwrap();

        let arr = ca.downcast_iter().next().unwrap();
        let pre_agg_len = self.pre_agg_partitions.len();

        let mut agg_idx_buf = vec![];

        // set all bits to false
        self.reset_ooc_filter_rows(ca.len());

        // different from standard sink
        // we only set aggregation idx when the entry in the hashmap already
        // exists. This way we don't grow the hashmap
        // rows that are not processed are sinked to disk and loaded in a second pass
        for (iteration_idx, (opt_v, &h)) in arr.iter().zip(self.hashes.iter()).enumerate() {
            let opt_v = opt_v.copied();
            if let Some(agg_idx) =
                try_insert_and_get(h, opt_v, pre_agg_len, &mut self.pre_agg_partitions)
            {
                agg_idx_buf.push(agg_idx);
            } else {
                // set this row to true: e.g. processed ooc
                unsafe {
                    // safety: bchk doesn't understand this borrow does not alias because it borrows from self
                    let ooc_filter = std::slice::from_raw_parts_mut(
                        self.ooc_filter.as_ptr() as *mut u8,
                        self.ooc_filter.len(),
                    );
                    // safety: we correctly set the length in `reset_in_memory_rows`
                    set_bit_unchecked(ooc_filter, iteration_idx, true)
                }
            }
        }

        // needed for bchk
        let agg_idxs = std::mem::take(&mut self.agg_idx_ooc);
        self.aggregate(&agg_idxs, &chunk);
        self.agg_idx_ooc = agg_idxs;

        // reset the agg_idx buf
        self.agg_idx_ooc.clear();

        let ooc_filter = self.get_ooc_filter(ca.len());
        let df = chunk.data._filter_seq(&ooc_filter).unwrap();
        let partitions = split_df_as_ref(&df, PARTITION_SIZE)?;
        self.dump(partitions);

        Ok(SinkResult::CanHaveMoreInput)
    }

    fn dump(&self, partitions: Vec<DataFrame>) {
        let iot = self.io_thread.lock().unwrap();
        let iot = iot.as_ref().unwrap();

        let part_idx = unsafe {
            self.partitions
                .as_ref()
                .map(|parts| IdxCa::mmap_slice("", parts.as_ref()))
        };
        iot.dump_iter(part_idx, Box::new(partitions.into_iter()))
    }
}

impl<K: PolarsNumericType> Sink for PrimitiveGroupbySink<K>
where
    K::Native: Hash + Eq + Debug + FxHash,
    ChunkedArray<K>: IntoSeries,
{
    fn sink(&mut self, context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult> {
        if self.ooc {
            return self.sink_ooc(context, chunk);
        }

        let s = self.prepare_key_and_aggregation_series(context, &chunk)?;
        // cow -> &series -> &dyn series_trait -> &chunkedarray
        let ca: &ChunkedArray<K> = s.as_ref().as_ref();

        // sorted fast path
        if matches!(ca.is_sorted_flag2(), IsSorted::Ascending) && ca.null_count() == 0 {
            return self.sink_sorted(ca, chunk);
        }

        s.vec_hash(self.hb.clone(), &mut self.hashes).unwrap();

        // this reuses the hashes buffer as [u64] as idx buffer as [idxsize]
        // write the hashes to self.hashes buffer
        // s.vec_hash(self.hb.clone(), &mut self.hashes).unwrap();
        // now we have written hashes, we take the pointer to this buffer
        // we will write the aggregation_function indexes in the same buffer
        // this is unsafe and we must check that we only write the hashes that
        // already read/taken. So we write on the slots we just read
        let agg_idx_ptr = self.hashes.as_ptr() as *mut i64 as *mut IdxSize;

        let arr = ca.downcast_iter().next().unwrap();
        let pre_agg_len = self.pre_agg_partitions.len();
        for (iteration_idx, (opt_v, &h)) in arr.iter().zip(self.hashes.iter()).enumerate() {
            let opt_v = opt_v.copied();
            let agg_idx = insert_and_get(
                h,
                opt_v,
                pre_agg_len,
                &mut self.pre_agg_partitions,
                &mut self.aggregators,
                &self.agg_fns,
            );
            // # Safety
            // we write to the hashes buffer we iterate over at the moment.
            // this is sound because the writes are trailing from iteration
            unsafe { write_agg_idx(agg_idx_ptr, iteration_idx, agg_idx) };
        }

        // note that this slice looks into the self.hashes buffer
        let agg_idxs = unsafe { std::slice::from_raw_parts(agg_idx_ptr, ca.len()) };
        self.aggregate(agg_idxs, &chunk);

        self.aggregation_series.clear();
        self.check_memory_usage()?;
        Ok(SinkResult::CanHaveMoreInput)
    }

    fn combine(&mut self, other: &mut dyn Sink) {
        // don't parallel this as this is already done in parallel.
        let other = other.as_any().downcast_ref::<Self>().unwrap();

        self.pre_agg_partitions
            .iter_mut()
            .zip(other.pre_agg_partitions.iter())
            .for_each(|(map_self, map_other)| {
                for (key, &agg_idx_other) in map_other.iter() {
                    let entry = map_self.raw_entry_mut().from_key(key);

                    let agg_idx_self = match entry {
                        RawEntryMut::Vacant(entry) => {
                            let offset = NumCast::from(self.aggregators.len()).unwrap();
                            entry.insert(*key, offset);
                            // initialize the aggregators
                            for agg_fn in &self.agg_fns {
                                self.aggregators.push(agg_fn.split2())
                            }
                            offset
                        }
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

    fn finalize(&mut self, _context: &PExecutionContext) -> PolarsResult<FinalizedSink> {
        let dfs = self.pre_finalize()?;
        let df = if dfs.is_empty() {
            DataFrame::from(self.output_schema.as_ref())
        } else {
            let mut df = accumulate_dataframes_vertical_unchecked(dfs);
            // re init to check duplicates
            unsafe { DataFrame::new(std::mem::take(df.get_columns_mut())) }?
        };

        if self.ooc {
            let mut iot = self.io_thread.lock().unwrap();
            // make sure that we reset the shared states
            // the OOC groupby will call split as well and it should
            // not send continue spilling to disk
            let iot = iot.take().unwrap();
            self.ooc = false;

            // we wait until all chunks are spilled
            block_thread_until_io_thread_done(&iot);

            Ok(FinalizedSink::Source(Box::new(GroupBySource::new(
                iot,
                df,
                self.split(0),
            )?)))
        } else {
            Ok(FinalizedSink::Finished(df))
        }
    }

    fn split(&self, thread_no: usize) -> Box<dyn Sink> {
        let mut new = Self::new_inner(
            self.key.clone(),
            self.aggregation_columns.clone(),
            self.agg_fns.iter().map(|func| func.split2()).collect(),
            self.input_schema.clone(),
            self.output_schema.clone(),
            self.slice,
            Some(self.io_thread.clone()),
            self.ooc,
        );
        new.hb = self.hb.clone();
        new.thread_no = thread_no;
        Box::new(new)
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
    fn fmt(&self) -> &str {
        "primitive_groupby"
    }
}

fn insert_and_get<T>(
    h: u64,
    opt_v: Option<T>,
    pre_agg_len: usize,
    pre_agg_partitions: &mut Vec<PlIdHashMap<Key<Option<T>>, IdxSize>>,
    current_aggregators: &mut Vec<AggregateFunction>,
    agg_fns: &Vec<AggregateFunction>,
) -> IdxSize
where
    T: NumericNative + FxHash,
{
    let part = hash_to_partition(h, pre_agg_len);
    let current_partition = unsafe { pre_agg_partitions.get_unchecked_release_mut(part) };

    let entry = current_partition
        .raw_entry_mut()
        .from_hash(h, |k| k.value == opt_v);
    match entry {
        RawEntryMut::Vacant(entry) => {
            let offset =
                unsafe { NumCast::from(current_aggregators.len()).unwrap_unchecked_release() };
            let key = Key {
                hash: h,
                value: opt_v,
            };
            entry.insert(key, offset);
            // initialize the aggregators
            for agg_fn in agg_fns {
                current_aggregators.push(agg_fn.split2())
            }
            offset
        }
        RawEntryMut::Occupied(entry) => *entry.get(),
    }
}

fn try_insert_and_get<T>(
    h: u64,
    opt_v: Option<T>,
    pre_agg_len: usize,
    pre_agg_partitions: &mut Vec<PlIdHashMap<Key<Option<T>>, IdxSize>>,
) -> Option<IdxSize>
where
    T: NumericNative + FxHash,
{
    let part = hash_to_partition(h, pre_agg_len);
    let current_partition = unsafe { pre_agg_partitions.get_unchecked_release_mut(part) };

    let entry = current_partition
        .raw_entry_mut()
        .from_hash(h, |k| k.value == opt_v);
    match entry {
        RawEntryMut::Vacant(_) => None,
        RawEntryMut::Occupied(entry) => Some(*entry.get()),
    }
}
