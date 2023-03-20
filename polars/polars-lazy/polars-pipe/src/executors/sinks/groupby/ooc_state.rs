use std::sync::Mutex;

use polars_arrow::export::arrow::bitmap::utils::set_bit_unchecked;
use polars_core::config::verbose;
use polars_core::prelude::*;

use crate::executors::sinks::groupby::MEMORY_FRACTION_THRESHOLD;
use crate::executors::sinks::io::IOThread;
use crate::executors::sinks::memory::MemTracker;
use crate::pipeline::{morsels_per_sink, PARTITION_SIZE};

pub(super) struct OocState {
    // OOC
    // Stores available memory in the system at the start of this sink.
    // and stores the memory used by this this sink.
    mem_track: MemTracker,
    // sort in-memory or out-of-core
    pub(super) ooc: bool,
    // bitmap that indicates the rows that are processed ooc
    // will be mmap converted to `BooleanArray`.
    pub(super) ooc_filter: Vec<u8>,
    // when ooc, we write to disk using an IO thread
    pub(super) io_thread: Arc<Mutex<Option<IOThread>>>,
}

impl OocState {
    pub(super) fn new(io_thread: Option<Arc<Mutex<Option<IOThread>>>>, ooc: bool) -> Self {
        Self {
            mem_track: MemTracker::new(morsels_per_sink()),
            ooc,
            ooc_filter: vec![],
            io_thread: io_thread.unwrap_or_default(),
        }
    }

    pub(super) fn init_ooc(&mut self, input_schema: SchemaRef) -> PolarsResult<()> {
        if verbose() {
            eprintln!("OOC groupby started");
        }
        self.ooc = true;

        // start IO thread
        let mut iot = self.io_thread.lock().unwrap();
        if iot.is_none() {
            *iot = Some(IOThread::try_new(input_schema, "groupby")?)
        }
        Ok(())
    }

    pub(super) fn get_ooc_filter(&self, len: usize) -> BooleanChunked {
        unsafe { BooleanChunked::mmap_slice("", &self.ooc_filter, 0, len) }
    }

    pub(super) fn reset_ooc_filter_rows(&mut self, len: usize) {
        // todo! single pass
        self.ooc_filter.fill(0);
        self.ooc_filter.resize_with(len / 8 + 1, || 0)
    }

    pub(super) fn check_memory_usage(&mut self, schema: &SchemaRef) -> PolarsResult<()> {
        if self.mem_track.free_memory_fraction_since_start() < MEMORY_FRACTION_THRESHOLD {
            self.init_ooc(schema.clone())?
        }
        Ok(())
    }

    #[inline]
    pub(super) unsafe fn set_row_as_ooc(&mut self, idx: usize) {
        // safety: should set the length in `reset_in_memory_rows`
        set_bit_unchecked(&mut self.ooc_filter, idx, true)
    }

    pub(super) fn dump(&self, data: DataFrame, hashes: &mut [u64]) {
        // we filter the rows that are not processed
        // these rows are spilled to disk
        let mask = self.get_ooc_filter(data.height());
        let df = data._filter_seq(&mask).unwrap();

        // determine partitions
        let parts = unsafe { UInt64Chunked::mmap_slice("", hashes) };
        let parts = parts.filter(&mask).unwrap();
        let parts = parts.apply_in_place(|h| h & (PARTITION_SIZE as u64 - 1));
        let gt = parts.group_tuples_perfect(
            PARTITION_SIZE - 1,
            false,
            (parts.len() / PARTITION_SIZE) * 2,
        );

        let mut part_idx = Vec::with_capacity(PARTITION_SIZE);
        let partitioned = gt
            .unwrap_idx()
            .iter()
            .map(|(first, group)| {
                let partition =
                    unsafe { *hashes.get_unchecked(first as usize) } & (PARTITION_SIZE as u64 - 1);
                part_idx.push(partition as IdxSize);

                // groups are in bounds
                unsafe { df._take_unchecked_slice(group, false) }
            })
            .collect::<Vec<_>>();

        let iot = self.io_thread.lock().unwrap();
        let iot = iot.as_ref().unwrap();
        let part_idx = Some(IdxCa::from_vec("", part_idx));

        iot.dump_iter(part_idx, Box::new(partitioned.into_iter()))
    }
}
