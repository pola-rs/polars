use std::sync::Mutex;

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
    pub(super) agg_idx_ooc: Vec<IdxSize>,
    // when ooc, we write to disk using an IO thread
    pub(super) io_thread: Arc<Mutex<Option<IOThread>>>,
    // This slice holds the partition numbers
    // this will materialize into `IdxCa` with memmap
    partitions: Option<Arc<[IdxSize]>>,
}

impl OocState {
    pub(super) fn new(io_thread: Option<Arc<Mutex<Option<IOThread>>>>, ooc: bool) -> Self {
        Self {
            mem_track: MemTracker::new(morsels_per_sink()),
            ooc,
            ooc_filter: vec![],
            agg_idx_ooc: vec![],
            io_thread: io_thread.unwrap_or_default(),
            partitions: None,
        }
    }

    pub(super) fn init_ooc(&mut self, input_schema: SchemaRef) -> PolarsResult<()> {
        if verbose() {
            eprintln!("OOC groupby started");
        }
        self.ooc = true;
        self.partitions = Some(Arc::from_iter((0 as IdxSize)..(PARTITION_SIZE as IdxSize)));

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

    pub(super) fn partitions(&self) -> Option<IdxCa> {
        unsafe {
            self.partitions
                .as_ref()
                .map(|parts| IdxCa::mmap_slice("", parts.as_ref()))
        }
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

    pub(super) fn dump(&self, partitions: Vec<DataFrame>) {
        let iot = self.io_thread.lock().unwrap();
        let iot = iot.as_ref().unwrap();

        let part_idx = self.partitions();
        iot.dump_iter(part_idx, Box::new(partitions.into_iter()))
    }
}
