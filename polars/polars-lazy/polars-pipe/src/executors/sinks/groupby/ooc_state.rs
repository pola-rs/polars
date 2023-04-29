use std::sync::Mutex;

use polars_arrow::export::arrow::bitmap::utils::set_bit_unchecked;
use polars_core::config::verbose;
use polars_core::prelude::*;

use crate::executors::sinks::io::IOThread;
use crate::executors::sinks::memory::MemTracker;
use crate::pipeline::morsels_per_sink;

pub(super) struct OocState {
    // OOC
    // Stores available memory in the system at the start of this sink.
    // and stores the memory used by this this sink.
    _mem_track: MemTracker,
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
            _mem_track: MemTracker::new(morsels_per_sink()),
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

    pub(super) fn reset_ooc_filter_rows(&mut self, len: usize) {
        // todo! single pass
        self.ooc_filter.fill(0);
        self.ooc_filter.resize_with(len / 8 + 1, || 0)
    }

    pub(super) fn check_memory_usage(&mut self, _schema: &SchemaRef) -> PolarsResult<()> {
        // ooc is broken we will rewrite to generic table
        todo!()
    }

    #[inline]
    pub(super) unsafe fn set_row_as_ooc(&mut self, idx: usize) {
        // safety: should set the length in `reset_in_memory_rows`
        set_bit_unchecked(&mut self.ooc_filter, idx, true)
    }

    pub(super) fn dump(&self, _data: DataFrame, _hashes: &mut [u64]) {
        // ooc is broken we will rewrite to generic table
        todo!()
    }
}
