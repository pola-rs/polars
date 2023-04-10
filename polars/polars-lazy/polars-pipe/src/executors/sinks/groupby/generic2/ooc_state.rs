use std::sync::Mutex;

use polars_core::config::verbose;

use super::*;
use crate::executors::sinks::io::IOThread;
use crate::executors::sinks::memory::MemTracker;
use crate::pipeline::morsels_per_sink;

#[derive(Clone)]
pub(super) struct OocState {
    // OOC
    // Stores available memory in the system at the start of this sink.
    // and stores the memory used by this this sink.
    mem_track: MemTracker,
    // sort in-memory or out-of-core
    pub(super) ooc: bool,
    // when ooc, we write to disk using an IO thread
    pub(super) io_thread: Arc<Mutex<Option<IOThread>>>,
}

impl Default for OocState {
    fn default() -> Self {
        Self {
            mem_track: MemTracker::new(morsels_per_sink()),
            ooc: false,
            io_thread: Default::default(),
        }
    }
}

// If this is reached we early merge the overflow buckets
// to free up memory
const EARLY_MERGE_THRESHOLD: f64 = 0.5;
// If this is reached we spill to disk and
// aggregate in a second run
const TO_DISK_THRESHOLD: f64 = 0.3;

impl OocState {
    fn init_ooc(&mut self) -> PolarsResult<()> {
        if verbose() {
            eprintln!("OOC groupby started");
        }
        self.ooc = true;

        // start IO thread
        let mut iot = self.io_thread.lock().unwrap();
        // if iot.is_none() {
        //     // todo!
        //     // *iot = Some(IOThread::try_new(input_schema, "groupby")?)
        // }
        Ok(())
    }

    pub(super) fn check_memory_usage(&mut self) -> PolarsResult<bool> {
        let free_frac = self.mem_track.free_memory_fraction_since_start();

        if free_frac < MEMORY_FRACTION_THRESHOLD {
            self.init_ooc()?;
        } else if free_frac < EARLY_MERGE_THRESHOLD {
            return Ok(true);
        }
        Ok(false)
    }
}
