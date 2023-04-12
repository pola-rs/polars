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
    count: u16,
}

impl Default for OocState {
    fn default() -> Self {
        Self {
            mem_track: MemTracker::new(morsels_per_sink()),
            ooc: false,
            io_thread: Default::default(),
            count: 0,
        }
    }
}

// If this is reached we early merge the overflow buckets
// to free up memory
const EARLY_MERGE_THRESHOLD: f64 = 0.5;
// If this is reached we spill to disk and
// aggregate in a second run
const TO_DISK_THRESHOLD: f64 = 0.3;

pub(super) enum SpillAction {
    EarlyMerge,
    Dump,
    None,
}

impl OocState {
    fn init_ooc(&mut self, spill_schema: &dyn Fn() -> Option<Schema>) -> PolarsResult<()> {
        if verbose() {
            eprintln!("OOC groupby started");
        }
        self.ooc = true;

        // start IO thread
        let mut iot = self.io_thread.lock().unwrap();
        if iot.is_none() {
            if let Some(schema) = spill_schema() {
                *iot = Some(IOThread::try_new(Arc::new(schema), "groupby")?)
            }
        }
        Ok(())
    }

    pub(super) fn check_memory_usage(
        &mut self,
        spill_schema: &dyn Fn() -> Option<Schema>,
    ) -> PolarsResult<SpillAction> {
        if self.ooc {
            return Ok(SpillAction::Dump);
        }
        let free_frac = self.mem_track.free_memory_fraction_since_start();
        self.count += 1;

        if free_frac < TO_DISK_THRESHOLD {
            self.init_ooc(spill_schema)?;
            Ok(SpillAction::Dump)
        } else if free_frac < EARLY_MERGE_THRESHOLD
        // clean up some spills
         || (self.count % 512) == 0
        {
            Ok(SpillAction::EarlyMerge)
        } else {
            Ok(SpillAction::None)
        }
    }

    pub(super) fn dump(&self, partition_no: usize, df: DataFrame) {
        let iot = self.io_thread.lock().unwrap();
        let iot = iot.as_ref().unwrap();
        iot.dump_partition(partition_no as IdxSize, df)
    }
}
