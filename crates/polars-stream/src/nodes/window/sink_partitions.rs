use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use parking_lot::Mutex;
use polars_arrow::array::builder::ShareStrategy;
use polars_core::frame::builder::DataFrameBuilder;
use polars_core::prelude::*;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_ooc::{
    MostRecentSpillContext, ParameterFreeSpillContext, SpillFrame, estimate_memory_usage,
    memory_manager,
};
use polars_utils::aliases::PlRandomState;
use polars_utils::hashing::HashPartitioner;

use super::WindowParams;
use super::tuning::WindowTuning;
use crate::hash_shuffle::split::{NullRows, split};

/// A partition merges its small frames once it has this many.
const MAX_SMALL_FRAMES: usize = 16;

/// The rows of one partition received so far.
#[derive(Default)]
struct PartitionBuffer {
    builder: Option<DataFrameBuilder>,
    /// The estimated size of `frames`.
    frames_bytes: usize,
    /// The estimated size of the rows in `builder`.
    builder_bytes: usize,
    /// Frames below the minimum frame size, not registered for spilling.
    small: Vec<DataFrame>,
    small_bytes: usize,
    frames: Vec<SpillFrame>,
}

impl PartitionBuffer {
    fn freeze(&mut self, min_frame_bytes: usize, spill_ctx: &MostRecentSpillContext) {
        let Some(builder) = self.builder.as_mut() else {
            return;
        };
        if builder.is_empty() {
            return;
        }
        let df = builder.freeze_reset();
        self.builder_bytes = 0;
        let bytes = df.estimated_size(false);
        if bytes >= min_frame_bytes {
            self.register(df, spill_ctx);
            return;
        }
        self.small.push(df);
        self.small_bytes += bytes;
        if self.small_bytes >= min_frame_bytes || self.small.len() >= MAX_SMALL_FRAMES {
            let mut df = accumulate_dataframes_vertical_unchecked(std::mem::take(&mut self.small));
            df.rechunk_mut();
            self.small_bytes = df.estimated_size(false);
            if self.small_bytes >= min_frame_bytes {
                self.small_bytes = 0;
                self.register(df, spill_ctx);
            } else {
                self.small.push(df);
            }
        }
    }

    fn register(&mut self, df: DataFrame, spill_ctx: &MostRecentSpillContext) {
        self.frames_bytes += df.estimated_size(false);
        let sf = SpillFrame::new_unregistered(df);
        spill_ctx.register_no_spill_check(&sf);
        self.frames.push(sf);
    }

    /// All frames of the partition, the small ones merged into one, and their estimated size.
    fn finish(
        mut self,
        min_frame_bytes: usize,
        spill_ctx: &MostRecentSpillContext,
    ) -> (Vec<SpillFrame>, usize) {
        self.freeze(min_frame_bytes, spill_ctx);
        if !self.small.is_empty() {
            let mut df = accumulate_dataframes_vertical_unchecked(std::mem::take(&mut self.small));
            df.rechunk_mut();
            self.register(df, spill_ctx);
        }
        (self.frames, self.frames_bytes)
    }
}

/// Hash partitions whole input rows as they arrive, into frames that can be spilled. Until the
/// morsels stored whole need more than `max_whole_bytes`, the rows are not partitioned.
pub struct SinkPartitions {
    whole_bytes: AtomicUsize,
    max_whole_bytes: usize,
    partitioning: AtomicBool,
    partitions: Vec<Mutex<PartitionBuffer>>,
    partitioner: HashPartitioner,
    random_state: PlRandomState,
    frame_bytes: usize,
    /// Smaller frames are kept in memory until they are merged into a larger frame, so that
    /// every registered frame can be spilled.
    min_frame_bytes: usize,
    spill_ctx: MostRecentSpillContext,
    /// Set once a pressure pass is running, so that only one task freezes all buffers at a time.
    freezing_all: AtomicBool,
}

impl SinkPartitions {
    pub fn new(tuning: &WindowTuning) -> Self {
        let (num_partitions, frame_bytes, min_frame_bytes) = tuning.sink_partitions();
        Self {
            whole_bytes: AtomicUsize::new(0),
            max_whole_bytes: tuning.max_whole_bytes(),
            partitioning: AtomicBool::new(false),
            partitions: (0..num_partitions).map(|_| Mutex::default()).collect(),
            partitioner: HashPartitioner::new(num_partitions, 0),
            random_state: PlRandomState::default(),
            frame_bytes,
            min_frame_bytes,
            spill_ctx: MostRecentSpillContext::new("window-partitions".into(), None),
            freezing_all: AtomicBool::new(false),
        }
    }

    /// Whether rows are partitioned as they arrive.
    pub fn is_partitioning(&self) -> bool {
        self.partitioning.load(Ordering::Relaxed)
    }

    /// Counts a morsel that is stored whole, and starts partitioning once they need too much
    /// memory.
    pub fn add_whole_bytes(&self, bytes: usize) {
        let total = self.whole_bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;
        if total > self.max_whole_bytes {
            self.partitioning.store(true, Ordering::Relaxed);
        }
    }

    /// Adds the rows of `df` to their partitions.
    pub async fn insert(&self, df: DataFrame, params: &WindowParams) -> PolarsResult<()> {
        self.add(df, params)?;
        memory_manager().spill().await;
        Ok(())
    }

    pub fn insert_blocking(&self, df: DataFrame, params: &WindowParams) -> PolarsResult<()> {
        self.add(df, params)?;
        memory_manager().spill_blocking();
        Ok(())
    }

    fn add(&self, mut df: DataFrame, params: &WindowParams) -> PolarsResult<()> {
        let height = df.height();
        if height == 0 {
            return Ok(());
        }
        // Gathering rows by index needs single-chunk columns.
        df.rechunk_mut();
        let row_bytes = df.estimated_size(false) as f64 / height as f64;
        let keys = df.select(params.partition_by.iter().cloned())?;
        let keys = params
            .key_encoder
            .hash_keys(&keys, self.random_state.clone(), true);
        let split = split(&keys, None, &self.partitioner, NullRows::FirstPartition);
        drop(keys);

        for (p, buffer) in self.partitions.iter().enumerate() {
            let rows = split.as_ref().map(|split| split.partition(p));
            let num_rows = rows.map_or(height, |rows| rows.len());
            if num_rows == 0 {
                continue;
            }
            let mut buffer = buffer.lock();
            let builder = buffer
                .builder
                .get_or_insert_with(|| DataFrameBuilder::new(df.schema().clone()));
            match rows {
                None => builder.extend(&df, ShareStrategy::Never),
                // SAFETY: the rows were split from `df`.
                Some(rows) => unsafe { builder.gather_extend(&df, rows, ShareStrategy::Never) },
            }
            buffer.builder_bytes += (num_rows as f64 * row_bytes) as usize;
            if buffer.builder_bytes >= self.frame_bytes {
                buffer.freeze(self.min_frame_bytes, &self.spill_ctx);
            }
        }
        drop(df);

        // The sizes above are averages, so check the real memory use too.
        if estimate_memory_usage() > polars_config::config().ooc_memory_budget_bytes()
            && !self.freezing_all.swap(true, Ordering::Relaxed)
        {
            for buffer in &self.partitions {
                buffer.lock().freeze(self.min_frame_bytes, &self.spill_ctx);
            }
            self.freezing_all.store(false, Ordering::Relaxed);
        }
        Ok(())
    }

    /// The frames of each partition, their estimated size and the spill context that keeps them
    /// spillable.
    pub fn finish(self) -> (Vec<Vec<SpillFrame>>, usize, MostRecentSpillContext) {
        let mut bytes = 0;
        let partitions = self
            .partitions
            .into_iter()
            .map(|buffer| {
                let (frames, frames_bytes) = buffer
                    .into_inner()
                    .finish(self.min_frame_bytes, &self.spill_ctx);
                bytes += frames_bytes;
                frames
            })
            .collect();
        (partitions, bytes, self.spill_ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_frames_stay_bounded() {
        let spill_ctx = MostRecentSpillContext::new("test".into(), None);
        let min_frame_bytes = 4096;
        let df = df!["a" => [1i64; 8]].unwrap();
        let mut buffer = PartitionBuffer::default();
        for _ in 0..1000 {
            buffer
                .builder
                .get_or_insert_with(|| DataFrameBuilder::new(df.schema().clone()))
                .extend(&df, ShareStrategy::Never);
            buffer.freeze(min_frame_bytes, &spill_ctx);
            assert!(buffer.small.len() < MAX_SMALL_FRAMES);
            assert!(buffer.small_bytes < min_frame_bytes);
        }
        assert!(!buffer.frames.is_empty());
        let (frames, _) = buffer.finish(min_frame_bytes, &spill_ctx);
        assert_eq!(frames.iter().map(|sf| sf.height()).sum::<usize>(), 8000);
    }
}
