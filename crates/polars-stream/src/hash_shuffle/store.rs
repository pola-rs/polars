use std::collections::VecDeque;

use polars_ooc::{MostRecentSpillContext, ParameterFreeSpillContext};

use super::batches::LoadedFrame;
use super::frame_with_keys::{FrameWithKeysBuilder, SpillFrameWithKeys};
use super::split::rows_or_all;

/// The frames of one partition, per input.
pub struct StoredPartition {
    index: usize,
    frames_per_input: Vec<VecDeque<SpillFrameWithKeys>>,
    /// The rows and estimated bytes of the frames added for each input.
    rows_per_input: Vec<u64>,
    bytes_per_input: Vec<u64>,
    /// Keeps the frames registered, so they stay spillable while the partition exists.
    spill_ctx: MostRecentSpillContext,
}

impl StoredPartition {
    pub fn new(index: usize, num_inputs: usize, spill_ctx: MostRecentSpillContext) -> Self {
        Self {
            index,
            frames_per_input: (0..num_inputs).map(|_| VecDeque::new()).collect(),
            rows_per_input: vec![0; num_inputs],
            bytes_per_input: vec![0; num_inputs],
            spill_ctx,
        }
    }

    pub fn index(&self) -> usize {
        self.index
    }

    /// The estimated memory of the frames added, in bytes.
    pub fn estimated_size(&self) -> u64 {
        self.bytes_per_input.iter().sum()
    }

    /// The rows of the frames added for `input`.
    pub fn num_rows(&self, input: usize) -> u64 {
        self.rows_per_input[input]
    }

    /// The estimated memory of the frames added for `input`, in bytes.
    pub fn input_size(&self, input: usize) -> u64 {
        self.bytes_per_input[input]
    }

    fn push(&mut self, input: usize, frame: SpillFrameWithKeys) {
        self.rows_per_input[input] += frame.num_rows() as u64;
        self.bytes_per_input[input] += frame.estimated_size() as u64;
        self.frames_per_input[input].push_back(frame);
    }

    /// Gathers this partition's rows of every frame of `batch` by index into one frame per input:
    /// reserved exactly, registered for spilling without waiting. Inputs without rows here are
    /// skipped. Each input's columns are those of its first frame in the batch.
    pub fn write_batch(&mut self, batch: &[LoadedFrame]) {
        let p = self.index;
        let mut all_rows = Vec::new();
        for input in 0..self.frames_per_input.len() {
            let rows_per_frame = || {
                batch
                    .iter()
                    .filter(move |f| f.input == input)
                    .map(move |f| (f, f.partition_rows(p)))
            };
            let num_rows: usize = rows_per_frame()
                .map(|(f, rows)| rows.map_or(f.frame.keys.len(), |rows| rows.len()))
                .sum();
            if num_rows == 0 {
                continue;
            }

            let (first, _) = rows_per_frame().next().unwrap();
            let schema = first.frame.frame.schema().clone();
            let mut builder = FrameWithKeysBuilder::new(schema, &first.frame.keys);
            builder.reserve(num_rows);
            for (f, rows) in rows_per_frame() {
                let rows = rows_or_all(rows, &mut all_rows, f.frame.keys.len());
                if !rows.is_empty() {
                    unsafe { builder.gather_extend(&f.frame.frame, &f.frame.keys, rows) };
                }
            }

            let frame = SpillFrameWithKeys::new_unregistered(builder.freeze_reset());
            self.spill_ctx.register_no_spill_check(&frame);
            self.push(input, frame);
        }
    }

    /// Adds frames of `input`, which must be registered in the spill context of this partition.
    pub fn append(&mut self, input: usize, frames: Vec<SpillFrameWithKeys>) {
        for frame in frames {
            self.push(input, frame);
        }
    }

    /// The oldest frame of the first input that has frames left. The newest frames are the first
    /// to be spilled, so the oldest are the most likely to be in memory.
    pub fn pop_frame(&mut self) -> Option<(usize, SpillFrameWithKeys)> {
        self.frames_per_input
            .iter_mut()
            .enumerate()
            .find_map(|(input, frames)| Some((input, frames.pop_front()?)))
    }
}
