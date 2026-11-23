use polars_ooc::{MostRecentSpillContext, ParameterFreeSpillContext};
use polars_utils::IdxSize;

use super::frame_with_keys::{FrameWithKeys, FrameWithKeysBuilder, SpillFrameWithKeys};
use super::split::{SplitRows, rows_or_all};
use crate::morsel::get_ideal_morsel_size;

/// Gathers the rows of one pipeline per partition into frames that are registered for spilling.
/// A frame is finished once it has the ideal morsel size in rows or about `max_frame_bytes`
/// bytes.
pub struct PartitionedWriter {
    /// The rows not yet in a frame of each partition, with their estimated bytes.
    builders: Vec<Option<(FrameWithKeysBuilder, usize)>>,
    frames: Vec<Vec<SpillFrameWithKeys>>,
    max_frame_bytes: usize,
    spill_ctx: MostRecentSpillContext,
    all_rows: Vec<IdxSize>,
}

impl PartitionedWriter {
    pub fn new(
        num_partitions: usize,
        max_frame_bytes: usize,
        spill_ctx: MostRecentSpillContext,
    ) -> Self {
        Self {
            builders: (0..num_partitions).map(|_| None).collect(),
            frames: (0..num_partitions).map(|_| Vec::new()).collect(),
            max_frame_bytes,
            spill_ctx,
            all_rows: Vec::new(),
        }
    }

    /// Appends the rows of `frame` of each partition in `partitions`, as `split` groups them
    /// (`None`: all rows, in partition 0). Panics unless the columns of `frame` are single chunks.
    ///
    /// # Safety
    /// The rows of `split` must be in-bounds of `frame`.
    pub async unsafe fn write(
        &mut self,
        frame: &FrameWithKeys,
        split: Option<&SplitRows>,
        partitions: impl Iterator<Item = usize>,
    ) {
        let num_rows = frame.keys.len();
        let frame_bytes = frame.estimated_size();
        let max_rows = get_ideal_morsel_size();
        for p in partitions {
            let rows = split.map(|split| split.partition(p));
            let rows = rows_or_all(rows, &mut self.all_rows, num_rows);
            if rows.is_empty() {
                continue;
            }
            let (builder, bytes) = self.builders[p].get_or_insert_with(|| {
                let builder = FrameWithKeysBuilder::new(frame.frame.schema().clone(), &frame.keys);
                (builder, 0)
            });
            unsafe { builder.gather_extend(&frame.frame, &frame.keys, rows) };
            *bytes += frame_bytes * rows.len() / num_rows;
            if builder.len() >= max_rows || *bytes >= self.max_frame_bytes {
                let frame = builder.freeze_reset();
                *bytes = 0;
                let frame = SpillFrameWithKeys::new(frame, &self.spill_ctx).await;
                self.frames[p].push(frame);
            }
        }
    }

    /// Finishes the frames of the remaining rows and returns the frames of each partition.
    pub fn finish(self) -> Vec<Vec<SpillFrameWithKeys>> {
        let Self {
            builders,
            mut frames,
            spill_ctx,
            ..
        } = self;
        for (builder, frames) in builders.into_iter().zip(&mut frames) {
            if let Some((mut builder, _)) = builder
                && !builder.is_empty()
            {
                let frame = SpillFrameWithKeys::new_unregistered(builder.freeze_reset());
                spill_ctx.register_no_spill_check(&frame);
                frames.push(frame);
            }
        }
        frames
    }
}
