use std::collections::VecDeque;
use std::ops::Range;
use std::sync::Arc;

use polars_async::executor::{AbortOnDropHandle, TaskMetricAggregator, spawn};
use polars_async::primitives::distributor_channel::{self};
use polars_buffer::Buffer;
use polars_core::runtime::ASYNC;
use polars_error::PolarsResult;
use polars_io::prelude::_csv_read_internal::CountLines;
use polars_io::utils::compression::ByteSourceReader;
use polars_io::utils::slice::SplitSlicePosition;
use polars_io::utils::stream_buf_reader::ReaderSource;
use polars_utils::mem::prefetch::prefetch_l2;
use polars_utils::slice_enum::Slice;

use super::{NO_SLICE, SLICE_ENDED};
use crate::nodes::{MorselSeq, TaskPriority};
use crate::utils::tokio_handle_ext;

pub(super) struct LineBatch {
    // Safety: All receivers (LineBatchProcessors) hold a Buffer ref to this.
    pub(super) mem_slice: Buffer<u8>,
    pub(super) n_lines: usize,
    pub(super) slice: (usize, usize),
    /// Position of this chunk relative to the start of the file according to CountLines.
    pub(super) row_offset: usize,
    pub(super) morsel_seq: MorselSeq,
}

pub(super) struct LineBatchSource {
    pub(super) base_leftover: Buffer<u8>,
    pub(super) reader: ByteSourceReader<ReaderSource>,
    pub(super) line_counter: CountLines,
    pub(super) line_batch_tx: distributor_channel::Sender<LineBatch>,
    pub(super) pre_slice: Option<Slice>,
    pub(super) needs_full_row_count: bool,
    pub(super) num_pipelines: usize,
    pub(super) task_metrics: Option<Arc<TaskMetricAggregator>>,
    pub(super) verbose: bool,
}

impl LineBatchSource {
    /// Returns the number of rows skipped from the start of the file according to CountLines.
    pub(crate) async fn run(self) -> PolarsResult<usize> {
        // With a comment prefix every block would need to start at a line start, so that case uses
        // the serial path.
        if self.line_counter.has_comment_prefix() {
            self.run_serial().await
        } else {
            self.run_parallel().await
        }
    }

    /// Splits the input into newline-aligned batches in parallel.
    ///
    /// A blocking task reads the input as a sequence of segments (zero-copy where possible), which
    /// are split into blocks that are analyzed in parallel. Since [`CountLines::analyze_chunk`]
    /// gives the result for both possible starting states (inside or outside a string), only a
    /// cheap serial pass that picks the correct result for each block remains. Only rows crossing
    /// a segment boundary are copied.
    async fn run_parallel(self) -> PolarsResult<usize> {
        if self.verbose {
            eprintln!("[CsvSource]: Start line splitting parallel");
        }

        let LineBatchSource {
            base_leftover,
            mut reader,
            line_counter,
            mut line_batch_tx,
            pre_slice,
            needs_full_row_count,
            num_pipelines,
            task_metrics,
            verbose: _,
        } = self;

        // Batches are larger than the analyzed blocks, since every batch has a fixed overhead
        // downstream. Small sizes in debug mode to exercise the boundary logic in tests.
        let (initial_size, max_block_size, max_batch_size, max_segment_size) =
            if cfg!(debug_assertions) {
                (7, 128, 300, 997)
            } else {
                (
                    ByteSourceReader::<ReaderSource>::initial_read_size(),
                    ByteSourceReader::<ReaderSource>::ideal_read_size(),
                    4 * ByteSourceReader::<ReaderSource>::ideal_read_size(),
                    usize::MAX,
                )
            };

        // Task: Read segments. This may block, so it runs on tokio's elastic blocking pool.
        let (segment_tx, mut segment_rx) = tokio::sync::mpsc::channel::<Buffer<u8>>(1);
        let segment_reader_handle =
            tokio_handle_ext::AbortOnDropHandle(ASYNC.spawn_blocking(move || {
                const MAX_READ_SIZE: usize = 4 * 1024 * 1024;
                let mut read_size = ByteSourceReader::<ReaderSource>::initial_read_size();
                let mut segment = base_leftover;

                loop {
                    for start in (0..segment.len()).step_by(max_segment_size) {
                        let end = start.saturating_add(max_segment_size).min(segment.len());
                        if segment_tx
                            .blocking_send(segment.clone().sliced(start..end))
                            .is_err()
                        {
                            return Ok(());
                        }
                    }

                    segment = match &mut reader {
                        // Take prefetched chunks without copying.
                        ByteSourceReader::UncompressedStream(ReaderSource::Streaming(r)) => {
                            r.next_buffer()?
                        },
                        // Zero-copy, everything at once.
                        ByteSourceReader::UncompressedMemory { .. } => {
                            reader.read_next_slice(&Buffer::new(), usize::MAX, None)?.0
                        },
                        _ => reader.read_next_slice(&Buffer::new(), read_size, None)?.0,
                    };
                    if segment.is_empty() {
                        return PolarsResult::Ok(());
                    }
                    read_size = (read_size * 4).min(MAX_READ_SIZE);
                }
            }));

        let mut slicer = BatchSlicer::new(pre_slice, needs_full_row_count);

        // Ramp up the lookahead so that small slices don't analyze far beyond what they need.
        let max_blocks_in_flight = 2 * num_pipelines.max(1);
        let mut blocks_in_flight_limit = 1;
        let mut blocks_in_flight = VecDeque::with_capacity(max_blocks_in_flight);

        // Segment that is being split into blocks.
        let mut spawn_segment = Buffer::default();
        let mut spawn_offset = 0;
        let mut block_size = initial_size;

        // Segment that is being split into batches.
        let mut segment = Buffer::default();
        let mut prev_cut = 0;
        // The pending batch is `segment[prev_cut..batch_end]` containing `batch_lines` rows.
        let mut batch_end = 0;
        let mut batch_lines = 0;
        let mut batch_size = initial_size;
        // Start of the current row if it began in an earlier segment.
        let mut carry = Vec::new();
        let mut in_string = false;

        let reached_end = loop {
            while blocks_in_flight.len() < blocks_in_flight_limit {
                if spawn_offset == spawn_segment.len() {
                    // Keeps returning None once all segments are received.
                    let Some(next_segment) = segment_rx.recv().await else {
                        break;
                    };
                    spawn_segment = next_segment;
                    spawn_offset = 0;
                    block_size = initial_size;
                }

                let block = spawn_offset..(spawn_offset + block_size).min(spawn_segment.len());
                spawn_offset = block.end;
                block_size = (block_size * 4).min(max_block_size);

                let block_bytes = spawn_segment.clone().sliced(block.clone());
                let line_counter = line_counter.clone();
                let handle = AbortOnDropHandle::new(spawn(
                    TaskPriority::High,
                    task_metrics.as_deref(),
                    async move { line_counter.analyze_chunk(&block_bytes) },
                ));
                blocks_in_flight.push_back((spawn_segment.clone(), block, handle));
            }

            let Some((block_segment, block, handle)) = blocks_in_flight.pop_front() else {
                break true;
            };
            blocks_in_flight_limit = (blocks_in_flight_limit * 2).min(max_blocks_in_flight);

            if block.start == 0 {
                // First block of a new segment, carry over the unfinished row.
                carry.extend_from_slice(&segment[prev_cut..]);
                segment = block_segment;
                prev_cut = 0;
                batch_end = 0;
            }

            let stats = handle.await[in_string as usize];
            in_string = stats.end_inside_string;

            if stats.newline_count > 0 {
                batch_end = block.start + stats.last_newline_offset + 1;
                batch_lines += stats.newline_count;
            }

            // Cut early if there is a carry and at the end of a segment, so that only rows crossing
            // a segment boundary are copied.
            let must_cut = !carry.is_empty() || block.end == segment.len();
            if batch_lines == 0 || (!must_cut && batch_end - prev_cut < batch_size) {
                continue;
            }

            let batch_slice = take_rows(&mut carry, &segment, prev_cut..batch_end);
            prev_cut = batch_end;
            batch_size = (batch_size * 4).min(max_batch_size);

            if !send_batch(
                &mut slicer,
                &mut line_batch_tx,
                batch_slice,
                std::mem::take(&mut batch_lines),
            )
            .await
            {
                break false;
            }
        };

        if reached_end {
            // Propagate read errors, they also end the segments.
            segment_reader_handle.await.unwrap()?;

            // The potentially unterminated final row.
            let tail = take_rows(&mut carry, &segment, prev_cut..segment.len());
            if !tail.is_empty() {
                let (n_lines, _) = line_counter.count_rows(&tail, true);
                send_batch(&mut slicer, &mut line_batch_tx, tail, n_lines).await;
            }
        }

        Ok(slicer.n_rows_skipped)
    }

    /// Serial path, the read loop runs on tokio's elastic blocking pool to avoid starvation of
    /// the polars-stream executor.
    async fn run_serial(self) -> PolarsResult<usize> {
        let verbose = self.verbose;
        let mut line_batch_tx = self.line_batch_tx;

        let read_loop_handle =
            tokio_handle_ext::AbortOnDropHandle(ASYNC.spawn_blocking(move || {
                if verbose {
                    eprintln!("[CsvSource]: Start line splitting serial");
                }

                let use_l2_prefetch =
                    matches!(self.reader, ByteSourceReader::UncompressedMemory { .. });

                let mut producer = LineBatchProducer::new(
                    self.reader,
                    self.base_leftover,
                    self.line_counter,
                    self.pre_slice,
                    self.needs_full_row_count,
                    use_l2_prefetch,
                );

                while let Some(batch) = producer.next_batch()? {
                    // Effectively, this is `blocking_send`.
                    if ASYNC.block_on(line_batch_tx.send(batch)).is_err() {
                        break;
                    }
                }

                PolarsResult::Ok(producer.n_rows_skipped())
            }));

        let n_rows_skipped = read_loop_handle.await.unwrap()?;

        Ok(n_rows_skipped)
    }
}

/// Returns `segment[range]`, prefixed by and consuming `carry` if it is non-empty.
fn take_rows(carry: &mut Vec<u8>, segment: &Buffer<u8>, range: Range<usize>) -> Buffer<u8> {
    if carry.is_empty() {
        segment.clone().sliced(range)
    } else {
        carry.extend_from_slice(&segment[range]);
        Buffer::from_vec(std::mem::take(carry))
    }
}

/// Returns false if no more batches should be sent.
async fn send_batch(
    slicer: &mut BatchSlicer,
    line_batch_tx: &mut distributor_channel::Sender<LineBatch>,
    mem_slice: Buffer<u8>,
    n_lines: usize,
) -> bool {
    match slicer.make_batch(mem_slice, n_lines) {
        BatchAction::Send(batch) => line_batch_tx.send(batch).await.is_ok(),
        BatchAction::Skip => true,
        BatchAction::Stop => false,
    }
}

enum BatchAction {
    Send(LineBatch),
    /// The batch lies entirely before the slice.
    Skip,
    /// The batch lies entirely after the slice and no full row count is needed.
    Stop,
}

/// Applies the pre-slice to a sequence of newline-aligned batches and assigns row offsets and
/// morsel sequence ids.
struct BatchSlicer {
    global_slice: Option<Range<usize>>,
    needs_full_row_count: bool,
    row_offset: usize,
    morsel_seq: MorselSeq,
    n_rows_skipped: usize,
}

impl BatchSlicer {
    fn new(pre_slice: Option<Slice>, needs_full_row_count: bool) -> Self {
        let global_slice = if let Some(pre_slice) = pre_slice {
            match pre_slice {
                Slice::Positive { .. } => Some(Range::<usize>::from(pre_slice)),
                // IR lowering puts negative slice in separate node.
                // TODO: Native line buffering for negative slice
                Slice::Negative { .. } => unreachable!(),
            }
        } else {
            None
        };

        Self {
            global_slice,
            needs_full_row_count,
            row_offset: 0,
            morsel_seq: MorselSeq::default(),
            n_rows_skipped: 0,
        }
    }

    fn make_batch(&mut self, mem_slice: Buffer<u8>, n_lines: usize) -> BatchAction {
        // Has to happen here before slicing, since there are slice operations that skip morsel
        // sending.
        let prev_row_offset = self.row_offset;
        self.row_offset += n_lines;

        let slice = if let Some(global_slice) = &self.global_slice {
            match SplitSlicePosition::split_slice_at_file(
                prev_row_offset,
                n_lines,
                global_slice.clone(),
            ) {
                SplitSlicePosition::Before => {
                    self.n_rows_skipped = self.n_rows_skipped.saturating_add(n_lines);
                    return BatchAction::Skip;
                },
                SplitSlicePosition::Overlapping(offset, len) => (offset, len),
                SplitSlicePosition::After => {
                    if self.needs_full_row_count {
                        SLICE_ENDED
                    } else {
                        return BatchAction::Stop;
                    }
                },
            }
        } else {
            NO_SLICE
        };

        self.morsel_seq = self.morsel_seq.successor();

        BatchAction::Send(LineBatch {
            mem_slice,
            n_lines,
            slice,
            row_offset: self.row_offset,
            morsel_seq: self.morsel_seq,
        })
    }
}

/// Produces LineBatches from a ByteSourceReader. Callers decide how to send each batch.
struct LineBatchProducer {
    reader: ByteSourceReader<ReaderSource>,
    prev_leftover: Buffer<u8>,
    line_counter: CountLines,
    slicer: BatchSlicer,
    use_prefetch_l2: bool,
    read_size: usize,
    finished: bool,
}

impl LineBatchProducer {
    fn new(
        reader: ByteSourceReader<ReaderSource>,
        base_leftover: Buffer<u8>,
        line_counter: CountLines,
        pre_slice: Option<Slice>,
        needs_full_row_count: bool,
        use_prefetch_l2: bool,
    ) -> Self {
        Self {
            reader,
            prev_leftover: base_leftover,
            line_counter,
            slicer: BatchSlicer::new(pre_slice, needs_full_row_count),
            use_prefetch_l2,
            read_size: ByteSourceReader::<ReaderSource>::initial_read_size(),
            finished: false,
        }
    }

    /// Returns the next LineBatch, or None if the source is exhausted.
    fn next_batch(&mut self) -> PolarsResult<Option<LineBatch>> {
        if self.finished {
            return Ok(None);
        }

        loop {
            let (mem_slice, bytes_read) = self.reader.read_next_slice(
                &self.prev_leftover,
                self.read_size,
                Some(self.read_size),
            )?;

            if mem_slice.is_empty() {
                self.finished = true;
                return Ok(None);
            }

            if self.use_prefetch_l2 {
                prefetch_l2(&mem_slice);
            }

            let is_eof = bytes_read == 0;
            let (n_lines, unconsumed_offset) = self.line_counter.count_rows(&mem_slice, is_eof);

            let batch_slice = mem_slice.clone().sliced(0..unconsumed_offset);
            self.prev_leftover = mem_slice.sliced(unconsumed_offset..);

            if batch_slice.is_empty() && !is_eof {
                // Grow until at least a single row is included.
                self.read_size = self.read_size.saturating_mul(2);
                continue;
            }

            let batch = match self.slicer.make_batch(batch_slice, n_lines) {
                BatchAction::Send(batch) => batch,
                BatchAction::Skip => continue,
                BatchAction::Stop => {
                    self.finished = true;
                    return Ok(None);
                },
            };

            if is_eof {
                self.finished = true;
            }

            if self.read_size < ByteSourceReader::<ReaderSource>::ideal_read_size() {
                self.read_size *= 4;
            }

            return Ok(Some(batch));
        }
    }

    fn n_rows_skipped(&self) -> usize {
        self.slicer.n_rows_skipped
    }
}
