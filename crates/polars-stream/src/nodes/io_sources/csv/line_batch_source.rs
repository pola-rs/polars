use std::ops::Range;

use polars_buffer::Buffer;
use polars_error::PolarsResult;
use polars_io::pl_async;
use polars_io::prelude::_csv_read_internal::CountLines;
use polars_io::utils::compression::ByteSourceReader;
use polars_io::utils::slice::SplitSlicePosition;
use polars_io::utils::stream_buf_reader::ReaderSource;
use polars_utils::mem::prefetch::prefetch_l2;
use polars_utils::slice_enum::Slice;

use super::{NO_SLICE, SLICE_ENDED};
use crate::async_primitives::distributor_channel::{self};
use crate::nodes::MorselSeq;
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
    pub(super) use_async_prefetch: bool,
    pub(super) verbose: bool,
}

impl LineBatchSource {
    /// Returns the number of rows skipped from the start of the file according to CountLines.
    pub(crate) async fn run(self) -> PolarsResult<usize> {
        if self.use_async_prefetch {
            self.run_async().await
        } else {
            self.run_direct().await
        }
    }

    /// Direct path for memory-mapped / non-streaming sources. No blocking calls.
    async fn run_direct(self) -> PolarsResult<usize> {
        if self.verbose {
            eprintln!("[CsvSource]: Start line splitting direct");
        }

        let use_l2_prefetch = true;

        let mut producer = LineBatchProducer::new(
            self.reader,
            self.base_leftover,
            self.line_counter,
            self.pre_slice,
            self.needs_full_row_count,
            use_l2_prefetch,
        );
        let mut line_batch_tx = self.line_batch_tx;

        while let Some(batch) = producer.next_batch()? {
            if line_batch_tx.send(batch).await.is_err() {
                break;
            }
        }

        Ok(producer.n_rows_skipped())
    }

    /// Streaming path for async-prefetched / compressed sources.
    /// The blocking read loop runs on tokio's elastic blocking pool to avoid
    /// starvation of the polars-stream executor.
    async fn run_async(self) -> PolarsResult<usize> {
        let verbose = self.verbose;
        let mut line_batch_tx = self.line_batch_tx;

        let read_loop_handle = tokio_handle_ext::AbortOnDropHandle(
            pl_async::get_runtime().spawn_blocking(move || {
                let handle = tokio::runtime::Handle::current();
                if verbose {
                    eprintln!("[CsvSource]: Start line splitting async");
                }

                let use_l2_prefetch = false;

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
                    if handle.block_on(line_batch_tx.send(batch)).is_err() {
                        break;
                    }
                }

                PolarsResult::Ok(producer.n_rows_skipped())
            }),
        );

        let n_rows_skipped = read_loop_handle.await.unwrap()?;

        Ok(n_rows_skipped)
    }
}

/// Produces LineBatches from a ByteSourceReader. Callers decide how to send each batch.
struct LineBatchProducer {
    reader: ByteSourceReader<ReaderSource>,
    prev_leftover: Buffer<u8>,
    line_counter: CountLines,
    global_slice: Option<Range<usize>>,
    needs_full_row_count: bool,
    use_prefetch_l2: bool,
    row_offset: usize,
    morsel_seq: MorselSeq,
    n_rows_skipped: usize,
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
            reader,
            prev_leftover: base_leftover,
            line_counter,
            global_slice,
            needs_full_row_count,
            use_prefetch_l2,
            row_offset: 0,
            morsel_seq: MorselSeq::default(),
            n_rows_skipped: 0,
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
                        continue;
                    },
                    SplitSlicePosition::Overlapping(offset, len) => (offset, len),
                    SplitSlicePosition::After => {
                        if self.needs_full_row_count {
                            SLICE_ENDED
                        } else {
                            self.finished = true;
                            return Ok(None);
                        }
                    },
                }
            } else {
                NO_SLICE
            };

            self.morsel_seq = self.morsel_seq.successor();

            let batch = LineBatch {
                mem_slice: batch_slice,
                n_lines,
                slice,
                row_offset: self.row_offset,
                morsel_seq: self.morsel_seq,
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
        self.n_rows_skipped
    }
}
