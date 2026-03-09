use std::ops::Range;

use polars_buffer::Buffer;
use polars_error::PolarsResult;
use polars_io::prelude::_csv_read_internal::CountLines;
use polars_io::utils::compression::ByteSourceReader;
use polars_io::utils::slice::SplitSlicePosition;
use polars_io::utils::stream_buf_reader::ReaderSource;
use polars_utils::mem::prefetch::prefetch_l2;
use polars_utils::slice_enum::Slice;

use super::{NO_SLICE, SLICE_ENDED};
use crate::async_primitives::distributor_channel::{self};
use crate::nodes::MorselSeq;

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
    pub(super) verbose: bool,
}

impl LineBatchSource {
    /// Returns the number of rows skipped from the start of the file according to CountLines.
    pub(crate) async fn run(self) -> PolarsResult<usize> {
        let LineBatchSource {
            base_leftover,
            mut reader,
            line_counter,
            mut line_batch_tx,
            pre_slice,
            needs_full_row_count,
            verbose,
        } = self;

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

        if verbose {
            eprintln!("[CsvSource]: Start line splitting",);
        }

        let mut prev_leftover = base_leftover;
        let mut row_offset = 0usize;
        let mut morsel_seq = MorselSeq::default();
        let mut n_rows_skipped: usize = 0;
        let mut read_size = ByteSourceReader::<ReaderSource>::initial_read_size();

        loop {
            let (mem_slice, bytes_read) =
                reader.read_next_slice(&prev_leftover, read_size, Some(read_size))?;
            if mem_slice.is_empty() {
                break;
            }

            prefetch_l2(&mem_slice);

            let is_eof = bytes_read == 0;
            let (n_lines, unconsumed_offset) = line_counter.count_rows(&mem_slice, is_eof);

            let batch_slice = mem_slice.clone().sliced(0..unconsumed_offset);
            prev_leftover = mem_slice.sliced(unconsumed_offset..);

            if batch_slice.is_empty() && !is_eof {
                // This allows the slice to grow until at least a single row is included. To avoid a quadratic run-time for large row sizes, we double the read size.
                read_size = read_size.saturating_mul(2);
                continue;
            }

            // Has to happen here before slicing, since there are slice operations that skip morsel
            // sending.
            let prev_row_offset = row_offset;
            row_offset += n_lines;

            let slice = if let Some(global_slice) = &global_slice {
                match SplitSlicePosition::split_slice_at_file(
                    prev_row_offset,
                    n_lines,
                    global_slice.clone(),
                ) {
                    // Note that we don't check that the skipped line batches actually contain this many
                    // lines.
                    SplitSlicePosition::Before => {
                        n_rows_skipped = n_rows_skipped.saturating_add(n_lines);
                        continue;
                    },
                    SplitSlicePosition::Overlapping(offset, len) => (offset, len),
                    SplitSlicePosition::After => {
                        if needs_full_row_count {
                            // If we need to know the unrestricted row count, we need
                            // to go until the end.
                            SLICE_ENDED
                        } else {
                            break;
                        }
                    },
                }
            } else {
                NO_SLICE
            };

            morsel_seq = morsel_seq.successor();

            let batch = LineBatch {
                mem_slice: batch_slice,
                n_lines,
                slice,
                row_offset,
                morsel_seq,
            };

            if line_batch_tx.send(batch).await.is_err() {
                break;
            }

            if is_eof {
                break;
            }

            if read_size < ByteSourceReader::<ReaderSource>::ideal_read_size() {
                read_size *= 4;
            }
        }

        Ok(n_rows_skipped)
    }
}
