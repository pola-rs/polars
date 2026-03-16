use std::cmp;
use std::num::NonZeroUsize;

use polars_buffer::Buffer;
use polars_core::config;
use polars_error::PolarsResult;
use polars_io::utils::compression::ByteSourceReader;
use polars_io::utils::stream_buf_reader::ReaderSource;
use polars_utils::mem::prefetch::prefetch_l2;

use super::line_batch_processor::LineBatch;
use crate::async_primitives::distributor_channel;

const LF: u8 = b'\n';

pub(super) struct LineBatchDistributor {
    pub(super) reader: ByteSourceReader<ReaderSource>,
    pub(super) reverse: bool,
    pub(super) row_skipper: RowSkipper,
    pub(super) line_batch_distribute_tx: distributor_channel::Sender<LineBatch>,
    pub(super) uncompressed_file_size_hint: Option<usize>,
}

impl LineBatchDistributor {
    /// Returns the number of rows skipped (i.e. were not sent to LineBatchProcessors).
    pub(super) async fn run(self) -> PolarsResult<usize> {
        let LineBatchDistributor {
            mut reader,
            reverse,
            mut row_skipper,
            mut line_batch_distribute_tx,
            uncompressed_file_size_hint,
        } = self;

        let verbose = config::verbose();

        let fixed_read_size = std::env::var("POLARS_FORCE_NDJSON_READ_SIZE")
            .map(|x| {
                x.parse::<NonZeroUsize>().ok().unwrap_or_else(|| {
                    panic!("invalid value for POLARS_FORCE_NDJSON_READ_SIZE: {x}")
                })
            })
            .ok();

        if verbose {
            eprintln!(
                "\
                [NDJSON LineBatchDistributor]: \
                n_rows_to_skip: {}, \
                reverse: {reverse} \
                fixed_read_size: {fixed_read_size:?} \
                ",
                row_skipper.cfg_n_rows_to_skip,
            );
        }

        let mut full_input_opt = if reverse {
            // Since decompression doesn't support reverse decompression, we have to fully
            // decompress the input. It's crucial for the streaming property that this doesn't get
            // called in the non-reverse case.
            debug_assert!(
                !reader.is_compressed(),
                "Negative slicing and decompression risk OOM, should be handled on higher level."
            );
            let (full_input, _) =
                reader.read_next_slice(&Buffer::new(), usize::MAX, uncompressed_file_size_hint)?;
            let offset = full_input.len();
            Some((full_input, offset))
        } else {
            None
        };

        let mut read_size = fixed_read_size
            .map(NonZeroUsize::get)
            .unwrap_or_else(ByteSourceReader::<ReaderSource>::initial_read_size);
        let mut prev_leftover = Buffer::new();
        let mut chunk_idx = 0;

        loop {
            let (mem_slice, bytes_read) = if reverse {
                let (full_input, offset) = full_input_opt.as_mut().unwrap();
                let new_offset = offset.saturating_sub(read_size);
                let bytes_read = *offset - new_offset;
                let new_slice = full_input
                    .clone()
                    .sliced(new_offset..(*offset + prev_leftover.len()));
                *offset = new_offset;
                (new_slice, bytes_read)
            } else {
                reader.read_next_slice(
                    &prev_leftover,
                    read_size,
                    Some(prev_leftover.len() + read_size),
                )?
            };

            if mem_slice.is_empty() {
                break;
            }

            prefetch_l2(&mem_slice);

            let is_eof = bytes_read == 0;
            let (unconsumed_offset, done) = process_chunk(
                mem_slice.clone(),
                is_eof,
                reverse,
                &mut chunk_idx,
                &mut row_skipper,
                &mut line_batch_distribute_tx,
            )
            .await;

            if done || is_eof {
                break;
            }

            if let Some(offset) = unconsumed_offset {
                prev_leftover = if reverse {
                    mem_slice.sliced(..offset)
                } else {
                    mem_slice.sliced(offset..)
                };
            } else {
                if fixed_read_size.is_none() {
                    // This allows the slice to grow until at least a single row is included. To avoid a quadratic run-time for large row sizes, we double the read size.
                    read_size = read_size.saturating_mul(2);
                }
                prev_leftover = mem_slice;
                continue;
            }

            if read_size < ByteSourceReader::<ReaderSource>::ideal_read_size()
                && fixed_read_size.is_none()
            {
                read_size *= 4;
            }
        }

        if verbose {
            eprintln!("[NDJSON LineBatchDistributor]: returning");
        }

        Ok(row_skipper.n_rows_skipped)
    }
}

async fn process_chunk(
    chunk: Buffer<u8>,
    is_eof: bool,
    reverse: bool,
    chunk_idx: &mut usize,
    row_skipper: &mut RowSkipper,
    line_batch_distribute_tx: &mut distributor_channel::Sender<LineBatch>,
) -> (Option<usize>, bool) {
    let len = chunk.len();
    if len == 0 {
        return (None, is_eof);
    }

    let unconsumed_offset = if is_eof {
        Some(0)
    } else if reverse {
        memchr::memchr(LF, &chunk)
    } else {
        memchr::memrchr(LF, &chunk)
    }
    .map(|offset| cmp::min(offset + 1, len));

    let mut done = false;
    if let Some(offset) = unconsumed_offset {
        let line_chunk = if is_eof {
            // Consume full input in EOF case.
            chunk
        } else if reverse {
            chunk.sliced(offset..)
        } else {
            chunk.sliced(..offset)
        };

        // Since this path is only executed if at least one line is found or EOF, we guarantee that
        // `skip_rows` will always make progress.
        let batch_chunk = row_skipper.skip_rows(line_chunk);

        if !batch_chunk.is_empty() {
            let batch = LineBatch {
                bytes: batch_chunk,
                chunk_idx: *chunk_idx,
            };
            done = line_batch_distribute_tx.send(batch).await.is_err();
            *chunk_idx += 1;
        }
    }

    (unconsumed_offset, done)
}

pub(super) struct RowSkipper {
    /// Configured number of rows to skip. This MUST NOT be mutated during runtime.
    pub(super) cfg_n_rows_to_skip: usize,
    /// Number of rows skipped so far.
    pub(super) n_rows_skipped: usize,
    pub(super) reverse: bool,
    /// Empty / whitespace lines are not counted for ndjson, but are counted for scan_lines.
    pub(super) is_line: fn(&[u8]) -> bool,
}

impl RowSkipper {
    /// Takes `chunk` which contains N lines and consumes min(N, M) lines, where M is the number of
    /// lines that have not yet been skipped.
    ///
    /// `chunk` is expected in the form "line_a\nline_b\nline_c(\n)?" regardless of `reverse`.
    fn skip_rows(&mut self, chunk: Buffer<u8>) -> Buffer<u8> {
        if self.n_rows_skipped >= self.cfg_n_rows_to_skip {
            return chunk;
        }

        if self.reverse {
            self._skip_rows_backward(chunk)
        } else {
            self._skip_rows_forward(chunk)
        }
    }

    fn _skip_rows_forward(&mut self, chunk: Buffer<u8>) -> Buffer<u8> {
        let len = chunk.len();
        let mut offset = 0;

        while let Some(pos) = memchr::memchr(LF, &chunk[offset..]) {
            let prev_offset = offset;
            offset = cmp::min(offset + pos + 1, len);

            if !(self.is_line)(&chunk[prev_offset..offset]) {
                continue;
            }

            self.n_rows_skipped += 1;

            if self.n_rows_skipped >= self.cfg_n_rows_to_skip {
                return chunk.sliced(offset..len);
            }
        }

        Buffer::new()
    }

    fn _skip_rows_backward(&mut self, chunk: Buffer<u8>) -> Buffer<u8> {
        let len = chunk.len();
        let mut offset = len.saturating_sub((chunk.last().copied() == Some(LF)) as usize);

        while let Some(pos) = memchr::memrchr(LF, &chunk[..offset]) {
            let prev_offset = offset;
            offset = pos;

            if !(self.is_line)(&chunk[offset..prev_offset]) {
                continue;
            }

            self.n_rows_skipped += 1;

            if self.n_rows_skipped >= self.cfg_n_rows_to_skip {
                return chunk.sliced(0..offset);
            }

            offset = offset.saturating_sub(1);
        }

        self.n_rows_skipped += !chunk.is_empty() as usize;

        Buffer::new()
    }
}
