use std::cmp;
use std::num::NonZeroUsize;

use polars_buffer::Buffer;
use polars_error::PolarsResult;
use polars_io::pl_async;
use polars_io::utils::compression::{ByteSourceReader, SupportedCompression};
use polars_io::utils::stream_buf_reader::ReaderSource;
use polars_utils::mem::prefetch::prefetch_l2;

use super::line_batch_processor::LineBatch;
use crate::async_primitives::distributor_channel;
use crate::utils::tokio_handle_ext;

const LF: u8 = b'\n';

pub(super) struct LineBatchDistributor {
    pub(super) reader: ReaderSource,
    pub(super) reverse: bool,
    pub(super) row_skipper: RowSkipper,
    pub(super) line_batch_distribute_tx: distributor_channel::Sender<LineBatch>,
    pub(super) compression: Option<SupportedCompression>,
    pub(super) uncompressed_file_size_hint: Option<usize>,
    pub(super) use_async_prefetch: bool,
    pub(super) verbose: bool,
}

impl LineBatchDistributor {
    /// Returns the number of rows skipped (i.e. were not sent to LineBatchProcessors).
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
            eprintln!("[NDJsonFileReader]: Start line batch distributor direct");
        }

        let reader = ByteSourceReader::try_new(self.reader, self.compression)?;
        let mut line_batch_tx = self.line_batch_distribute_tx;
        let use_prefetch_l2 = true;

        let mut producer = LineBatchProducer::new(
            reader,
            self.reverse,
            self.row_skipper,
            self.uncompressed_file_size_hint,
            use_prefetch_l2,
            self.verbose,
        )?;

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
        let LineBatchDistributor {
            reader: reader_source,
            reverse,
            row_skipper,
            line_batch_distribute_tx: mut line_batch_tx,
            compression,
            uncompressed_file_size_hint,
            use_async_prefetch: _,
            verbose,
        } = self;

        let read_loop_handle = tokio_handle_ext::AbortOnDropHandle(
            pl_async::get_runtime().spawn_blocking(move || {
                let handle = tokio::runtime::Handle::current();
                if verbose {
                    eprintln!("[NDJsonFileReader]: Start line batch distributor async");
                }

                let reader = ByteSourceReader::try_new(reader_source, compression)?;
                let use_prefetch_l2 = false;

                let mut producer = LineBatchProducer::new(
                    reader,
                    reverse,
                    row_skipper,
                    uncompressed_file_size_hint,
                    use_prefetch_l2,
                    verbose,
                )?;

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

/// Produces LineBatches from a ByteSourceReader.
struct LineBatchProducer {
    reader: ByteSourceReader<ReaderSource>,
    reverse: bool,
    row_skipper: RowSkipper,
    use_prefetch_l2: bool,
    fixed_read_size: Option<NonZeroUsize>,
    full_input_opt: Option<(Buffer<u8>, usize)>,
    prev_leftover: Buffer<u8>,
    read_size: usize,
    chunk_idx: usize,
    finished: bool,
}

impl LineBatchProducer {
    fn new(
        mut reader: ByteSourceReader<ReaderSource>,
        reverse: bool,
        row_skipper: RowSkipper,
        uncompressed_file_size_hint: Option<usize>,
        use_prefetch_l2: bool,
        verbose: bool,
    ) -> PolarsResult<Self> {
        let fixed_read_size = std::env::var("POLARS_FORCE_NDJSON_READ_SIZE")
            .map(|x| {
                x.parse::<NonZeroUsize>().ok().unwrap_or_else(|| {
                    panic!("invalid value for POLARS_FORCE_NDJSON_READ_SIZE: {x}")
                })
            })
            .ok();

        let read_size = fixed_read_size
            .map(NonZeroUsize::get)
            .unwrap_or_else(ByteSourceReader::<ReaderSource>::initial_read_size);

        if verbose {
            eprintln!(
                "[NDJson LineBatchDistributor]: \
                n_rows_to_skip: {}, \
                reverse: {reverse}, \
                fixed_read_size: {fixed_read_size:?}",
                row_skipper.cfg_n_rows_to_skip,
            );
        }

        let full_input_opt = if reverse {
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

        Ok(Self {
            reader,
            reverse,
            row_skipper,
            use_prefetch_l2,
            fixed_read_size,
            full_input_opt,
            prev_leftover: Buffer::new(),
            read_size,
            chunk_idx: 0,
            finished: false,
        })
    }

    fn next_batch(&mut self) -> PolarsResult<Option<LineBatch>> {
        if self.finished {
            return Ok(None);
        }

        loop {
            let (mem_slice, bytes_read) = if self.reverse {
                let (full_input, offset) = self.full_input_opt.as_mut().unwrap();
                let new_offset = offset.saturating_sub(self.read_size);
                let bytes_read = *offset - new_offset;
                let new_slice = full_input
                    .clone()
                    .sliced(new_offset..(*offset + self.prev_leftover.len()));
                *offset = new_offset;
                (new_slice, bytes_read)
            } else {
                self.reader.read_next_slice(
                    &self.prev_leftover,
                    self.read_size,
                    Some(self.prev_leftover.len() + self.read_size),
                )?
            };

            if mem_slice.is_empty() {
                self.finished = true;
                return Ok(None);
            }

            if self.use_prefetch_l2 {
                prefetch_l2(&mem_slice);
            }

            let is_eof = bytes_read == 0;
            let (batch, unconsumed_offset) = process_chunk(
                mem_slice.clone(),
                is_eof,
                self.reverse,
                &mut self.chunk_idx,
                &mut self.row_skipper,
            );

            if is_eof {
                self.finished = true;
                return Ok(batch);
            }

            if let Some(offset) = unconsumed_offset {
                self.prev_leftover = if self.reverse {
                    mem_slice.sliced(..offset)
                } else {
                    mem_slice.sliced(offset..)
                };
            } else {
                if self.fixed_read_size.is_none() {
                    self.read_size = self.read_size.saturating_mul(2);
                }
                self.prev_leftover = mem_slice;
                continue;
            }

            if self.read_size < ByteSourceReader::<ReaderSource>::ideal_read_size()
                && self.fixed_read_size.is_none()
            {
                self.read_size *= 4;
            }

            if batch.is_some() {
                return Ok(batch);
            }
        }
    }

    fn n_rows_skipped(&self) -> usize {
        self.row_skipper.n_rows_skipped
    }
}

/// Processes a raw buffer and returns a newline-aligned batch and the matching offset.
fn process_chunk(
    chunk: Buffer<u8>,
    is_eof: bool,
    reverse: bool,
    chunk_idx: &mut usize,
    row_skipper: &mut RowSkipper,
) -> (Option<LineBatch>, Option<usize>) {
    let len = chunk.len();
    if len == 0 {
        return (None, None);
    }

    let unconsumed_offset = if is_eof {
        Some(0)
    } else if reverse {
        memchr::memchr(LF, &chunk)
    } else {
        memchr::memrchr(LF, &chunk)
    }
    .map(|offset| cmp::min(offset + 1, len));

    let batch = if let Some(offset) = unconsumed_offset {
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
            *chunk_idx += 1;
            Some(batch)
        } else {
            None
        }
    } else {
        None
    };

    (batch, unconsumed_offset)
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
