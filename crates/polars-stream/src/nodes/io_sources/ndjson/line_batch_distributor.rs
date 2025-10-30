use polars_core::config;
use polars_error::{PolarsResult, polars_bail};
use polars_io::prelude::json_lines;
use polars_utils::idx_mapper::IdxMapper;
use polars_utils::mmap::MemSlice;

use super::line_batch_processor::LineBatch;
use crate::async_primitives::distributor_channel;

const EOL_CHAR: u8 = b'\n';

pub(super) struct LineBatchDistributor {
    pub(super) global_bytes: MemSlice,
    pub(super) chunk_size: usize,
    pub(super) max_chunk_size: usize,
    pub(super) n_rows_to_skip: usize,
    pub(super) reverse: bool,
    pub(super) line_batch_distribute_tx: distributor_channel::Sender<LineBatch>,
}

impl LineBatchDistributor {
    /// Returns the number of rows skipped (i.e. were not sent to LineBatchProcessors).
    pub(super) async fn run(self) -> PolarsResult<usize> {
        let LineBatchDistributor {
            global_bytes: global_bytes_buffer,
            chunk_size,
            max_chunk_size,
            n_rows_to_skip,
            reverse,
            mut line_batch_distribute_tx,
        } = self;

        // Safety: All receivers (LineBatchProcessors) hold a ref to global_bytes_buffer.
        let global_bytes: &'static [u8] =
            unsafe { std::mem::transmute(global_bytes_buffer.as_ref()) };
        let n_chunks = global_bytes.len().div_ceil(chunk_size);
        let verbose = config::verbose();

        let max_chunk_size =
            std::env::var("POLARS_FORCE_NDJSON_MAX_CHUNK_SIZE").map_or(max_chunk_size, |x| {
                x.parse::<usize>()
                    .expect("expected `POLARS_FORCE_NDJSON_MAX_CHUNK_SIZE` to be an integer")
            });

        if verbose {
            eprintln!(
                "\
                [NDJSON LineBatchDistributor]: \
                global_bytes.len(): {}, \
                chunk_size: {}, \
                max_chunk_size: {}, \
                n_chunks: {}, \
                n_rows_to_skip: {}, \
                reverse: {} \
                ",
                global_bytes.len(),
                chunk_size,
                max_chunk_size,
                n_chunks,
                n_rows_to_skip,
                reverse
            )
        }

        // The logic below processes in fixed chunks with remainder handling so that in the future
        // we can handle receiving data in a batched manner.

        let mut prev_remainder: &'static [u8] = &[];

        let global_idx_map = IdxMapper::new(global_bytes.len(), reverse);

        let mut row_skipper = RowSkipper {
            remaining_rows_to_skip: n_rows_to_skip,
            reverse,
        };

        let mut morsel_idx: u64 = 0;

        'chunk_loop: for chunk_idx in 0..n_chunks {
            let offset = chunk_idx.saturating_mul(chunk_size);
            let range = offset..offset.saturating_add(chunk_size).min(global_bytes.len());
            let range = global_idx_map.map_range(range);

            let chunk = &global_bytes[range];

            // Split off the chunk occurring after the last newline char.
            let chunk_remainder = if chunk_idx == n_chunks - 1 {
                // Last chunk, send everything.
                &[]
            } else if reverse {
                // Remainder is on the left because we are parsing lines in reverse:
                // chunk:     ---\n---------
                // remainder: ---
                &chunk[..chunk.split(|&c| c == EOL_CHAR).next().unwrap().len()]
            } else {
                // chunk:     ---------\n---
                // remainder:            ---
                &chunk[chunk.len() - chunk.rsplit(|&c| c == EOL_CHAR).next().unwrap().len()..]
            };

            let n_chars_without_remainder = chunk.len() - chunk_remainder.len();

            if n_chars_without_remainder > 0 {
                let range = 0..n_chars_without_remainder;
                let range = IdxMapper::new(chunk.len(), reverse).map_range(range);

                // Holds a complete set of lines (remainder trimmed above).
                let lines_chunk = &chunk[range];

                let mut lines_chunk = if prev_remainder.is_empty() {
                    lines_chunk
                } else if reverse {
                    unsafe { merge_adjacent_non_empty_slices(lines_chunk, prev_remainder) }
                } else {
                    unsafe { merge_adjacent_non_empty_slices(prev_remainder, lines_chunk) }
                };

                prev_remainder = &[];
                row_skipper.skip_rows(&mut lines_chunk);

                while !lines_chunk.is_empty() {
                    let next_chunk: &[u8];

                    if lines_chunk.len() <= max_chunk_size {
                        next_chunk = std::mem::take(&mut lines_chunk);
                    } else {
                        let split_idx = max_chunk_size
                            - lines_chunk[..max_chunk_size]
                                .rsplit(|c| *c == EOL_CHAR)
                                .next()
                                .unwrap()
                                .len();

                        if split_idx == 0 {
                            let line_length = max_chunk_size
                                + lines_chunk[max_chunk_size..]
                                    .split(|c| *c == EOL_CHAR)
                                    .next()
                                    .unwrap()
                                    .len();
                            polars_bail!(
                                ComputeError:
                                "line byte length of {} exceeded max chunk size of {}",
                                line_length, max_chunk_size
                            )
                        }

                        (next_chunk, lines_chunk) = lines_chunk.split_at(split_idx);
                    }

                    if line_batch_distribute_tx
                        .send(LineBatch {
                            bytes: next_chunk,
                            morsel_idx,
                        })
                        .await
                        .is_err()
                    {
                        break 'chunk_loop;
                    }

                    morsel_idx = morsel_idx.saturating_add(1);
                }
            }

            // Note: If `prev_remainder` is non-empty at this point, it means the entire current
            // chunk does not contain a newline.
            prev_remainder = if prev_remainder.is_empty() {
                chunk_remainder
            } else if reverse {
                // Current chunk comes before the previous remainder in memory when reversed.
                unsafe { merge_adjacent_non_empty_slices(chunk_remainder, prev_remainder) }
            } else {
                unsafe { merge_adjacent_non_empty_slices(prev_remainder, chunk_remainder) }
            };
        }

        if verbose {
            eprintln!("[NDJSON LineBatchDistributor]: returning");
        }

        let n_rows_skipped = n_rows_to_skip - row_skipper.remaining_rows_to_skip;

        Ok(n_rows_skipped)
    }
}

struct RowSkipper {
    remaining_rows_to_skip: usize,
    reverse: bool,
}

impl RowSkipper {
    fn skip_rows(&mut self, chunk: &mut &[u8]) {
        if self.remaining_rows_to_skip == 0 {
            return;
        }

        if self.reverse {
            self._skip_reversed(chunk)
        } else {
            let mut iter = json_lines(chunk);
            let n_skipped = (&mut iter).take(self.remaining_rows_to_skip).count();
            self.remaining_rows_to_skip -= n_skipped;

            *chunk = if let Some(line) = iter.next() {
                // chunk --------------
                // line        ---
                // out         --------
                let chunk_end_addr = chunk.as_ptr() as usize + chunk.len();

                debug_assert!(
                    (chunk.as_ptr() as usize..chunk_end_addr).contains(&(line.as_ptr() as usize))
                );

                let truncated_len = chunk_end_addr - line.as_ptr() as usize;

                unsafe { std::slice::from_raw_parts(line.as_ptr(), truncated_len) }
            } else {
                &[]
            }
        }
    }

    /// Skip lines in reverse (right to left).
    fn _skip_reversed(&mut self, chunk: &mut &[u8]) {
        // Note: This is `rsplit`
        let mut iter = chunk.rsplit(|&c| c == EOL_CHAR).filter(|&bytes| {
            bytes
                .iter()
                .any(|&byte| !matches!(byte, b' ' | b'\t' | b'\r'))
        });

        let n_skipped = (&mut iter).take(self.remaining_rows_to_skip).count();
        self.remaining_rows_to_skip -= n_skipped;

        *chunk = if let Some(line) = iter.next() {
            // chunk --------------
            // line        ---
            // out    --------
            let line_end_addr = line.as_ptr() as usize + line.len();
            let truncated_len = line_end_addr - chunk.as_ptr() as usize;

            debug_assert!(truncated_len <= chunk.len());

            unsafe { std::slice::from_raw_parts(chunk.as_ptr(), truncated_len) }
        } else {
            &[]
        }
    }
}

/// # Safety
/// `left` and `right` should be non-empty and `right` should be immediately  after `left` in memory.
/// The slice resulting from combining them should adhere to all safety preconditions of [`std::slice::from_raw_parts`].
unsafe fn merge_adjacent_non_empty_slices<'a>(left: &'a [u8], right: &'a [u8]) -> &'a [u8] {
    assert!(!left.is_empty());
    assert!(!right.is_empty());
    assert_eq!(left.as_ptr() as usize + left.len(), right.as_ptr() as usize);
    unsafe { std::slice::from_raw_parts(left.as_ptr(), left.len() + right.len()) }
}
