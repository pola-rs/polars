use polars_core::config;
use polars_error::PolarsResult;
use polars_io::prelude::json_lines;
use polars_utils::idx_mapper::IdxMapper;
use polars_utils::mmap::MemSlice;

use super::line_batch_processor::LineBatch;
use crate::async_primitives::distributor_channel;

pub(super) struct LineBatchDistributor {
    pub(super) global_bytes: MemSlice,
    pub(super) chunk_size: usize,
    pub(super) n_rows_to_skip: usize,
    pub(super) reverse: bool,
    pub(super) line_batch_distribute_tx: distributor_channel::Sender<LineBatch>,
}

impl LineBatchDistributor {
    /// Returns the number of rows skipped (i.e. were not sent to LineBatchProcessors).
    pub(super) async fn run(self) -> PolarsResult<usize> {
        let LineBatchDistributor {
            global_bytes: global_bytes_mem_slice,
            chunk_size,
            n_rows_to_skip,
            reverse,
            mut line_batch_distribute_tx,
        } = self;

        // Safety: All receivers (LineBatchProcessors) hold a MemSlice ref to this.
        let global_bytes: &'static [u8] =
            unsafe { std::mem::transmute(global_bytes_mem_slice.as_ref()) };
        let n_chunks = global_bytes.len().div_ceil(chunk_size);
        let verbose = config::verbose();

        if verbose {
            eprintln!(
                "\
                [NDJSON LineBatchDistributor]: \
                global_bytes.len(): {} \
                chunk_size: {} \
                n_chunks: {} \
                n_rows_to_skip: {} \
                reverse: {} \
                ",
                global_bytes.len(),
                chunk_size,
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

        for chunk_idx in 0..n_chunks {
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
                &chunk[..chunk.split(|&c| c == b'\n').next().unwrap().len()]
            } else {
                // chunk:     ---------\n---
                // remainder:            ---
                &chunk[chunk.len() - chunk.rsplit(|&c| c == b'\n').next().unwrap().len()..]
            };

            let n_chars_without_remainder = chunk.len() - chunk_remainder.len();

            if n_chars_without_remainder > 0 {
                let range = 0..n_chars_without_remainder;
                let range = IdxMapper::new(chunk.len(), reverse).map_range(range);

                let full_chunk = &chunk[range];

                let mut full_chunk = if prev_remainder.is_empty() {
                    full_chunk
                } else if reverse {
                    unsafe { merge_adjacent_non_empty_slices(full_chunk, prev_remainder) }
                } else {
                    unsafe { merge_adjacent_non_empty_slices(prev_remainder, full_chunk) }
                };

                prev_remainder = &[];
                row_skipper.skip_rows(&mut full_chunk);

                if !full_chunk.is_empty()
                    && line_batch_distribute_tx
                        .send(LineBatch {
                            bytes: full_chunk,
                            chunk_idx,
                        })
                        .await
                        .is_err()
                {
                    break;
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
        let mut iter = chunk.rsplit(|&c| c == b'\n').filter(|&bytes| {
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
