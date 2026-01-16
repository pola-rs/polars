use polars_core::config;
use polars_error::PolarsResult;
use polars_utils::idx_mapper::IdxMapper;
use polars_utils::mmap::MemSlice;

use super::line_batch_processor::LineBatch;
use crate::async_primitives::distributor_channel;

const LF: u8 = b'\n';

pub(super) struct LineBatchDistributor {
    pub(super) global_bytes: MemSlice,
    pub(super) chunk_size: usize,
    pub(super) reverse: bool,
    pub(super) row_skipper: RowSkipper,
    pub(super) line_batch_distribute_tx: distributor_channel::Sender<LineBatch>,
}

impl LineBatchDistributor {
    /// Returns the number of rows skipped (i.e. were not sent to LineBatchProcessors).
    pub(super) async fn run(self) -> PolarsResult<usize> {
        let LineBatchDistributor {
            global_bytes: global_bytes_mem_slice,
            chunk_size,
            reverse,
            mut row_skipper,
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
                global_bytes.len(): {}, \
                chunk_size: {}, \
                n_chunks: {}, \
                n_rows_to_skip: {}, \
                reverse: {} \
                ",
                global_bytes.len(),
                chunk_size,
                n_chunks,
                row_skipper.cfg_n_rows_to_skip,
                reverse
            )
        }

        // The logic below processes in fixed chunks with remainder handling so that in the future
        // we can handle receiving data in a batched manner.

        let mut prev_remainder: &'static [u8] = &[];

        let global_idx_map = IdxMapper::new(global_bytes.len(), reverse);

        for chunk_idx in 0..n_chunks {
            let offset = chunk_idx.saturating_mul(chunk_size);
            let range = offset..offset.saturating_add(chunk_size).min(global_bytes.len());
            let range = global_idx_map.map_range(range);

            let chunk = &global_bytes[range];
            let mut remainder_combines_to_full_chunk = false;

            // Split off the chunk occurring after the last newline char.
            let chunk_remainder = if chunk_idx == n_chunks - 1 {
                // Last chunk, send everything.
                &[]
            } else if reverse {
                // Remainder is on the left because we are parsing lines in reverse:
                // N = '\n'
                // chunk:     ---N---------
                // remainder: ---N
                let eol_idx = chunk.iter().position(|c| *c == LF);

                remainder_combines_to_full_chunk = eol_idx.is_some() && !prev_remainder.is_empty();

                let end_idx = eol_idx.map_or(chunk.len(), |i| 1 + i);

                &chunk[..end_idx]
            } else {
                // N = '\n'
                // chunk:     ---------N---
                // remainder:           ---
                let start_idx = chunk.iter().rposition(|c| *c == LF).map_or(0, |i| 1 + i);

                &chunk[start_idx..]
            };

            // Holds a complete set of lines (remainder trimmed above).
            let full_lines_chunk = &chunk[IdxMapper::new(chunk.len(), reverse)
                .map_range(0..chunk.len() - chunk_remainder.len())];

            // Ensure LF is not split from the line it originates from.
            // This is important for `scan_lines` as it does not ignore empty lines.
            if reverse {
                debug_assert!(
                    chunk_remainder.ends_with(&[LF])
                        || chunk_remainder.is_empty()
                        || full_lines_chunk.is_empty()
                );
            } else {
                debug_assert!(
                    full_lines_chunk.ends_with(&[LF])
                        || full_lines_chunk.is_empty()
                        || chunk_idx == n_chunks - 1
                )
            }

            if !full_lines_chunk.is_empty() || remainder_combines_to_full_chunk {
                let mut full_lines_chunk = if prev_remainder.is_empty() {
                    full_lines_chunk
                } else if full_lines_chunk.is_empty() {
                    prev_remainder
                } else if reverse {
                    unsafe { merge_adjacent_non_empty_slices(full_lines_chunk, prev_remainder) }
                } else {
                    unsafe { merge_adjacent_non_empty_slices(prev_remainder, full_lines_chunk) }
                };

                prev_remainder = &[];
                row_skipper.skip_rows(&mut full_lines_chunk);

                if !full_lines_chunk.is_empty()
                    && line_batch_distribute_tx
                        .send(LineBatch {
                            bytes: full_lines_chunk,
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

        Ok(row_skipper.n_rows_skipped)
    }
}

pub(super) struct RowSkipper {
    /// Configured number of rows to skip. This must not be mutated during runtime.
    pub(super) cfg_n_rows_to_skip: usize,
    /// Number of rows skipped so far.
    pub(super) n_rows_skipped: usize,
    pub(super) reverse: bool,
    /// Empty / whitespace lines are not counted for ndjson, but are counted for scan_lines.
    pub(super) is_line: fn(&[u8]) -> bool,
}

impl RowSkipper {
    #[inline]
    fn remaining_rows_to_skip(&self) -> usize {
        self.cfg_n_rows_to_skip - self.n_rows_skipped
    }

    fn skip_rows(&mut self, chunk: &mut &[u8]) {
        if self.remaining_rows_to_skip() == 0 {
            return;
        }

        if self.reverse {
            self._skip_reversed(chunk)
        } else {
            // strip_suffix: E.g. ['\n'].split(c == '\n') will yield 2 empty slices (from LHS/RHS of the
            // newline). But we don't want to consider the slice from the RHS of the '\n'.
            let mut iter = if chunk.last() == Some(&LF) {
                &chunk[..chunk.len() - 1]
            } else {
                *chunk
            }
            .split(|byte| *byte == LF)
            .filter(|line| (self.is_line)(line));
            let n_skipped = iter.by_ref().take(self.remaining_rows_to_skip()).count();
            self.n_rows_skipped += n_skipped;

            *chunk = if let Some(line) = iter.next() {
                // chunk ----\n---\n---
                // line        ---
                // out         --------
                assert!(chunk.as_ptr_range().contains(&line.as_ptr()));
                unsafe {
                    let len = chunk.len() - line.as_ptr().offset_from_unsigned(chunk.as_ptr());
                    std::slice::from_raw_parts(line.as_ptr(), len)
                }
            } else {
                &[]
            }
        }
    }

    /// Skip lines in reverse (right to left).
    fn _skip_reversed(&mut self, chunk: &mut &[u8]) {
        let mut iter = if chunk.last() == Some(&LF) {
            &chunk[..chunk.len() - 1]
        } else {
            *chunk
        }
        .rsplit(|&c| c == LF)
        .filter(|line| (self.is_line)(line));

        let n_skipped = iter.by_ref().take(self.remaining_rows_to_skip()).count();
        self.n_rows_skipped += n_skipped;

        *chunk = if let Some(line) = iter.next() {
            // chunk ----\n---\n---
            // line        ---
            // out    --------
            let end_ptr: *const u8 = unsafe { line.as_ptr().add(line.len()) };
            assert!(chunk.as_ptr_range().contains(&end_ptr));
            unsafe {
                std::slice::from_raw_parts(
                    chunk.as_ptr(),
                    end_ptr.offset_from_unsigned(chunk.as_ptr()),
                )
            }
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
