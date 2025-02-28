use core::json_lines;
use std::num::NonZeroUsize;

use arrow::array::StructArray;
use polars_core::prelude::*;
use polars_core::POOL;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub(crate) mod buffer;
pub mod core;

pub fn infer_schema<R: std::io::BufRead>(
    reader: &mut R,
    infer_schema_len: Option<NonZeroUsize>,
) -> PolarsResult<Schema> {
    let dtypes = polars_json::ndjson::iter_unique_dtypes(reader, infer_schema_len)?;
    let dtype =
        crate::json::infer::dtypes_to_supertype(dtypes.map(|dt| DataType::from_arrow_dtype(&dt)))?;
    let schema = StructArray::get_fields(&dtype.to_arrow(CompatLevel::newest()))
        .iter()
        .map(Into::<Field>::into)
        .collect();
    Ok(schema)
}

/// Statistics for a chunk of text used for NDJSON parsing.
#[derive(Debug, Clone, PartialEq)]
struct ChunkStats {
    non_empty_rows: usize,
    /// Set to None if the chunk was empty.
    last_newline_offset: Option<usize>,
    /// Used when counting rows.
    has_leading_empty_line: bool,
    has_non_empty_remainder: bool,
}

impl ChunkStats {
    /// Assumes that:
    /// * There is no quoting of newlines characters (unlike CSV)
    /// * We do not count empty lines (successive newlines, or lines containing only whitespace / tab)
    fn from_chunk(chunk: &[u8]) -> Self {
        // Notes: Offsets are right-to-left in reverse mode.
        let first_newline_offset = memchr::memchr(b'\n', chunk);
        let last_newline_offset = memchr::memrchr(b'\n', chunk);

        let has_leading_empty_line =
            first_newline_offset.is_some_and(|i| json_lines(&chunk[..i]).next().is_none());
        let has_non_empty_remainder =
            json_lines(&chunk[last_newline_offset.map_or(0, |i| 1 + i)..chunk.len()])
                .next()
                .is_some();

        let mut non_empty_rows = if first_newline_offset.is_some() && !has_leading_empty_line {
            1
        } else {
            0
        };

        if first_newline_offset.is_some() {
            let range = first_newline_offset.unwrap() + 1..last_newline_offset.unwrap() + 1;
            non_empty_rows += json_lines(&chunk[range]).count()
        }

        Self {
            non_empty_rows,
            has_leading_empty_line,
            last_newline_offset,
            has_non_empty_remainder,
        }
    }

    /// Reduction state for counting rows.
    ///
    /// Note: `rhs` should be from the chunk immediately after `slf`, otherwise the results will be
    /// incorrect.
    pub fn reduce_count_rows(slf: &Self, rhs: &Self) -> Self {
        let mut non_empty_rows = slf.non_empty_rows + rhs.non_empty_rows;

        if slf.has_non_empty_remainder && rhs.has_leading_empty_line {
            non_empty_rows += 1;
        }

        ChunkStats {
            non_empty_rows,
            last_newline_offset: rhs.last_newline_offset,
            has_leading_empty_line: slf.has_leading_empty_line,
            has_non_empty_remainder: rhs.has_non_empty_remainder
                || (rhs.last_newline_offset.is_none() && slf.has_non_empty_remainder),
        }
    }

    /// The non-empty row count of this chunk assuming it is the last chunk (adds 1 if there is a
    /// non-empty remainder).
    pub fn non_empty_row_count_as_last_chunk(&self) -> usize {
        self.non_empty_rows + self.has_non_empty_remainder as usize
    }
}

/// Count the number of rows. The slice passed must represent the entire file. This will
/// potentially parallelize using rayon.
///
/// This does not check if the lines are valid NDJSON - it assumes that is the case.
pub fn count_rows_par(full_bytes: &[u8]) -> usize {
    _count_rows_impl(
        full_bytes,
        std::env::var("POLARS_FORCE_NDJSON_CHUNK_SIZE")
            .ok()
            .and_then(|x| x.parse::<usize>().ok()),
    )
}

/// Count the number of rows. The slice passed must represent the entire file.
/// This does not check if the lines are valid NDJSON - it assumes that is the case.
pub fn count_rows(full_bytes: &[u8]) -> usize {
    json_lines(full_bytes).count()
}

/// This is separate for testing purposes.
fn _count_rows_impl(full_bytes: &[u8], force_chunk_size: Option<usize>) -> usize {
    let min_chunk_size = if cfg!(debug_assertions) { 0 } else { 16 * 1024 };

    // Row count does not have a parsing dependency between threads, so we can just split into
    // the same number of chunks as threads.
    let chunk_size = force_chunk_size.unwrap_or(
        full_bytes
            .len()
            .div_ceil(POOL.current_num_threads())
            .max(min_chunk_size),
    );

    if full_bytes.is_empty() {
        return 0;
    }

    let n_chunks = full_bytes.len().div_ceil(chunk_size);

    if n_chunks > 1 {
        let identity = ChunkStats::from_chunk(&[]);
        let acc_stats = POOL.install(|| {
            (0..n_chunks)
                .into_par_iter()
                .map(|i| {
                    ChunkStats::from_chunk(
                        &full_bytes[i * chunk_size
                            ..(1 + i).saturating_mul(chunk_size).min(full_bytes.len())],
                    )
                })
                .reduce(
                    || identity.clone(),
                    |l, r| ChunkStats::reduce_count_rows(&l, &r),
                )
        });

        acc_stats.non_empty_row_count_as_last_chunk()
    } else {
        count_rows(full_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::ChunkStats;

    #[test]
    fn test_chunk_stats() {
        let bytes = r#"
{"a": 1}
{"a": 2}
"#
        .as_bytes();

        assert_eq!(
            ChunkStats::from_chunk(bytes),
            ChunkStats {
                non_empty_rows: 2,
                last_newline_offset: Some(18),
                has_leading_empty_line: true,
                has_non_empty_remainder: false,
            }
        );

        assert_eq!(
            ChunkStats::from_chunk(&bytes[..bytes.len() - 3]),
            ChunkStats {
                non_empty_rows: 1,
                last_newline_offset: Some(9),
                has_leading_empty_line: true,
                has_non_empty_remainder: true,
            }
        );

        assert_eq!(super::_count_rows_impl(&[], Some(1)), 0);
        assert_eq!(super::_count_rows_impl(bytes, Some(1)), 2);
        assert_eq!(super::_count_rows_impl(bytes, Some(3)), 2);
        assert_eq!(super::_count_rows_impl(bytes, Some(5)), 2);
        assert_eq!(super::_count_rows_impl(bytes, Some(7)), 2);
        assert_eq!(super::_count_rows_impl(bytes, Some(bytes.len())), 2);

        assert_eq!(super::count_rows_par(&[]), 0);

        assert_eq!(
            ChunkStats::from_chunk(&[]),
            ChunkStats {
                non_empty_rows: 0,
                last_newline_offset: None,
                has_leading_empty_line: false,
                has_non_empty_remainder: false,
            }
        );

        // Single-chars

        assert_eq!(
            ChunkStats::from_chunk(b"\n"),
            ChunkStats {
                non_empty_rows: 0,
                last_newline_offset: Some(0),
                has_leading_empty_line: true,
                has_non_empty_remainder: false,
            }
        );

        assert_eq!(
            ChunkStats::from_chunk(b"a"),
            ChunkStats {
                non_empty_rows: 0,
                last_newline_offset: None,
                has_leading_empty_line: false,
                has_non_empty_remainder: true,
            }
        );

        assert_eq!(
            ChunkStats::from_chunk(b" "),
            ChunkStats {
                non_empty_rows: 0,
                last_newline_offset: None,
                has_leading_empty_line: false,
                has_non_empty_remainder: false,
            }
        );

        // Double-char combinations

        assert_eq!(
            ChunkStats::from_chunk(b"a\n"),
            ChunkStats {
                non_empty_rows: 1,
                last_newline_offset: Some(1),
                has_leading_empty_line: false,
                has_non_empty_remainder: false,
            }
        );

        assert_eq!(
            ChunkStats::from_chunk(b" \n"),
            ChunkStats {
                non_empty_rows: 0,
                last_newline_offset: Some(1),
                has_leading_empty_line: true,
                has_non_empty_remainder: false,
            }
        );

        assert_eq!(
            ChunkStats::from_chunk(b"a "),
            ChunkStats {
                non_empty_rows: 0,
                last_newline_offset: None,
                has_leading_empty_line: false,
                has_non_empty_remainder: true,
            }
        );
    }

    #[test]
    fn test_chunk_stats_whitespace() {
        let space_char = ' ';
        let tab_char = '\t';
        // This is not valid JSON, but we simply need to test that ChunkStats only counts lines
        // containing at least 1 non-whitespace character.
        let bytes = format!(
            "
abc

abc

{tab_char}
{space_char}{space_char}{space_char}

     abc{space_char}

"
        );
        let bytes = bytes.as_bytes();

        assert_eq!(
            ChunkStats::from_chunk(bytes),
            ChunkStats {
                non_empty_rows: 3,
                last_newline_offset: Some(28),
                has_leading_empty_line: true,
                has_non_empty_remainder: false,
            }
        );
    }

    #[test]
    fn test_count_rows() {
        let bytes = r#"{"text": "\"hello", "id": 1}     
{"text": "\"hello", "id": 1}     "#
            .as_bytes();

        assert_eq!(super::count_rows_par(bytes), 2);
    }
}
