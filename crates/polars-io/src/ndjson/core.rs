use std::io::Cursor;
use std::num::NonZeroUsize;

pub use arrow::array::StructArray;
use num_traits::pow::Pow;
use polars_core::POOL;
use polars_core::prelude::*;
use polars_core::utils::accumulate_dataframes_vertical;
use rayon::prelude::*;

use crate::RowIndex;
use crate::mmap::ReaderBytes;
use crate::ndjson::buffer::*;
use crate::predicates::PhysicalIoExpr;
use crate::prelude::*;
const NEWLINE: u8 = b'\n';
const CLOSING_BRACKET: u8 = b'}';

pub(crate) struct CoreJsonReader<'a> {
    reader_bytes: Option<ReaderBytes<'a>>,
    n_rows: Option<usize>,
    schema: SchemaRef,
    n_threads: Option<usize>,
    sample_size: usize,
    chunk_size: NonZeroUsize,
    low_memory: bool,
    ignore_errors: bool,
    row_index: Option<&'a mut RowIndex>,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
    projection: Option<Arc<[PlSmallStr]>>,
}
impl<'a> CoreJsonReader<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        reader_bytes: ReaderBytes<'a>,
        n_rows: Option<usize>,
        schema: Option<SchemaRef>,
        schema_overwrite: Option<&Schema>,
        n_threads: Option<usize>,
        sample_size: usize,
        chunk_size: NonZeroUsize,
        low_memory: bool,
        infer_schema_len: Option<NonZeroUsize>,
        ignore_errors: bool,
        row_index: Option<&'a mut RowIndex>,
        predicate: Option<Arc<dyn PhysicalIoExpr>>,
        projection: Option<Arc<[PlSmallStr]>>,
    ) -> PolarsResult<CoreJsonReader<'a>> {
        let reader_bytes = reader_bytes;

        let mut schema = match schema {
            Some(schema) => schema,
            None => {
                let bytes: &[u8] = &reader_bytes;
                let mut cursor = Cursor::new(bytes);
                Arc::new(crate::ndjson::infer_schema(&mut cursor, infer_schema_len)?)
            },
        };
        if let Some(overwriting_schema) = schema_overwrite {
            let schema = Arc::make_mut(&mut schema);
            overwrite_schema(schema, overwriting_schema)?;
        }

        Ok(CoreJsonReader {
            reader_bytes: Some(reader_bytes),
            schema,
            sample_size,
            n_rows,
            n_threads,
            chunk_size,
            low_memory,
            ignore_errors,
            row_index,
            predicate,
            projection,
        })
    }

    fn parse_json(&mut self, mut n_threads: usize, bytes: &[u8]) -> PolarsResult<DataFrame> {
        let mut bytes = bytes;
        let mut total_rows = 128;

        if let Some((mean, std)) = get_line_stats_json(bytes, self.sample_size) {
            let line_length_upper_bound = mean + 1.1 * std;

            total_rows = (bytes.len() as f32 / (mean - 0.01 * std)) as usize;
            if let Some(n_rows) = self.n_rows {
                total_rows = std::cmp::min(n_rows, total_rows);
                // the guessed upper bound of  the no. of bytes in the file
                let n_bytes = (line_length_upper_bound * (n_rows as f32)) as usize;

                if n_bytes < bytes.len() {
                    if let Some(pos) = next_line_position_naive_json(&bytes[n_bytes..]) {
                        bytes = &bytes[..n_bytes + pos]
                    }
                }
            }
        }

        if total_rows <= 128 {
            n_threads = 1;
        }

        let rows_per_thread = total_rows / n_threads;

        let max_proxy = bytes.len() / n_threads / 2;
        let capacity = if self.low_memory {
            usize::from(self.chunk_size)
        } else {
            std::cmp::min(rows_per_thread, max_proxy)
        };
        let file_chunks = get_file_chunks_json(bytes, n_threads);

        let row_index = self.row_index.as_ref().map(|ri| ri as &RowIndex);
        let (mut dfs, prepredicate_heights) = POOL.install(|| {
            file_chunks
                .into_par_iter()
                .map(|(start_pos, stop_at_nbytes)| {
                    let mut local_df = parse_ndjson(
                        &bytes[start_pos..stop_at_nbytes],
                        Some(capacity),
                        &self.schema,
                        self.ignore_errors,
                    )?;

                    let prepredicate_height = local_df.height() as IdxSize;
                    if let Some(projection) = self.projection.as_deref() {
                        local_df = local_df.select(projection.iter().cloned())?;
                    }

                    if let Some(row_index) = row_index {
                        local_df = local_df
                            .with_row_index(row_index.name.clone(), Some(row_index.offset))?;
                    }

                    if let Some(predicate) = &self.predicate {
                        let s = predicate.evaluate_io(&local_df)?;
                        let mask = s.bool()?;
                        local_df = local_df.filter(mask)?;
                    }

                    Ok((local_df, prepredicate_height))
                })
                .collect::<PolarsResult<(Vec<_>, Vec<_>)>>()
        })?;

        if let Some(ref mut row_index) = self.row_index {
            update_row_counts3(&mut dfs, &prepredicate_heights, 0);
            row_index.offset += prepredicate_heights.iter().copied().sum::<IdxSize>();
        }

        accumulate_dataframes_vertical(dfs)
    }

    pub fn as_df(&mut self) -> PolarsResult<DataFrame> {
        let n_threads = self.n_threads.unwrap_or_else(|| POOL.current_num_threads());

        let reader_bytes = self.reader_bytes.take().unwrap();

        let mut df = self.parse_json(n_threads, &reader_bytes)?;

        // if multi-threaded the n_rows was probabilistically determined.
        // Let's slice to correct number of rows if possible.
        if let Some(n_rows) = self.n_rows {
            if n_rows < df.height() {
                df = df.slice(0, n_rows)
            }
        }
        Ok(df)
    }
}

#[inline(always)]
fn parse_impl(
    bytes: &[u8],
    buffers: &mut PlIndexMap<BufferKey, Buffer>,
    scratch: &mut Scratch,
    ignore_errors: bool,
) -> PolarsResult<usize> {
    scratch.json.clear();
    scratch.json.extend_from_slice(bytes);
    let n = scratch.json.len();
    let value = simd_json::to_borrowed_value_with_buffers(&mut scratch.json, &mut scratch.buffers)
        .map_err(|e| polars_err!(ComputeError: "error parsing line: {}", e))?;
    match value {
        simd_json::BorrowedValue::Object(value) => {
            buffers.iter_mut().try_for_each(|(s, inner)| {
                match s.0.map_lookup(&value) {
                    Some(v) => inner.add(v)?,
                    None => inner.add_null(),
                }
                PolarsResult::Ok(())
            })?;
        },
        _ if ignore_errors => {
            buffers.iter_mut().for_each(|(_, inner)| inner.add_null());
        },
        v => {
            polars_bail!(ComputeError: "NDJSON line expected to contain JSON object: {v}");
        },
    };
    Ok(n)
}

#[derive(Default)]
struct Scratch {
    json: Vec<u8>,
    buffers: simd_json::Buffers,
}

pub fn json_lines(bytes: &[u8]) -> impl Iterator<Item = &[u8]> {
    // This previously used `serde_json`'s `RawValue` to deserialize chunks without really deserializing them.
    // However, this convenience comes at a cost. serde_json allocates and parses and does UTF-8 validation, all
    // things we don't need since we use simd_json for them. Also, `serde_json::StreamDeserializer` has a more
    // ambitious goal: it wants to parse potentially *non-delimited* sequences of JSON values, while we know
    // our values are line-delimited. Turns out, custom splitting is very easy, and gives a very nice performance boost.
    bytes
        .split(|&byte| byte == b'\n')
        .filter(|bytes| is_json_line(bytes))
}

#[inline]
pub fn is_json_line(bytes: &[u8]) -> bool {
    bytes
        .iter()
        .any(|byte| !matches!(*byte, b' ' | b'\t' | b'\r'))
}

fn parse_lines(
    bytes: &[u8],
    buffers: &mut PlIndexMap<BufferKey, Buffer>,
    ignore_errors: bool,
) -> PolarsResult<()> {
    let mut scratch = Scratch::default();

    let iter = json_lines(bytes);
    for bytes in iter {
        parse_impl(bytes, buffers, &mut scratch, ignore_errors)?;
    }
    Ok(())
}

pub fn parse_ndjson(
    bytes: &[u8],
    n_rows_hint: Option<usize>,
    schema: &Schema,
    ignore_errors: bool,
) -> PolarsResult<DataFrame> {
    let capacity = n_rows_hint.unwrap_or_else(|| estimate_n_lines_in_chunk(bytes));

    let mut buffers = init_buffers(schema, capacity, ignore_errors)?;
    parse_lines(bytes, &mut buffers, ignore_errors)?;

    DataFrame::new_infer_height(
        buffers
            .into_values()
            .map(|buf| Ok(buf.into_series()?.into_column()))
            .collect::<PolarsResult<_>>()
            .map_err(|e| match e {
                // Nested types raise SchemaMismatch instead of ComputeError, we map it back here to
                // be consistent.
                PolarsError::ComputeError(..) => e,
                PolarsError::SchemaMismatch(e) => PolarsError::ComputeError(e),
                e => e,
            })?,
    )
}

pub fn estimate_n_lines_in_file(file_bytes: &[u8], sample_size: usize) -> usize {
    if let Some((mean, std)) = get_line_stats_json(file_bytes, sample_size) {
        (file_bytes.len() as f32 / (mean - 0.01 * std)) as usize
    } else {
        estimate_n_lines_in_chunk(file_bytes)
    }
}

/// Total len divided by max len of first and last non-empty lines. This is intended to be cheaper
/// than `estimate_n_lines_in_file`.
pub fn estimate_n_lines_in_chunk(chunk: &[u8]) -> usize {
    chunk
        .split(|&c| c == b'\n')
        .find(|x| !x.is_empty())
        .map_or(1, |x| {
            chunk.len().div_ceil(
                x.len().max(
                    chunk
                        .rsplit(|&c| c == b'\n')
                        .find(|x| !x.is_empty())
                        .unwrap()
                        .len(),
                ),
            )
        })
}

/// Find the nearest next line position.
/// Does not check for new line characters embedded in String fields.
/// This just looks for `}\n`
pub(crate) fn next_line_position_naive_json(input: &[u8]) -> Option<usize> {
    let pos = memchr::memchr(NEWLINE, input)?;
    if pos == 0 {
        return Some(1);
    }

    let is_closing_bracket = input.get(pos - 1) == Some(&CLOSING_BRACKET);
    if is_closing_bracket {
        Some(pos + 1)
    } else {
        None
    }
}

/// Get the mean and standard deviation of length of lines in bytes
pub(crate) fn get_line_stats_json(bytes: &[u8], n_lines: usize) -> Option<(f32, f32)> {
    let mut lengths = Vec::with_capacity(n_lines);

    let mut bytes_trunc;
    let n_lines_per_iter = n_lines / 2;

    let mut n_read = 0;

    let bytes_len = bytes.len();

    // sample from start and 75% in the file
    for offset in [0, (bytes_len as f32 * 0.75) as usize] {
        bytes_trunc = &bytes[offset..];
        let pos = next_line_position_naive_json(bytes_trunc)?;
        if pos >= bytes_len {
            return None;
        }
        bytes_trunc = &bytes_trunc[pos + 1..];

        for _ in offset..(offset + n_lines_per_iter) {
            let pos = next_line_position_naive_json(bytes_trunc);
            if let Some(pos) = pos {
                lengths.push(pos);
                let next_bytes = &bytes_trunc[pos..];
                if next_bytes.is_empty() {
                    return None;
                }
                bytes_trunc = next_bytes;
                n_read += pos;
            } else {
                break;
            }
        }
    }

    let n_samples = lengths.len();
    let mean = (n_read as f32) / (n_samples as f32);
    let mut std = 0.0;
    for &len in lengths.iter() {
        std += (len as f32 - mean).pow(2.0)
    }
    std = (std / n_samples as f32).sqrt();
    Some((mean, std))
}

pub(crate) fn get_file_chunks_json(bytes: &[u8], n_threads: usize) -> Vec<(usize, usize)> {
    let mut last_pos = 0;
    let total_len = bytes.len();
    let chunk_size = total_len / n_threads;
    let mut offsets = Vec::with_capacity(n_threads);
    for _ in 0..n_threads {
        let search_pos = last_pos + chunk_size;

        if search_pos >= bytes.len() {
            break;
        }

        let end_pos = match next_line_position_naive_json(&bytes[search_pos..]) {
            Some(pos) => search_pos + pos,
            None => {
                break;
            },
        };
        offsets.push((last_pos, end_pos));
        last_pos = end_pos;
    }
    offsets.push((last_pos, total_len));
    offsets
}
