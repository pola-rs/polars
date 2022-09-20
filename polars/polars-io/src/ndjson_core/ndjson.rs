use std::borrow::Cow;
use std::fs::File;
use std::io::Cursor;
use std::path::PathBuf;

pub use arrow::array::StructArray;
pub use arrow::io::ndjson as arrow_ndjson;
use polars_core::prelude::*;
use polars_core::utils::accumulate_dataframes_vertical;
use polars_core::POOL;
use rayon::prelude::*;

use crate::csv::parser::*;
use crate::csv::utils::*;
use crate::mmap::{MmapBytesReader, ReaderBytes};
use crate::ndjson_core::buffer::*;
use crate::prelude::*;
const QUOTE_CHAR: u8 = b'"';
const SEP: u8 = b',';
const NEWLINE: u8 = b'\n';
const RETURN: u8 = b'\r';

#[must_use]
pub struct JsonLineReader<'a, R>
where
    R: MmapBytesReader,
{
    reader: R,
    rechunk: bool,
    n_rows: Option<usize>,
    n_threads: Option<usize>,
    infer_schema_len: Option<usize>,
    chunk_size: usize,
    schema: Option<&'a Schema>,
    path: Option<PathBuf>,
    low_memory: bool,
}

impl<'a, R> JsonLineReader<'a, R>
where
    R: 'a + MmapBytesReader,
{
    pub fn with_n_rows(mut self, num_rows: Option<usize>) -> Self {
        self.n_rows = num_rows;
        self
    }
    pub fn with_schema(mut self, schema: &'a Schema) -> Self {
        self.schema = Some(schema);
        self
    }
    pub fn with_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    pub fn infer_schema_len(mut self, infer_schema_len: Option<usize>) -> Self {
        self.infer_schema_len = infer_schema_len;
        self
    }

    pub fn with_n_threads(mut self, n: Option<usize>) -> Self {
        self.n_threads = n;
        self
    }

    pub fn with_path<P: Into<PathBuf>>(mut self, path: Option<P>) -> Self {
        self.path = path.map(|p| p.into());
        self
    }
    /// Sets the chunk size used by the parser. This influences performance
    pub fn with_chunk_size(mut self, chunk_size: Option<usize>) -> Self {
        if let Some(chunk_size) = chunk_size {
            self.chunk_size = chunk_size;
        };

        self
    }
    /// Reduce memory consumption at the expense of performance
    pub fn low_memory(mut self, toggle: bool) -> Self {
        self.low_memory = toggle;
        self
    }
}

impl<'a> JsonLineReader<'a, File> {
    /// This is the recommended way to create a json reader as this allows for fastest parsing.
    pub fn from_path<P: Into<PathBuf>>(path: P) -> PolarsResult<Self> {
        let path = resolve_homedir(&path.into());
        let f = std::fs::File::open(&path)?;
        Ok(Self::new(f).with_path(Some(path)))
    }
}
impl<'a, R> SerReader<R> for JsonLineReader<'a, R>
where
    R: MmapBytesReader,
{
    /// Create a new JsonLineReader from a file/ stream
    fn new(reader: R) -> Self {
        JsonLineReader {
            reader,
            rechunk: true,
            n_rows: None,
            n_threads: None,
            infer_schema_len: Some(128),
            schema: None,
            path: None,
            chunk_size: 1 << 18,
            low_memory: false,
        }
    }
    fn finish(mut self) -> PolarsResult<DataFrame> {
        let rechunk = self.rechunk;
        let reader_bytes = get_reader_bytes(&mut self.reader)?;
        let mut json_reader = CoreJsonReader::new(
            reader_bytes,
            self.n_rows,
            self.schema,
            self.n_threads,
            1024, // sample size
            self.chunk_size,
            self.low_memory,
            self.infer_schema_len,
        )?;

        let mut df: DataFrame = json_reader.as_df()?;
        if rechunk && df.n_chunks()? > 1 {
            df.as_single_chunk_par();
        }
        Ok(df)
    }
}

pub(crate) struct CoreJsonReader<'a> {
    reader_bytes: Option<ReaderBytes<'a>>,
    n_rows: Option<usize>,
    schema: Cow<'a, Schema>,
    n_threads: Option<usize>,
    sample_size: usize,
    chunk_size: usize,
    low_memory: bool,
}
impl<'a> CoreJsonReader<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        reader_bytes: ReaderBytes<'a>,
        n_rows: Option<usize>,
        schema: Option<&'a Schema>,
        n_threads: Option<usize>,
        sample_size: usize,
        chunk_size: usize,
        low_memory: bool,
        infer_schema_len: Option<usize>,
    ) -> PolarsResult<CoreJsonReader<'a>> {
        let reader_bytes = reader_bytes;

        let schema = match schema {
            Some(schema) => Cow::Borrowed(schema),
            None => {
                let bytes: &[u8] = &reader_bytes;
                let mut cursor = Cursor::new(bytes);

                let data_type = arrow_ndjson::read::infer(&mut cursor, infer_schema_len).unwrap();
                let schema: polars_core::prelude::Schema =
                    StructArray::get_fields(&data_type).into();

                Cow::Owned(schema)
            }
        };
        Ok(CoreJsonReader {
            reader_bytes: Some(reader_bytes),
            schema,
            sample_size,
            n_rows,
            n_threads,
            chunk_size,
            low_memory,
        })
    }
    fn parse_json(&mut self, mut n_threads: usize, bytes: &[u8]) -> PolarsResult<DataFrame> {
        let mut bytes = bytes;
        let mut total_rows = 128;

        if let Some((mean, std)) = get_line_stats(
            bytes,
            self.sample_size,
            NEWLINE,
            self.schema.len(),
            b',',
            None,
        ) {
            let line_length_upper_bound = mean + 1.1 * std;

            total_rows = (bytes.len() as f32 / (mean - 0.01 * std)) as usize;
            if let Some(n_rows) = self.n_rows {
                total_rows = std::cmp::min(n_rows, total_rows);
                // the guessed upper bound of  the no. of bytes in the file
                let n_bytes = (line_length_upper_bound * (n_rows as f32)) as usize;

                if n_bytes < bytes.len() {
                    if let Some(pos) = next_line_position_naive(&bytes[n_bytes..], NEWLINE) {
                        bytes = &bytes[..n_bytes + pos]
                    }
                }
            }
        }

        if total_rows == 128 {
            n_threads = 1;
        }

        let rows_per_thread = total_rows / n_threads;

        let max_proxy = bytes.len() / n_threads / 2;
        let capacity = if self.low_memory {
            self.chunk_size
        } else {
            std::cmp::min(rows_per_thread, max_proxy)
        };

        let expected_fields = &self.schema.len();

        let file_chunks = get_file_chunks(
            bytes,
            n_threads,
            *expected_fields + 1,
            SEP,
            Some(QUOTE_CHAR),
            NEWLINE,
        );
        let dfs = POOL.install(|| {
            file_chunks
                .into_par_iter()
                .map(|(start_pos, stop_at_nbytes)| {
                    let mut buffers = init_buffers(&self.schema, capacity)?;
                    let _ = parse_lines(&bytes[start_pos..stop_at_nbytes], &mut buffers);

                    DataFrame::new(
                        buffers
                            .into_values()
                            .map(|buf| buf.into_series())
                            .collect::<PolarsResult<_>>()?,
                    )
                })
                .collect::<PolarsResult<Vec<_>>>()
        })?;
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
fn parse_impl<'a>(
    bytes: &[u8],
    buffers: &mut PlIndexMap<BufferKey, Buffer<'a>>,
    line: &mut Vec<u8>,
) -> PolarsResult<usize> {
    line.clear();
    line.extend_from_slice(bytes);

    match line.len() {
        0 => Ok(0),
        1 => {
            if line[0] == NEWLINE {
                Ok(1)
            } else {
                Err(PolarsError::ComputeError(
                    "Invalid JSON: unexpected end of file".into(),
                ))
            }
        }
        2 => {
            if line[0] == NEWLINE && line[1] == RETURN {
                Ok(2)
            } else {
                Err(PolarsError::ComputeError(
                    "Invalid JSON: unexpected end of file".into(),
                ))
            }
        }
        n => {
            let value: simd_json::BorrowedValue =
                simd_json::to_borrowed_value(line).map_err(|e| {
                    PolarsError::ComputeError(format!("Error parsing line: {}", e).into())
                })?;

            match value {
                simd_json::BorrowedValue::Object(value) => {
                    buffers
                        .iter_mut()
                        .for_each(|(s, inner)| match s.0.map_lookup(&value) {
                            Some(v) => inner.add(v).expect("inner.add(v)"),
                            None => inner.add_null(),
                        });
                }
                _ => {
                    buffers.iter_mut().for_each(|(_, inner)| inner.add_null());
                }
            };
            Ok(n)
        }
    }
}

fn parse_lines<'a>(
    bytes: &[u8],
    buffers: &mut PlIndexMap<BufferKey, Buffer<'a>>,
) -> PolarsResult<()> {
    let mut buf = vec![];

    let total_bytes = bytes.len();
    let mut offset = 0;
    for line in SplitLines::new(bytes, NEWLINE) {
        offset += 1; // the newline
        offset += parse_impl(line, buffers, &mut buf)?;
    }

    // if file doesn't end with a newline, parse the last line
    if offset < total_bytes {
        let rem = &bytes[offset..];
        offset += rem.len();
        parse_impl(rem, buffers, &mut buf)?;
    }

    if offset != total_bytes {
        return Err(PolarsError::ComputeError(
            format!("Expected {} bytes, but only parsed {}", total_bytes, offset).into(),
        ));
    };

    Ok(())
}
