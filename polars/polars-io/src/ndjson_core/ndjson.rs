use crate::csv::parser::*;
use crate::csv::utils::*;
use crate::mmap::ReaderBytes;
use crate::ndjson_core::buffer::*;
use crate::prelude::*;

pub use arrow::{array::StructArray, io::ndjson};

use crate::mmap::MmapBytesReader;
use polars_core::{prelude::*, utils::accumulate_dataframes_vertical, POOL};
use rayon::prelude::*;
use serde_json::{Deserializer, Value};
use std::borrow::Cow;
use std::fs::File;
use std::io::Cursor;
use std::path::PathBuf;
const QUOTE_CHAR: u8 = "\"".as_bytes()[0];
const SEP: u8 = ",".as_bytes()[0];

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
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
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
    pub fn from_path<P: Into<PathBuf>>(path: P) -> Result<Self> {
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
    fn finish(mut self) -> Result<DataFrame> {
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
    ) -> Result<CoreJsonReader<'a>> {
        let reader_bytes = reader_bytes;

        let schema = match schema {
            Some(schema) => Cow::Borrowed(schema),
            None => {
                let bytes: &[u8] = &reader_bytes;
                let mut cursor = Cursor::new(bytes);

                let data_type = ndjson::read::infer(&mut cursor, infer_schema_len).unwrap();
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
    fn parse_json(&mut self, mut n_threads: usize, bytes: &[u8]) -> Result<DataFrame> {
        let mut bytes = bytes;
        let mut total_rows = 128;

        if let Some((mean, std)) = get_line_stats(bytes, self.sample_size, b'\n') {
            let line_length_upper_bound = mean + 1.1 * std;

            total_rows = (bytes.len() as f32 / (mean - 0.01 * std)) as usize;
            if let Some(n_rows) = self.n_rows {
                total_rows = std::cmp::min(n_rows, total_rows);
                // the guessed upper bound of  the no. of bytes in the file
                let n_bytes = (line_length_upper_bound * (n_rows as f32)) as usize;

                if n_bytes < bytes.len() {
                    if let Some(pos) = next_line_position_naive(&bytes[n_bytes..], b'\n') {
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
            *expected_fields,
            SEP,
            Some(QUOTE_CHAR),
            b'\n',
        );

        let dfs = POOL.install(|| {
            file_chunks
                .into_par_iter()
                .map(|(bytes_offset_thread, stop_at_nbytes)| {
                    let mut read = bytes_offset_thread;

                    let mut last_read = usize::MAX;

                    let mut buffers = init_buffers(&self.schema, capacity)?;

                    loop {
                        if read >= stop_at_nbytes || read == last_read {
                            break;
                        }
                        let local_bytes = &bytes[read..stop_at_nbytes];

                        last_read = read;
                        read += parse_lines(local_bytes, &mut buffers)?;
                    }
                    DataFrame::new(
                        buffers
                            .into_values()
                            .map(|buf| buf.into_series())
                            .collect::<Result<_>>()?,
                    )
                })
                .collect::<Result<Vec<_>>>()
        })?;
        accumulate_dataframes_vertical(dfs)
    }
    pub fn as_df(&mut self) -> Result<DataFrame> {
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

fn parse_lines<'a>(bytes: &[u8], buffers: &mut PlIndexMap<String, Buffer<'a>>) -> Result<usize> {
    let mut stream = Deserializer::from_slice(bytes).into_iter::<Value>();
    for value in stream.by_ref() {
        let v = value.unwrap_or(Value::Null);
        match v {
            Value::Object(value) => {
                buffers
                    .iter_mut()
                    .for_each(|(s, inner)| match value.get(s) {
                        Some(v) => inner.add(v).expect("inner.add(v)"),
                        None => inner.add_null(),
                    });
            }
            _ => {
                buffers.iter_mut().for_each(|(_, inner)| inner.add_null());
            }
        };
    }

    let byte_offset = stream.byte_offset();

    Ok(byte_offset)
}
