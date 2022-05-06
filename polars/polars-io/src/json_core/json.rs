use crate::csv_core::parser::*;
use crate::csv_core::utils::*;
use crate::json_core::buffer::*;
use crate::mmap::ReaderBytes;
use crate::prelude::*;
use crate::RowCount;
use arrow::array::{ArrayRef, StructArray};
use arrow::io::ndjson::read::FallibleStreamingIterator;
pub use arrow::{
    error::Result as ArrowResult,
    io::{json, ndjson},
};
use polars_arrow::conversion::chunk_to_struct;
use polars_arrow::kernels::concatenate::concatenate_owned_unchecked;
use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::convert::TryFrom;
use std::fs::File;
use std::io::{BufRead, Seek, Write};
use std::path::PathBuf;

use crate::mmap::MmapBytesReader;
use polars_core::utils::accumulate_dataframes_vertical;
use polars_core::{prelude::*, POOL};
use rayon::prelude::*;
use serde_json::{Deserializer, Value};
use std::io::Cursor;

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
    infer_schema_len: Option<usize>,
    chunk_size: usize,
    projection: Option<Vec<String>>,
    schema: Option<&'a Schema>,
    row_count: Option<RowCount>,
    path: Option<PathBuf>,
}

impl<'a, R> JsonLineReader<'a, R>
where
    R: 'a + MmapBytesReader,
{
    /// Add a `row_count` column.
    pub fn with_row_count(mut self, rc: Option<RowCount>) -> Self {
        self.row_count = rc;
        self
    }
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
        // used by error ignore logic
        self.infer_schema_len = infer_schema_len;
        self
    }
    pub fn with_path<P: Into<PathBuf>>(mut self, path: Option<P>) -> Self {
        self.path = path.map(|p| p.into());
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
    /// Create a new CsvReader from a file/ stream
    fn new(reader: R) -> Self {
        JsonLineReader {
            reader,
            rechunk: true,
            n_rows: None,
            infer_schema_len: Some(128),
            projection: None,
            schema: None,
            path: None,
            chunk_size: 1 << 18,
            row_count: None,
        }
    }
    fn finish(mut self) -> Result<DataFrame> {
        let rechunk = self.rechunk;
        let reader_bytes = get_reader_bytes(&mut self.reader)?;
        let mut json_reader = CoreJsonReader::new(
            reader_bytes,
            self.n_rows,
            self.schema,
            self.infer_schema_len,
            None,
            self.row_count,
            1024,
            self.chunk_size,
            0,
            None,
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
    /// Explicit schema for the CSV file
    schema: Cow<'a, Schema>,
    /// Optional projection for which columns to load (zero-based column indices)
    projection: Option<Vec<usize>>,
    /// Current line number, used in error reporting
    line_number: usize,
    skip_rows: usize,
    n_rows: Option<usize>,
    n_threads: Option<usize>,
    sample_size: usize,
    chunk_size: usize,
    row_count: Option<RowCount>,
}
impl<'a> CoreJsonReader<'a> {
    pub(crate) fn new(
        reader_bytes: ReaderBytes<'a>,
        n_rows: Option<usize>,
        schema: Option<&'a Schema>,
        infer_schema_len: Option<usize>,
        n_threads: Option<usize>,
        row_count: Option<RowCount>,
        sample_size: usize,
        chunk_size: usize,
        skip_rows: usize,
        mut projection: Option<Vec<usize>>,
    ) -> Result<CoreJsonReader<'a>> {
        let mut reader_bytes = reader_bytes;

        let mut schema = match schema {
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
            chunk_size,
            skip_rows,
            line_number: 0,
            n_rows,
            n_threads,
            row_count,
            projection,
        })
    }
    fn parse_json(&mut self, mut n_threads: usize, mut bytes: &[u8]) -> Result<DataFrame> {
        let logging = std::env::var("POLARS_VERBOSE").is_ok();
        // todo!()
        let low_memory = false;

        let mut total_rows = 128;
        let infer_len = Some(128);

        let mut bytes = &bytes[..];

        if let Some((mean, std)) = get_line_stats(bytes, self.sample_size) {
            // x % upper bound of byte length per line assuming normally distributed
            let line_length_upper_bound = mean + 1.1 * std;
            total_rows = (bytes.len() as f32 / (mean - 0.01 * std)) as usize;
        }

        if total_rows == 128 {
            n_threads = 1;
        }

        let n_chunks = total_rows / self.chunk_size;

        let rows_per_thread = total_rows / n_threads;

        let max_proxy = bytes.len() / n_threads / 2;
        let capacity = if low_memory {
            self.chunk_size
        } else {
            std::cmp::min(rows_per_thread, max_proxy)
        };

        // assume 10 chars per str
        let init_str_bytes = capacity * 5;

        let expected_fields = &self.schema.len();
        let file_chunks =
            get_file_chunks(bytes, n_threads, *expected_fields, SEP, Some(QUOTE_CHAR));

        let mut dfs = file_chunks
            .into_par_iter()
            .map(|(bytes_offset_thread, stop_at_nbytes)| {
                let mut read = bytes_offset_thread;

                let mut last_read = usize::MAX;

                let mut buffers = init_buffers(&self.schema, capacity).unwrap();

                loop {
                    if read >= stop_at_nbytes || read == last_read {
                        break;
                    }
                    let byte_len = bytes.len();
                    let local_bytes = &bytes[read..stop_at_nbytes];

                    last_read = read;
                    read += parse_lines(local_bytes, &mut buffers).unwrap();
                }
                let df = DataFrame::new_no_checks(
                    buffers
                        .into_iter()
                        .map(|(_, buf)| buf.into_series().unwrap())
                        .collect::<Vec<_>>(),
                );
                df
            })
            .collect::<Vec<_>>();

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

fn infer(reader_bytes: &ReaderBytes, infer_schema_len: Option<usize>) -> Result<(Schema, usize)> {
    todo!()
}

fn parse_lines(bytes: &[u8], buffers: &mut PlHashMap<String, Buffer>) -> Result<usize> {
    let mut stream = Deserializer::from_slice(bytes).into_iter::<Value>();
    while let Some(value) = stream.next() {
        let v = value.unwrap();

        match v {
            Value::Object(value) => {
                buffers
                    .iter_mut()
                    .for_each(|(s, inner)| match value.get(s) {
                        Some(v) => {
                            inner.add(v).expect("inner.add(v)");
                        }
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
