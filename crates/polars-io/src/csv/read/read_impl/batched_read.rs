use std::collections::VecDeque;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use polars_core::datatypes::Field;
use polars_core::frame::DataFrame;
use polars_core::schema::SchemaRef;
use polars_core::POOL;
use polars_error::PolarsResult;
use polars_utils::sync::SyncPtr;
use polars_utils::IdxSize;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use super::{cast_columns, read_chunk, CoreReader};
use crate::csv::read::options::{CommentPrefix, CsvEncoding, NullValuesCompiled};
use crate::csv::read::parser::next_line_position;
use crate::csv::read::CsvReader;
use crate::mmap::{MmapBytesReader, ReaderBytes};
use crate::prelude::update_row_counts2;
use crate::RowIndex;

#[allow(clippy::too_many_arguments)]
pub(crate) fn get_offsets(
    offsets: &mut VecDeque<(usize, usize)>,
    n_chunks: usize,
    chunk_size: usize,
    bytes: &[u8],
    expected_fields: usize,
    separator: u8,
    quote_char: Option<u8>,
    eol_char: u8,
) {
    let mut start = 0;
    for i in 1..(n_chunks + 1) {
        let search_pos = chunk_size * i;

        if search_pos >= bytes.len() {
            break;
        }

        let end_pos = match next_line_position(
            &bytes[search_pos..],
            Some(expected_fields),
            separator,
            quote_char,
            eol_char,
        ) {
            Some(pos) => search_pos + pos,
            None => {
                break;
            },
        };
        offsets.push_back((start, end_pos));
        start = end_pos;
    }
}

/// Reads bytes from `file` to `buf` and returns pointers into `buf` that can be parsed.
/// TODO! this can be implemented without copying by pointing in the memmapped file.
struct ChunkReader<'a> {
    file: &'a File,
    buf: Vec<u8>,
    finished: bool,
    page_size: u64,
    // position in the buffer we read
    // this must be set by the caller of this chunkreader
    // after it iterated all offsets.
    buf_end: usize,
    offsets: VecDeque<(usize, usize)>,
    n_chunks: usize,
    // not a promise, but something we want
    rows_per_batch: usize,
    expected_fields: usize,
    separator: u8,
    quote_char: Option<u8>,
    eol_char: u8,
}

impl<'a> ChunkReader<'a> {
    fn new(
        file: &'a File,
        rows_per_batch: usize,
        expected_fields: usize,
        separator: u8,
        quote_char: Option<u8>,
        eol_char: u8,
        page_size: u64,
    ) -> Self {
        Self {
            file,
            buf: vec![],
            buf_end: 0,
            offsets: VecDeque::new(),
            finished: false,
            page_size,
            // this is arbitrarily chosen.
            // we don't want this to depend on the thread pool size
            // otherwise the chunks are not deterministic
            n_chunks: 16,
            rows_per_batch,
            expected_fields,
            separator,
            quote_char,
            eol_char,
        }
    }

    fn reslice(&mut self) {
        // memcopy the remaining bytes to the start
        self.buf.copy_within(self.buf_end.., 0);
        self.buf.truncate(self.buf.len() - self.buf_end);
        self.buf_end = 0;
    }

    fn return_slice(&self, start: usize, end: usize) -> (SyncPtr<u8>, usize) {
        let slice = &self.buf[start..end];
        let len = slice.len();
        (slice.as_ptr().into(), len)
    }

    fn get_buf_remaining(&self) -> (SyncPtr<u8>, usize) {
        let slice = &self.buf[self.buf_end..];
        let len = slice.len();
        (slice.as_ptr().into(), len)
    }

    // Get next `n` offset positions. Where `n` is number of chunks.

    // This returns pointers into slices into `buf`
    // we must process the slices before the next call
    // as that will overwrite the slices
    fn read(&mut self, n: usize) -> bool {
        self.reslice();

        if self.buf.len() <= self.page_size as usize {
            let read = self
                .file
                .take(self.page_size)
                .read_to_end(&mut self.buf)
                .unwrap();

            if read == 0 {
                self.finished = true;
                return false;
            }
        }

        let bytes_first_row = if self.rows_per_batch > 1 {
            let mut bytes_first_row;
            loop {
                bytes_first_row = next_line_position(
                    &self.buf[2..],
                    Some(self.expected_fields),
                    self.separator,
                    self.quote_char,
                    self.eol_char,
                );

                if bytes_first_row.is_some() {
                    break;
                } else {
                    let read = self
                        .file
                        .take(self.page_size)
                        .read_to_end(&mut self.buf)
                        .unwrap();
                    if read == 0 {
                        self.finished = true;
                        return false;
                    }
                }
            }
            bytes_first_row.unwrap_or(1) + 2
        } else {
            1
        };
        let expected_bytes = self.rows_per_batch * bytes_first_row * (n + 1);
        if self.buf.len() < expected_bytes {
            let to_read = expected_bytes - self.buf.len();
            let read = self
                .file
                .take(to_read as u64)
                .read_to_end(&mut self.buf)
                .unwrap();
            if read == 0 {
                self.finished = true;
                // don't return yet as we initially
                // read `page_size` len.
                // This can mean that the whole file
                // fits into `page_size`, so we continue
                // to collect offsets
            }
        }

        get_offsets(
            &mut self.offsets,
            self.n_chunks,
            self.rows_per_batch * bytes_first_row,
            &self.buf,
            self.expected_fields,
            self.separator,
            self.quote_char,
            self.eol_char,
        );
        !self.offsets.is_empty()
    }
}

impl<'a> CoreReader<'a> {
    /// Create a batched csv reader that uses read calls to load data.
    pub fn batched_read(mut self, _has_cat: bool) -> PolarsResult<BatchedCsvReaderRead<'a>> {
        let reader_bytes = self.reader_bytes.take().unwrap();

        let ReaderBytes::Mapped(bytes, mut file) = &reader_bytes else {
            unreachable!()
        };
        let (_, starting_point_offset) =
            self.find_starting_point(bytes, self.quote_char, self.eol_char)?;
        if let Some(starting_point_offset) = starting_point_offset {
            file.seek(SeekFrom::Current(starting_point_offset as i64))
                .unwrap();
        }

        let chunk_iter = ChunkReader::new(
            file,
            self.chunk_size,
            self.schema.len(),
            self.separator,
            self.quote_char,
            self.eol_char,
            4096,
        );

        let projection = self.get_projection()?;

        // RAII structure that will ensure we maintain a global stringcache
        #[cfg(feature = "dtype-categorical")]
        let _cat_lock = if _has_cat {
            Some(polars_core::StringCacheHolder::hold())
        } else {
            None
        };

        #[cfg(not(feature = "dtype-categorical"))]
        let _cat_lock = None;

        Ok(BatchedCsvReaderRead {
            chunk_size: self.chunk_size,
            file_chunk_reader: chunk_iter,
            file_chunks: vec![],
            projection,
            starting_point_offset,
            row_index: self.row_index,
            comment_prefix: self.comment_prefix,
            quote_char: self.quote_char,
            eol_char: self.eol_char,
            null_values: self.null_values,
            missing_is_null: self.missing_is_null,
            to_cast: self.to_cast,
            ignore_errors: self.ignore_errors,
            truncate_ragged_lines: self.truncate_ragged_lines,
            remaining: self.n_rows.unwrap_or(usize::MAX),
            encoding: self.encoding,
            separator: self.separator,
            schema: self.schema,
            rows_read: 0,
            _cat_lock,
            decimal_comma: self.decimal_comma,
            finished: false,
        })
    }
}

pub struct BatchedCsvReaderRead<'a> {
    chunk_size: usize,
    file_chunk_reader: ChunkReader<'a>,
    file_chunks: Vec<(SyncPtr<u8>, usize)>,
    projection: Vec<usize>,
    starting_point_offset: Option<usize>,
    row_index: Option<RowIndex>,
    comment_prefix: Option<CommentPrefix>,
    quote_char: Option<u8>,
    eol_char: u8,
    null_values: Option<NullValuesCompiled>,
    missing_is_null: bool,
    to_cast: Vec<Field>,
    ignore_errors: bool,
    truncate_ragged_lines: bool,
    remaining: usize,
    encoding: CsvEncoding,
    separator: u8,
    schema: SchemaRef,
    rows_read: IdxSize,
    #[cfg(feature = "dtype-categorical")]
    _cat_lock: Option<polars_core::StringCacheHolder>,
    #[cfg(not(feature = "dtype-categorical"))]
    _cat_lock: Option<u8>,
    decimal_comma: bool,
    finished: bool,
}
//
impl<'a> BatchedCsvReaderRead<'a> {
    /// `n` number of batches.
    pub fn next_batches(&mut self, n: usize) -> PolarsResult<Option<Vec<DataFrame>>> {
        if n == 0 || self.remaining == 0 || self.finished {
            return Ok(None);
        }

        // get next `n` offset positions.

        // This returns pointers into slices into `buf`
        // we must process the slices before the next call
        // as that will overwrite the slices
        if self.file_chunk_reader.read(n) {
            let mut latest_end = 0;
            while let Some((start, end)) = self.file_chunk_reader.offsets.pop_front() {
                latest_end = end;
                self.file_chunks
                    .push(self.file_chunk_reader.return_slice(start, end))
            }
            // ensure that this is set correctly
            self.file_chunk_reader.buf_end = latest_end;
        }
        // ensure we process the final slice as well.
        if self.file_chunk_reader.finished && self.file_chunks.len() < n {
            // get the final slice
            self.file_chunks
                .push(self.file_chunk_reader.get_buf_remaining());
            self.finished = true;
        }

        // depleted the offsets iterator, we are done as well.
        if self.file_chunks.is_empty() {
            return Ok(None);
        }

        let mut chunks = POOL.install(|| {
            self.file_chunks
                .par_iter()
                .map(|(ptr, len)| {
                    let chunk = unsafe { std::slice::from_raw_parts(ptr.get(), *len) };
                    let stop_at_n_bytes = chunk.len();
                    let mut df = read_chunk(
                        chunk,
                        self.separator,
                        self.schema.as_ref(),
                        self.ignore_errors,
                        &self.projection,
                        0,
                        self.quote_char,
                        self.eol_char,
                        self.comment_prefix.as_ref(),
                        self.chunk_size,
                        self.encoding,
                        self.null_values.as_ref(),
                        self.missing_is_null,
                        self.truncate_ragged_lines,
                        self.chunk_size,
                        stop_at_n_bytes,
                        self.starting_point_offset,
                        self.decimal_comma,
                    )?;

                    cast_columns(&mut df, &self.to_cast, false, self.ignore_errors)?;

                    if let Some(rc) = &self.row_index {
                        df.with_row_index_mut(&rc.name, Some(rc.offset));
                    }
                    Ok(df)
                })
                .collect::<PolarsResult<Vec<_>>>()
        })?;
        self.file_chunks.clear();

        if self.row_index.is_some() {
            update_row_counts2(&mut chunks, self.rows_read)
        }
        for df in &mut chunks {
            let h = df.height();

            if self.remaining < h {
                *df = df.slice(0, self.remaining)
            };
            self.remaining = self.remaining.saturating_sub(h);

            self.rows_read += h as IdxSize;
        }
        Ok(Some(chunks))
    }
}

pub struct OwnedBatchedCsvReader {
    #[allow(dead_code)]
    // this exist because we need to keep ownership
    schema: SchemaRef,
    batched_reader: BatchedCsvReaderRead<'static>,
    // keep ownership
    _reader: CsvReader<Box<dyn MmapBytesReader>>,
}

impl OwnedBatchedCsvReader {
    pub fn next_batches(&mut self, n: usize) -> PolarsResult<Option<Vec<DataFrame>>> {
        self.batched_reader.next_batches(n)
    }
}

pub fn to_batched_owned_read(
    mut reader: CsvReader<Box<dyn MmapBytesReader>>,
) -> OwnedBatchedCsvReader {
    let schema = reader.get_schema().unwrap();
    let batched_reader = reader.batched_borrowed_read().unwrap();
    // If you put a drop(reader) here, rust will complain that reader is borrowed,
    // so we presumably have to keep ownership of it to maintain the safety of the
    // 'static transmute.
    let batched_reader: BatchedCsvReaderRead<'static> =
        unsafe { std::mem::transmute(batched_reader) };

    OwnedBatchedCsvReader {
        schema,
        batched_reader,
        _reader: reader,
    }
}
