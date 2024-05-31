use std::collections::VecDeque;
use std::ops::Deref;

use polars_core::datatypes::Field;
use polars_core::frame::DataFrame;
use polars_core::schema::SchemaRef;
use polars_core::POOL;
use polars_error::PolarsResult;
use polars_utils::IdxSize;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use super::{cast_columns, read_chunk, CoreReader};
use crate::csv::read::options::{CommentPrefix, CsvEncoding, NullValuesCompiled};
use crate::csv::read::parser::next_line_position;
use crate::csv::read::CsvReader;
use crate::mmap::{MmapBytesReader, ReaderBytes};
use crate::prelude::update_row_counts2;
use crate::RowIndex;

#[allow(clippy::too_many_arguments)]
pub(crate) fn get_file_chunks_iterator(
    offsets: &mut VecDeque<(usize, usize)>,
    last_pos: &mut usize,
    n_chunks: usize,
    chunk_size: usize,
    bytes: &[u8],
    expected_fields: usize,
    separator: u8,
    quote_char: Option<u8>,
    eol_char: u8,
) {
    for _ in 0..n_chunks {
        let search_pos = *last_pos + chunk_size;

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
        offsets.push_back((*last_pos, end_pos));
        *last_pos = end_pos;
    }
}

struct ChunkOffsetIter<'a> {
    bytes: &'a [u8],
    offsets: VecDeque<(usize, usize)>,
    last_offset: usize,
    n_chunks: usize,
    // not a promise, but something we want
    rows_per_batch: usize,
    expected_fields: usize,
    separator: u8,
    quote_char: Option<u8>,
    eol_char: u8,
}

impl<'a> Iterator for ChunkOffsetIter<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        match self.offsets.pop_front() {
            Some(offsets) => Some(offsets),
            None => {
                if self.last_offset == self.bytes.len() {
                    return None;
                }
                let bytes_first_row = if self.rows_per_batch > 1 {
                    let bytes_first_row = next_line_position(
                        &self.bytes[self.last_offset + 2..],
                        Some(self.expected_fields),
                        self.separator,
                        self.quote_char,
                        self.eol_char,
                    )
                    .unwrap_or(1);
                    bytes_first_row + 2
                } else {
                    1
                };
                get_file_chunks_iterator(
                    &mut self.offsets,
                    &mut self.last_offset,
                    self.n_chunks,
                    self.rows_per_batch * bytes_first_row,
                    self.bytes,
                    self.expected_fields,
                    self.separator,
                    self.quote_char,
                    self.eol_char,
                );
                match self.offsets.pop_front() {
                    Some(offsets) => Some(offsets),
                    // We depleted the iterator. Ensure we deplete the slice as well
                    None => {
                        let out = Some((self.last_offset, self.bytes.len()));
                        self.last_offset = self.bytes.len();
                        out
                    },
                }
            },
        }
    }
}

impl<'a> CoreReader<'a> {
    /// Create a batched csv reader that uses mmap to load data.
    pub fn batched(mut self, _has_cat: bool) -> PolarsResult<BatchedCsvReader<'a>> {
        let reader_bytes = self.reader_bytes.take().unwrap();
        let bytes = reader_bytes.as_ref();
        let (bytes, starting_point_offset) =
            self.find_starting_point(bytes, self.quote_char, self.eol_char)?;

        // this is arbitrarily chosen.
        // we don't want this to depend on the thread pool size
        // otherwise the chunks are not deterministic
        let offset_batch_size = 16;
        // extend lifetime. It is bound to `readerbytes` and we keep track of that
        // lifetime so this is sound.
        let bytes = unsafe { std::mem::transmute::<&[u8], &'static [u8]>(bytes) };
        let file_chunks = ChunkOffsetIter {
            bytes,
            offsets: VecDeque::with_capacity(offset_batch_size),
            last_offset: 0,
            n_chunks: offset_batch_size,
            rows_per_batch: self.chunk_size,
            expected_fields: self.schema.len(),
            separator: self.separator,
            quote_char: self.quote_char,
            eol_char: self.eol_char,
        };

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

        Ok(BatchedCsvReader {
            reader_bytes,
            chunk_size: self.chunk_size,
            file_chunks_iter: file_chunks,
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
        })
    }
}

pub struct BatchedCsvReader<'a> {
    reader_bytes: ReaderBytes<'a>,
    chunk_size: usize,
    file_chunks_iter: ChunkOffsetIter<'a>,
    file_chunks: Vec<(usize, usize)>,
    projection: Vec<usize>,
    starting_point_offset: Option<usize>,
    row_index: Option<RowIndex>,
    comment_prefix: Option<CommentPrefix>,
    quote_char: Option<u8>,
    eol_char: u8,
    null_values: Option<NullValuesCompiled>,
    missing_is_null: bool,
    truncate_ragged_lines: bool,
    to_cast: Vec<Field>,
    ignore_errors: bool,
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
}

impl<'a> BatchedCsvReader<'a> {
    pub fn next_batches(&mut self, n: usize) -> PolarsResult<Option<Vec<DataFrame>>> {
        if n == 0 || self.remaining == 0 {
            return Ok(None);
        }

        // get next `n` offset positions.
        let file_chunks_iter = (&mut self.file_chunks_iter).take(n);
        self.file_chunks.extend(file_chunks_iter);
        // depleted the offsets iterator, we are done as well.
        if self.file_chunks.is_empty() {
            return Ok(None);
        }
        let chunks = &self.file_chunks;

        let mut bytes = self.reader_bytes.deref();
        if let Some(pos) = self.starting_point_offset {
            bytes = &bytes[pos..];
        }

        let mut chunks = POOL.install(|| {
            chunks
                .into_par_iter()
                .copied()
                .map(|(bytes_offset_thread, stop_at_nbytes)| {
                    let mut df = read_chunk(
                        bytes,
                        self.separator,
                        self.schema.as_ref(),
                        self.ignore_errors,
                        &self.projection,
                        bytes_offset_thread,
                        self.quote_char,
                        self.eol_char,
                        self.comment_prefix.as_ref(),
                        self.chunk_size,
                        self.encoding,
                        self.null_values.as_ref(),
                        self.missing_is_null,
                        self.truncate_ragged_lines,
                        self.chunk_size,
                        stop_at_nbytes,
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
    batched_reader: BatchedCsvReader<'static>,
    // keep ownership
    _reader: CsvReader<Box<dyn MmapBytesReader>>,
}

impl OwnedBatchedCsvReader {
    pub fn next_batches(&mut self, n: usize) -> PolarsResult<Option<Vec<DataFrame>>> {
        self.batched_reader.next_batches(n)
    }
}

pub fn to_batched_owned(mut reader: CsvReader<Box<dyn MmapBytesReader>>) -> OwnedBatchedCsvReader {
    let schema = reader.get_schema().unwrap();
    let batched_reader = reader.batched_borrowed().unwrap();
    // If you put a drop(reader) here, rust will complain that reader is borrowed,
    // so we presumably have to keep ownership of it to maintain the safety of the
    // 'static transmute.
    let batched_reader: BatchedCsvReader<'static> = unsafe { std::mem::transmute(batched_reader) };

    OwnedBatchedCsvReader {
        schema,
        batched_reader,
        _reader: reader,
    }
}
