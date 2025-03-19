use std::collections::VecDeque;
use std::ops::Deref;

use polars_core::POOL;
use polars_core::datatypes::Field;
use polars_core::frame::DataFrame;
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_utils::IdxSize;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use super::{CoreReader, CountLines, cast_columns, read_chunk};
use crate::RowIndex;
use crate::csv::read::CsvReader;
use crate::csv::read::options::NullValuesCompiled;
use crate::mmap::{MmapBytesReader, ReaderBytes};
use crate::prelude::{CsvParseOptions, update_row_counts2};

#[allow(clippy::too_many_arguments)]
pub(crate) fn get_file_chunks_iterator(
    offsets: &mut VecDeque<(usize, usize)>,
    last_pos: &mut usize,
    n_chunks: usize,
    chunk_size: &mut usize,
    bytes: &[u8],
    quote_char: Option<u8>,
    eol_char: u8,
) {
    let cl = CountLines::new(quote_char, eol_char);

    for _ in 0..n_chunks {
        let bytes = &bytes[*last_pos..];

        if bytes.is_empty() {
            break;
        }

        let position;

        loop {
            let b = &bytes[..(*chunk_size).min(bytes.len())];
            let (count, position_) = cl.count(b);

            let (count, position_) = if b.len() == bytes.len() {
                (if count != 0 { count } else { 1 }, b.len())
            } else {
                (
                    count,
                    if position_ < b.len() {
                        // 1+ for the '\n'
                        1 + position_
                    } else {
                        position_
                    },
                )
            };

            if count == 0 {
                *chunk_size *= 2;
                continue;
            }

            position = position_;
            break;
        }

        offsets.push_back((*last_pos, *last_pos + position));
        *last_pos += position;
    }
}

struct ChunkOffsetIter<'a> {
    bytes: &'a [u8],
    offsets: VecDeque<(usize, usize)>,
    last_offset: usize,
    n_chunks: usize,
    chunk_size: usize,
    // not a promise, but something we want
    #[allow(unused)]
    rows_per_batch: usize,
    quote_char: Option<u8>,
    eol_char: u8,
}

impl Iterator for ChunkOffsetIter<'_> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        match self.offsets.pop_front() {
            Some(offsets) => Some(offsets),
            None => {
                if self.last_offset == self.bytes.len() {
                    return None;
                }
                get_file_chunks_iterator(
                    &mut self.offsets,
                    &mut self.last_offset,
                    self.n_chunks,
                    &mut self.chunk_size,
                    self.bytes,
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
    pub fn batched(mut self) -> PolarsResult<BatchedCsvReader<'a>> {
        let reader_bytes = self.reader_bytes.take().unwrap();
        let bytes = reader_bytes.as_ref();
        let (bytes, starting_point_offset) = self.find_starting_point(
            bytes,
            self.parse_options.quote_char,
            self.parse_options.eol_char,
        )?;

        let n_threads = self.n_threads.unwrap_or_else(|| POOL.current_num_threads());

        // Copied from [`Self::parse_csv`]
        let n_parts_hint = n_threads * 16;
        let chunk_size = std::cmp::min(bytes.len() / n_parts_hint, 16 * 1024 * 1024);

        // Use a small min chunk size to catch failures in tests.
        #[cfg(debug_assertions)]
        let min_chunk_size = 64;
        #[cfg(not(debug_assertions))]
        let min_chunk_size = 1024 * 4;

        let chunk_size = std::cmp::max(chunk_size, min_chunk_size);

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
            chunk_size,
            rows_per_batch: self.chunk_size,
            quote_char: self.parse_options.quote_char,
            eol_char: self.parse_options.eol_char,
        };

        let projection = self.get_projection()?;

        // RAII structure that will ensure we maintain a global stringcache
        #[cfg(feature = "dtype-categorical")]
        let _cat_lock = if self.has_categorical {
            Some(polars_core::StringCacheHolder::hold())
        } else {
            None
        };

        #[cfg(not(feature = "dtype-categorical"))]
        let _cat_lock = None;

        Ok(BatchedCsvReader {
            reader_bytes,
            parse_options: self.parse_options,
            chunk_size: self.chunk_size,
            file_chunks_iter: file_chunks,
            file_chunks: vec![],
            projection,
            starting_point_offset,
            row_index: self.row_index,
            null_values: self.null_values,
            to_cast: self.to_cast,
            ignore_errors: self.ignore_errors,
            remaining: self.n_rows.unwrap_or(usize::MAX),
            schema: self.schema,
            rows_read: 0,
            _cat_lock,
        })
    }
}

pub struct BatchedCsvReader<'a> {
    reader_bytes: ReaderBytes<'a>,
    parse_options: CsvParseOptions,
    chunk_size: usize,
    file_chunks_iter: ChunkOffsetIter<'a>,
    file_chunks: Vec<(usize, usize)>,
    projection: Vec<usize>,
    starting_point_offset: Option<usize>,
    row_index: Option<RowIndex>,
    null_values: Option<NullValuesCompiled>,
    to_cast: Vec<Field>,
    ignore_errors: bool,
    remaining: usize,
    schema: SchemaRef,
    rows_read: IdxSize,
    #[cfg(feature = "dtype-categorical")]
    _cat_lock: Option<polars_core::StringCacheHolder>,
    #[cfg(not(feature = "dtype-categorical"))]
    _cat_lock: Option<u8>,
}

impl BatchedCsvReader<'_> {
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
                        &self.parse_options,
                        self.schema.as_ref(),
                        self.ignore_errors,
                        &self.projection,
                        bytes_offset_thread,
                        self.chunk_size,
                        self.null_values.as_ref(),
                        usize::MAX,
                        stop_at_nbytes,
                        self.starting_point_offset,
                    )?;

                    cast_columns(&mut df, &self.to_cast, false, self.ignore_errors)?;

                    if let Some(rc) = &self.row_index {
                        unsafe { df.with_row_index_mut(rc.name.clone(), Some(rc.offset)) };
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

pub fn to_batched_owned(
    mut reader: CsvReader<Box<dyn MmapBytesReader>>,
) -> PolarsResult<OwnedBatchedCsvReader> {
    let batched_reader = reader.batched_borrowed()?;
    let schema = batched_reader.schema.clone();
    // If you put a drop(reader) here, rust will complain that reader is borrowed,
    // so we presumably have to keep ownership of it to maintain the safety of the
    // 'static transmute.
    let batched_reader: BatchedCsvReader<'static> = unsafe { std::mem::transmute(batched_reader) };

    Ok(OwnedBatchedCsvReader {
        schema,
        batched_reader,
        _reader: reader,
    })
}
