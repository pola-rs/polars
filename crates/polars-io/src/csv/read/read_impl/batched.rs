use std::collections::VecDeque;
use std::ops::Deref;

use polars_core::POOL;
use polars_core::datatypes::Field;
use polars_core::frame::DataFrame;
use polars_core::schema::SchemaRef;
use polars_error::{PolarsResult, polars_err};
use polars_utils::IdxSize;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use super::{CoreReader, CountLines, cast_columns, read_chunk};
use crate::RowIndex;
use crate::csv::read::CsvReader;
use crate::csv::read::options::{BatchSizeOptions, NullValuesCompiled};
use crate::mmap::{MmapBytesReader, ReaderBytes};
use crate::prelude::{CsvParseOptions, update_row_counts2};

#[allow(clippy::too_many_arguments)]
pub(crate) fn get_file_chunks_iterator(
    offsets: &mut VecDeque<(usize, usize, usize)>,
    last_pos: &mut usize,
    n_chunks: usize,
    chunk_size: &mut usize,
    bytes: &[u8],
    quote_char: Option<u8>,
    eol_char: u8,
    n_rows: Option<usize>,
    chunk_size_strict: bool,
) -> PolarsResult<()> {
    let cl = CountLines::new(quote_char, eol_char);

    for _ in 0..n_chunks {
        let bytes = &bytes[*last_pos..];

        if bytes.is_empty() {
            break;
        }

        let position;
        let count;

        loop {
            let b = &bytes[..(*chunk_size).min(bytes.len())];
            let (count_, position_) = if let Some(n_rows) = n_rows {
                cl.count_at_most_n_rows(b, n_rows)
            } else {
                cl.count(b)
            };

            let (count_, position_) = if b.len() == bytes.len() {
                (if count_ != 0 { count_ } else { 1 }, b.len())
            } else {
                (
                    count_,
                    if position_ < b.len() {
                        // 1+ for the '\n'
                        1 + position_
                    } else {
                        position_
                    },
                )
            };

            if count_ == 0 {
                if chunk_size_strict {
                    return Err(
                        polars_err!(ComputeError: "A chunk of specified size contained zero full CSV lines, and strict mode was used. Impossible to read the CSV file."),
                    );
                }
                *chunk_size *= 2;
                continue;
            } else if b.len() < bytes.len() {
                if let Some(n_rows) = n_rows {
                    if count_ < n_rows {
                        *chunk_size *= 2;
                        continue;
                    }
                }
            }

            position = position_;
            count = count_;
            break;
        }

        offsets.push_back((*last_pos, *last_pos + position, count));
        *last_pos += position;
    }

    Ok(())
}

struct ChunkOffsetIter<'a> {
    bytes: &'a [u8],
    // (begin, end, number of lines)
    offsets: VecDeque<(usize, usize, usize)>,
    last_offset: usize,
    n_chunks: usize,
    chunk_size: usize,
    quote_char: Option<u8>,
    eol_char: u8,
    chunk_size_strict: bool,
}

impl Iterator for ChunkOffsetIter<'_> {
    // (begin, end, number of lines)
    type Item = PolarsResult<(usize, usize, usize)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.offsets.pop_front() {
            Some(offsets) => Some(Ok(offsets)),
            None => {
                if self.last_offset == self.bytes.len() {
                    return None;
                }
                if let Err(e) = get_file_chunks_iterator(
                    &mut self.offsets,
                    &mut self.last_offset,
                    self.n_chunks,
                    &mut self.chunk_size,
                    self.bytes,
                    self.quote_char,
                    self.eol_char,
                    None,
                    self.chunk_size_strict,
                ) {
                    return Some(Err(e));
                }

                match self.offsets.pop_front() {
                    Some(offsets) => Some(Ok(offsets)),
                    // We depleted the iterator. Ensure we deplete the slice as well
                    None => {
                        let cl = CountLines::new(self.quote_char, self.eol_char);
                        let (count, _) = cl.count(&self.bytes[self.last_offset..]);
                        let out = Some(Ok((self.last_offset, self.bytes.len(), count)));
                        self.last_offset = self.bytes.len();
                        out
                    },
                }
            },
        }
    }
}

// ChunkOffsetReader was created because when using `BatchSizeOptions::TotalNextBatchesNRows`,
// the full offset scanning settings are only known after `BatchedCsvReader::next_batches` is
// called, so an iterator model is not a good fit. On the other had, when using
// `BatchSizeOptions::EachBatchNBytes`, `ChunkOffsetIter` is more suitable. That's why there are
// two options (tied together by `ChunkOffsetScanner`).
struct ChunkOffsetReader<'a> {
    bytes: &'a [u8],
    offsets: VecDeque<(usize, usize, usize)>,
    last_offset: usize,
    chunk_size: usize,
    quote_char: Option<u8>,
    eol_char: u8,
    batches_n_rows: Vec<usize>,
}

impl ChunkOffsetReader<'_> {
    fn n_batches_n_rows(
        &mut self,
        n_batches: usize,
        n_rows: usize,
        out: &mut Vec<(usize, usize, usize)>,
    ) -> PolarsResult<()> {
        self.batches_n_rows.resize(n_batches, 0);
        self.batches_n_rows.fill(n_rows / n_batches);

        let missing = n_rows - (n_rows / n_batches) * n_batches;
        for batch_n_rows in self.batches_n_rows.iter_mut().take(missing) {
            *batch_n_rows += 1;
        }

        for batch_n_rows in self.batches_n_rows.iter().copied() {
            get_file_chunks_iterator(
                &mut self.offsets,
                &mut self.last_offset,
                1,
                &mut self.chunk_size,
                self.bytes,
                self.quote_char,
                self.eol_char,
                Some(batch_n_rows),
                false,
            )?;
        }

        while let Some(offset) = self.offsets.pop_front() {
            out.push(offset);
        }

        Ok(())
    }
}

// A helper used to perform the initial scanning to determine byte ranges
// of CSV rows. There are two variants, because each of them is suitable for
// different values of `BatchSizeOptions`
enum ChunkOffsetScanner<'a> {
    Iter(ChunkOffsetIter<'a>),
    Reader(ChunkOffsetReader<'a>),
}

impl<'a> ChunkOffsetScanner<'a> {
    fn from_options(
        bytes: &'a [u8],
        n_threads: usize,
        quote_char: Option<u8>,
        eol_char: u8,
        options: BatchSizeOptions,
    ) -> Self {
        // this is arbitrarily chosen.
        // we don't want this to depend on the thread pool size
        // otherwise the chunks are not deterministic
        let offset_batch_size = 16;

        // Copied from [`Self::parse_csv`]
        let n_parts_hint = n_threads * 16;
        let chunk_size = std::cmp::min(bytes.len() / n_parts_hint, 16 * 1024 * 1024);

        // Use a small min chunk size to catch failures in tests.
        #[cfg(debug_assertions)]
        let min_chunk_size = 64;
        #[cfg(not(debug_assertions))]
        let min_chunk_size = 1024 * 4;

        let chunk_size = std::cmp::max(chunk_size, min_chunk_size);

        match options {
            BatchSizeOptions::UseDefault => Self::Iter(ChunkOffsetIter {
                bytes,
                offsets: VecDeque::with_capacity(offset_batch_size),
                last_offset: 0,
                n_chunks: offset_batch_size,
                chunk_size,
                quote_char,
                eol_char,
                chunk_size_strict: false,
            }),
            BatchSizeOptions::EachBatchNBytes(n) => Self::Iter(ChunkOffsetIter {
                bytes,
                offsets: VecDeque::with_capacity(offset_batch_size),
                last_offset: 0,
                n_chunks: offset_batch_size,
                chunk_size: n,
                quote_char,
                eol_char,
                chunk_size_strict: false,
            }),
            BatchSizeOptions::EachBatchNBytesStrict(n) => Self::Iter(ChunkOffsetIter {
                bytes,
                offsets: VecDeque::with_capacity(offset_batch_size),
                last_offset: 0,
                n_chunks: offset_batch_size,
                chunk_size: n,
                quote_char,
                eol_char,
                chunk_size_strict: true,
            }),
            BatchSizeOptions::EachBatchNRows(_) => Self::Reader(ChunkOffsetReader {
                bytes,
                offsets: VecDeque::with_capacity(offset_batch_size),
                last_offset: 0,
                chunk_size,
                quote_char,
                eol_char,
                batches_n_rows: Vec::with_capacity(offset_batch_size),
            }),
            BatchSizeOptions::TotalNextBatchesNRows(_) => Self::Reader(ChunkOffsetReader {
                bytes,
                offsets: VecDeque::with_capacity(offset_batch_size),
                last_offset: 0,
                chunk_size,
                quote_char,
                eol_char,
                batches_n_rows: Vec::with_capacity(offset_batch_size),
            }),
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

        // extend lifetime. It is bound to `readerbytes` and we keep track of that
        // lifetime so this is sound.
        let bytes = unsafe { std::mem::transmute::<&[u8], &'static [u8]>(bytes) };

        let file_chunks_scanner = ChunkOffsetScanner::from_options(
            bytes,
            n_threads,
            self.parse_options.quote_char,
            self.parse_options.eol_char,
            self.batch_size_options.clone(),
        );

        let projection = self.get_projection()?;

        Ok(BatchedCsvReader {
            reader_bytes,
            parse_options: self.parse_options,
            batch_size_options: self.batch_size_options.clone(),
            file_chunks_scanner,
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
        })
    }
}

pub struct BatchedCsvReader<'a> {
    reader_bytes: ReaderBytes<'a>,
    parse_options: CsvParseOptions,
    batch_size_options: BatchSizeOptions,
    file_chunks_access: ChunkOffsetScanner<'a>,
    file_chunks: Vec<(usize, usize, usize)>,
    projection: Vec<usize>,
    starting_point_offset: Option<usize>,
    row_index: Option<RowIndex>,
    null_values: Option<NullValuesCompiled>,
    to_cast: Vec<Field>,
    ignore_errors: bool,
    remaining: usize,
    schema: SchemaRef,
    rows_read: IdxSize,
}

impl BatchedCsvReader<'_> {
    pub fn next_batches(&mut self, n: usize) -> PolarsResult<Option<Vec<DataFrame>>> {
        if n == 0 || self.remaining == 0 {
            return Ok(None);
        }

        // The CSV file is first pre-scanned using `self.file_chunks_scanner` to determine
        // the byte ranges of CSV rows. This procedure is influenced by `self.batch_size_options`
        // in two ways: `CoreReader::batched` sets `self.file_chunks_scanner` to a `ChunkOffsetAccess`
        // instance that is suitable for given `batch_size_options`, and the options are further
        // used here to perform the correct action with `self.file_chunks_iter`

        let chunks = match &mut self.file_chunks_access {
            ChunkOffsetScanner::Iter(chunk_offset_iter) => {
                // get next `n` offset positions.
                let file_chunks_iter = chunk_offset_iter.take(n);
                for file_chunk in file_chunks_iter {
                    match file_chunk {
                        Ok(chunk) => self.file_chunks.push(chunk),
                        Err(e) => return Err(e),
                    }
                }
                // depleted the offsets iterator, we are done as well.
                if self.file_chunks.is_empty() {
                    return Ok(None);
                }
                &self.file_chunks
            },
            ChunkOffsetScanner::Reader(chunk_offset_reader) => {
                let n_rows = match self.batch_size_options {
                    BatchSizeOptions::TotalNextBatchesNRows(n) => n,
                    BatchSizeOptions::EachBatchNRows(n_) => n * n_,
                    _ => panic!(
                        "ChunkOffsetsAccess::Reader is only created with BatchSizeOptions::TotalNextBatchesNRows or BatchSizeOptions::EachBatchNRows."
                    ),
                };
                self.file_chunks.clear();
                chunk_offset_reader.n_batches_n_rows(n, n_rows, &mut self.file_chunks)?;
                // depleted the offsets iterator, we are done as well.
                if self.file_chunks.is_empty() {
                    return Ok(None);
                }
                &self.file_chunks
            },
        };

        let mut bytes = self.reader_bytes.deref();
        if let Some(pos) = self.starting_point_offset {
            bytes = &bytes[pos..];
        }

        let mut chunks = POOL.install(|| {
            chunks
                .into_par_iter()
                .copied()
                .map(|(bytes_offset_thread, stop_at_nbytes, lines_count)| {
                    let mut df = read_chunk(
                        bytes,
                        &self.parse_options,
                        self.schema.as_ref(),
                        self.ignore_errors,
                        &self.projection,
                        bytes_offset_thread,
                        // Use the actual number of lines for buffers capacity
                        lines_count,
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
