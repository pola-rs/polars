use std::collections::VecDeque;

use super::*;
use crate::csv::CsvReader;
use crate::mmap::MmapBytesReader;
use crate::prelude::update_row_counts2;

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
    pub fn batched_mmap(mut self, _has_cat: bool) -> PolarsResult<BatchedCsvReaderMmap<'a>> {
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

        let projection = self.get_projection();

        let str_columns = self.get_string_columns(&projection)?;

        // RAII structure that will ensure we maintain a global stringcache
        #[cfg(feature = "dtype-categorical")]
        let _cat_lock = if _has_cat {
            Some(polars_core::StringCacheHolder::hold())
        } else {
            None
        };

        #[cfg(not(feature = "dtype-categorical"))]
        let _cat_lock = None;

        Ok(BatchedCsvReaderMmap {
            reader_bytes,
            chunk_size: self.chunk_size,
            file_chunks_iter: file_chunks,
            file_chunks: vec![],
            str_capacities: self.init_string_size_stats(&str_columns, self.chunk_size),
            str_columns,
            projection,
            starting_point_offset,
            row_count: self.row_count,
            comment_prefix: self.comment_prefix,
            quote_char: self.quote_char,
            eol_char: self.eol_char,
            null_values: self.null_values,
            missing_is_null: self.missing_is_null,
            to_cast: self.to_cast,
            ignore_errors: self.ignore_errors,
            truncate_ragged_lines: self.truncate_ragged_lines,
            n_rows: self.n_rows,
            encoding: self.encoding,
            separator: self.separator,
            schema: self.schema,
            rows_read: 0,
            _cat_lock,
        })
    }
}

pub struct BatchedCsvReaderMmap<'a> {
    reader_bytes: ReaderBytes<'a>,
    chunk_size: usize,
    file_chunks_iter: ChunkOffsetIter<'a>,
    file_chunks: Vec<(usize, usize)>,
    str_capacities: Vec<RunningSize>,
    str_columns: StringColumns,
    projection: Vec<usize>,
    starting_point_offset: Option<usize>,
    row_count: Option<RowIndex>,
    comment_prefix: Option<CommentPrefix>,
    quote_char: Option<u8>,
    eol_char: u8,
    null_values: Option<NullValuesCompiled>,
    missing_is_null: bool,
    truncate_ragged_lines: bool,
    to_cast: Vec<Field>,
    ignore_errors: bool,
    n_rows: Option<usize>,
    encoding: CsvEncoding,
    separator: u8,
    schema: SchemaRef,
    rows_read: IdxSize,
    #[cfg(feature = "dtype-categorical")]
    _cat_lock: Option<polars_core::StringCacheHolder>,
    #[cfg(not(feature = "dtype-categorical"))]
    _cat_lock: Option<u8>,
}

impl<'a> BatchedCsvReaderMmap<'a> {
    pub fn next_batches(&mut self, n: usize) -> PolarsResult<Option<Vec<DataFrame>>> {
        if n == 0 {
            return Ok(None);
        }
        if let Some(n_rows) = self.n_rows {
            if self.rows_read >= n_rows as IdxSize {
                return Ok(None);
            }
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
                        &self.str_capacities,
                        self.encoding,
                        self.null_values.as_ref(),
                        self.missing_is_null,
                        self.truncate_ragged_lines,
                        self.chunk_size,
                        stop_at_nbytes,
                        self.starting_point_offset,
                    )?;

                    cast_columns(&mut df, &self.to_cast, false, self.ignore_errors)?;

                    update_string_stats(&self.str_capacities, &self.str_columns, &df)?;
                    if let Some(rc) = &self.row_count {
                        df.with_row_index_mut(&rc.name, Some(rc.offset));
                    }
                    Ok(df)
                })
                .collect::<PolarsResult<Vec<_>>>()
        })?;
        self.file_chunks.clear();

        if self.row_count.is_some() {
            update_row_counts2(&mut chunks, self.rows_read)
        }
        for df in &chunks {
            self.rows_read += df.height() as IdxSize;
        }
        Ok(Some(chunks))
    }
}

pub struct OwnedBatchedCsvReaderMmap {
    #[allow(dead_code)]
    // this exist because we need to keep ownership
    schema: SchemaRef,
    reader: *mut CsvReader<'static, Box<dyn MmapBytesReader>>,
    batched_reader: *mut BatchedCsvReaderMmap<'static>,
}

unsafe impl Send for OwnedBatchedCsvReaderMmap {}
unsafe impl Sync for OwnedBatchedCsvReaderMmap {}

impl OwnedBatchedCsvReaderMmap {
    pub fn next_batches(&mut self, n: usize) -> PolarsResult<Option<Vec<DataFrame>>> {
        let reader = unsafe { &mut *self.batched_reader };
        reader.next_batches(n)
    }
}

impl Drop for OwnedBatchedCsvReaderMmap {
    fn drop(&mut self) {
        // release heap allocated
        unsafe {
            let _to_drop = Box::from_raw(self.batched_reader);
            let _to_drop = Box::from_raw(self.reader);
        };
    }
}

pub fn to_batched_owned_mmap(
    reader: CsvReader<'_, Box<dyn MmapBytesReader>>,
    schema: SchemaRef,
) -> OwnedBatchedCsvReaderMmap {
    // make sure that the schema is bound to the schema we have
    // we will keep ownership of the schema so that the lifetime remains bound to ourselves
    let reader = reader.with_schema(Some(schema.clone()));
    // extend the lifetime
    // the lifetime was bound to schema, which we own and will store on the heap
    let reader = unsafe {
        std::mem::transmute::<
            CsvReader<'_, Box<dyn MmapBytesReader>>,
            CsvReader<'static, Box<dyn MmapBytesReader>>,
        >(reader)
    };
    let reader = Box::new(reader);

    let reader = Box::leak(reader) as *mut CsvReader<'static, Box<dyn MmapBytesReader>>;
    let batched_reader = unsafe { Box::new((*reader).batched_borrowed_mmap().unwrap()) };
    let batched_reader = Box::leak(batched_reader) as *mut BatchedCsvReaderMmap;

    OwnedBatchedCsvReaderMmap {
        schema,
        reader,
        batched_reader,
    }
}
