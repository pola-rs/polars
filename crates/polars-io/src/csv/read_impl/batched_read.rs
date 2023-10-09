use std::collections::VecDeque;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use super::*;
use crate::csv::CsvReader;
use crate::mmap::MmapBytesReader;
use crate::prelude::update_row_counts2;

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

    fn return_slice(&self, start: usize, end: usize) -> (usize, usize) {
        let slice = &self.buf[start..end];
        let len = slice.len();
        (slice.as_ptr() as usize, len)
    }

    fn get_buf(&self) -> (usize, usize) {
        let slice = &self.buf[self.buf_end..];
        let len = slice.len();
        (slice.as_ptr() as usize, len)
    }

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

        Ok(BatchedCsvReaderRead {
            chunk_size: self.chunk_size,
            finished: false,
            file_chunk_reader: chunk_iter,
            file_chunks: vec![],
            str_capacities: self.init_string_size_stats(&str_columns, self.chunk_size),
            str_columns,
            projection,
            starting_point_offset,
            row_count: self.row_count,
            comment_char: self.comment_char,
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

pub struct BatchedCsvReaderRead<'a> {
    chunk_size: usize,
    finished: bool,
    file_chunk_reader: ChunkReader<'a>,
    file_chunks: Vec<(usize, usize)>,
    str_capacities: Vec<RunningSize>,
    str_columns: StringColumns,
    projection: Vec<usize>,
    starting_point_offset: Option<usize>,
    row_count: Option<RowCount>,
    comment_char: Option<u8>,
    quote_char: Option<u8>,
    eol_char: u8,
    null_values: Option<NullValuesCompiled>,
    missing_is_null: bool,
    to_cast: Vec<Field>,
    ignore_errors: bool,
    truncate_ragged_lines: bool,
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
//
impl<'a> BatchedCsvReaderRead<'a> {
    pub fn next_batches(&mut self, n: usize) -> PolarsResult<Option<Vec<DataFrame>>> {
        if n == 0 || self.finished {
            return Ok(None);
        }
        if let Some(n_rows) = self.n_rows {
            if self.rows_read >= n_rows as IdxSize {
                return Ok(None);
            }
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
            self.file_chunks.push(self.file_chunk_reader.get_buf());
            self.finished = true
        }

        // depleted the offsets iterator, we are done as well.
        if self.file_chunks.is_empty() {
            return Ok(None);
        }

        let mut chunks = POOL.install(|| {
            self.file_chunks
                .par_iter()
                .map(|(ptr, len)| {
                    let chunk = unsafe { std::slice::from_raw_parts(*ptr as *const u8, *len) };
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
                        self.comment_char,
                        self.chunk_size,
                        &self.str_capacities,
                        self.encoding,
                        self.null_values.as_ref(),
                        self.missing_is_null,
                        self.truncate_ragged_lines,
                        self.chunk_size,
                        stop_at_n_bytes,
                        self.starting_point_offset,
                    )?;

                    cast_columns(&mut df, &self.to_cast, false, self.ignore_errors)?;

                    update_string_stats(&self.str_capacities, &self.str_columns, &df)?;
                    if let Some(rc) = &self.row_count {
                        df.with_row_count_mut(&rc.name, Some(rc.offset));
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

pub struct OwnedBatchedCsvReader {
    #[allow(dead_code)]
    // this exist because we need to keep ownership
    schema: SchemaRef,
    reader: *mut CsvReader<'static, Box<dyn MmapBytesReader>>,
    batched_reader: *mut BatchedCsvReaderRead<'static>,
}

unsafe impl Send for OwnedBatchedCsvReader {}
unsafe impl Sync for OwnedBatchedCsvReader {}

impl OwnedBatchedCsvReader {
    pub fn next_batches(&mut self, n: usize) -> PolarsResult<Option<Vec<DataFrame>>> {
        let reader = unsafe { &mut *self.batched_reader };
        reader.next_batches(n)
    }
}

impl Drop for OwnedBatchedCsvReader {
    fn drop(&mut self) {
        // release heap allocated
        unsafe {
            let _to_drop = Box::from_raw(self.batched_reader);
            let _to_drop = Box::from_raw(self.reader);
        };
    }
}

pub fn to_batched_owned_read(
    reader: CsvReader<'_, Box<dyn MmapBytesReader>>,
    schema: SchemaRef,
) -> OwnedBatchedCsvReader {
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
    let batched_reader = unsafe { Box::new((*reader).batched_borrowed_read().unwrap()) };
    let batched_reader = Box::leak(batched_reader) as *mut BatchedCsvReaderRead;

    OwnedBatchedCsvReader {
        schema,
        reader,
        batched_reader,
    }
}

#[cfg(test)]
mod test {
    use polars_core::utils::concat_df;

    use super::*;
    use crate::SerReader;

    #[test]
    fn test_read_io_reader() {
        let path = "../../examples/datasets/foods1.csv";
        let file = std::fs::File::open(path).unwrap();
        let mut reader = CsvReader::from_path(path).unwrap().with_chunk_size(5);

        let mut reader = reader.batched_borrowed_read().unwrap();
        let batches = reader.next_batches(5).unwrap().unwrap();
        assert_eq!(batches.len(), 5);
        let df = concat_df(&batches).unwrap();
        let expected = CsvReader::new(file).finish().unwrap();
        assert!(df.frame_equal(&expected))
    }
}
