use polars_core::config::verbose;

use super::*;
use crate::csv::CsvReader;
use crate::mmap::MmapBytesReader;

impl<'a> CoreReader<'a> {
    pub fn batched(mut self, _has_cat: bool) -> PolarsResult<BatchedCsvReader<'a>> {
        let mut n_threads = self.n_threads.unwrap_or_else(|| POOL.current_num_threads());
        let reader_bytes = self.reader_bytes.take().unwrap();
        let logging = verbose();
        let (file_chunks, chunk_size, _total_rows, starting_point_offset, _bytes) = self
            .determine_file_chunks_and_statistics(&mut n_threads, &reader_bytes, logging, true)?;
        let projection = self.get_projection();

        // safety
        // we extend the lifetime because we are sure they are bound
        // to 'a, as the &str refer to the &schema which is bound by 'a
        let str_columns = unsafe {
            std::mem::transmute::<Vec<&str>, Vec<&'a str>>(self.get_string_columns(&projection)?)
        };

        // RAII structure that will ensure we maintain a global stringcache
        #[cfg(feature = "dtype-categorical")]
        let _cat_lock = if _has_cat {
            Some(polars_core::IUseStringCache::new())
        } else {
            None
        };

        #[cfg(not(feature = "dtype-categorical"))]
        let _cat_lock = None;

        Ok(BatchedCsvReader {
            reader_bytes,
            chunk_size,
            file_chunks,
            chunk_offset: 0,
            str_capacities: self.init_string_size_stats(&str_columns, chunk_size),
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
            n_rows: self.n_rows,
            encoding: self.encoding,
            delimiter: self.delimiter,
            schema: self.schema,
            rows_read: 0,
            _cat_lock,
        })
    }
}

pub struct BatchedCsvReader<'a> {
    reader_bytes: ReaderBytes<'a>,
    chunk_size: usize,
    file_chunks: Vec<(usize, usize)>,
    chunk_offset: IdxSize,
    str_capacities: Vec<RunningSize>,
    str_columns: Vec<&'a str>,
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
    n_rows: Option<usize>,
    encoding: CsvEncoding,
    delimiter: u8,
    schema: Cow<'a, Schema>,
    rows_read: IdxSize,
    #[cfg(feature = "dtype-categorical")]
    _cat_lock: Option<polars_core::IUseStringCache>,
    #[cfg(not(feature = "dtype-categorical"))]
    _cat_lock: Option<u8>,
}

impl<'a> BatchedCsvReader<'a> {
    pub fn next_batches(&mut self, n: usize) -> PolarsResult<Option<Vec<(IdxSize, DataFrame)>>> {
        if n == 0 {
            return Ok(None);
        }
        if self.chunk_offset == self.file_chunks.len() as IdxSize {
            return Ok(None);
        }
        if let Some(n_rows) = self.n_rows {
            if self.rows_read >= n_rows as IdxSize {
                return Ok(None);
            }
        }
        let end = std::cmp::min(self.chunk_offset as usize + n, self.file_chunks.len());

        let chunks = &self.file_chunks[self.chunk_offset as usize..end];
        self.chunk_offset = end as IdxSize;
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
                        self.delimiter,
                        self.schema.as_ref(),
                        self.ignore_errors,
                        &self.projection,
                        bytes_offset_thread,
                        self.quote_char,
                        self.eol_char,
                        self.comment_char,
                        self.chunk_size,
                        &self.str_capacities,
                        self.encoding,
                        self.null_values.as_ref(),
                        self.missing_is_null,
                        self.chunk_size,
                        stop_at_nbytes,
                        self.starting_point_offset,
                    )?;

                    cast_columns(&mut df, &self.to_cast, false)?;

                    update_string_stats(&self.str_capacities, &self.str_columns, &df)?;
                    if let Some(rc) = &self.row_count {
                        df.with_row_count_mut(&rc.name, Some(rc.offset));
                    }
                    let n_read = df.height() as IdxSize;
                    Ok((df, n_read))
                })
                .collect::<PolarsResult<Vec<_>>>()
        })?;

        if self.row_count.is_some() {
            update_row_counts(&mut chunks, self.rows_read)
        }
        self.rows_read += chunks[chunks.len() - 1].1;
        Ok(Some(
            chunks
                .into_iter()
                .enumerate()
                .map(|(i, t)| (i as IdxSize + self.chunk_offset, t.0))
                .collect(),
        ))
    }
}

pub struct OwnedBatchedCsvReader {
    #[allow(dead_code)]
    // this exist because we need to keep ownership
    schema: SchemaRef,
    reader: *mut CsvReader<'static, Box<dyn MmapBytesReader>>,
    batched_reader: *mut BatchedCsvReader<'static>,
}

unsafe impl Send for OwnedBatchedCsvReader {}
unsafe impl Sync for OwnedBatchedCsvReader {}

impl OwnedBatchedCsvReader {
    pub fn next_batches(&mut self, n: usize) -> PolarsResult<Option<Vec<(IdxSize, DataFrame)>>> {
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

pub fn to_batched_owned(
    reader: CsvReader<'_, Box<dyn MmapBytesReader>>,
    schema: SchemaRef,
) -> OwnedBatchedCsvReader {
    // make sure that the schema is bound to the schema we have
    // we will keep ownership of the schema so that the lifetime remains bound to ourselves
    let reader = reader.with_schema(schema.as_ref());
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
    let batched_reader = unsafe { Box::new((*reader).batched_borrowed().unwrap()) };
    let batched_reader = Box::leak(batched_reader) as *mut BatchedCsvReader;

    OwnedBatchedCsvReader {
        schema,
        reader,
        batched_reader,
    }
}
