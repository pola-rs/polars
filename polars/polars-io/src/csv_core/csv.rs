use crate::csv::CsvEncoding;
use crate::csv_core::chunked_parser::{
    add_to_builders_core, finish_builder, init_builders, next_rows_core,
};
use crate::csv_core::utils::*;
use crate::csv_core::{buffer::*, parser::*};
use crate::PhysicalIOExpr;
use crate::ScanAggregation;
use crossbeam::thread;
use crossbeam::thread::ScopedJoinHandle;
use csv::ByteRecordsIntoIter;
use polars_core::prelude::*;
use rayon::prelude::*;
use std::fmt;
use std::io::{Read, Seek};
use std::sync::Arc;

/// Is multiplied with batch_size to determine capacity of builders
const CAPACITY_MULTIPLIER: usize = 512;

/// CSV file reader
pub struct SequentialReader<R: Read> {
    /// Explicit schema for the CSV file
    schema: SchemaRef,
    /// Optional projection for which columns to load (zero-based column indices)
    projection: Option<Vec<usize>>,
    /// File reader
    record_iter: Option<ByteRecordsIntoIter<R>>,
    /// Batch size (number of records to load each time)
    batch_size: usize,
    /// Current line number, used in error reporting
    line_number: usize,
    ignore_parser_errors: bool,
    skip_rows: usize,
    n_rows: Option<usize>,
    encoding: CsvEncoding,
    n_threads: Option<usize>,
    path: Option<String>,
    has_header: bool,
    delimiter: u8,
    sample_size: usize,
    stable_parser: bool
}

impl<R> fmt::Debug for SequentialReader<R>
where
    R: Read,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Reader")
            .field("schema", &self.schema)
            .field("projection", &self.projection)
            .field("batch_size", &self.batch_size)
            .field("line_number", &self.line_number)
            .finish()
    }
}

impl<R: Read + Sync + Send> SequentialReader<R> {
    /// Returns the schema of the reader, useful for getting the schema without reading
    /// record batches
    pub fn schema(&self) -> SchemaRef {
        match &self.projection {
            Some(projection) => {
                let fields = self.schema.fields();
                let projected_fields: Vec<Field> =
                    projection.iter().map(|i| fields[*i].clone()).collect();

                Arc::new(Schema::new(projected_fields))
            }
            None => self.schema.clone(),
        }
    }

    /// Create a new CsvReader from a `BufReader<R: Read>
    ///
    /// This constructor allows you more flexibility in what records are processed by the
    /// csv reader.
    #[allow(clippy::too_many_arguments)]
    pub fn from_reader(
        reader: R,
        schema: SchemaRef,
        has_header: bool,
        delimiter: u8,
        batch_size: usize,
        projection: Option<Vec<usize>>,
        ignore_parser_errors: bool,
        n_rows: Option<usize>,
        skip_rows: usize,
        encoding: CsvEncoding,
        n_threads: Option<usize>,
        path: Option<String>,
        sample_size: usize,
        stable_parser: bool,
    ) -> Self {
        let csv_reader = init_csv_reader(reader, has_header, delimiter);
        let record_iter = Some(csv_reader.into_byte_records());

        Self {
            schema,
            projection,
            record_iter,
            batch_size,
            line_number: if has_header { 1 } else { 0 },
            ignore_parser_errors,
            skip_rows,
            n_rows,
            encoding,
            n_threads,
            path,
            has_header,
            delimiter,
            sample_size,
            stable_parser
        }
    }

    fn find_starting_point<'a>(&self, mut bytes: &'a [u8]) -> Result<&'a [u8]> {
        // Skip all leading white space and the occasional utf8-bom
        bytes = skip_line_ending(skip_whitespace(skip_bom(bytes)).0).0;

        // If there is a header we skip it.
        if self.has_header {
            bytes = skip_header(bytes).0;
        }

        if self.skip_rows > 0 {
            for _ in 0..self.skip_rows {
                let pos = next_line_position(bytes)
                    .ok_or_else(|| PolarsError::NoData("not enough lines to skip".into()))?;
                bytes = &bytes[pos..];
            }
        }
        Ok(bytes)
    }

    fn parse_csv_chunked(
        &mut self,
        predicate: Option<&Arc<dyn PhysicalIOExpr>>,
        aggregate: Option<&[ScanAggregation]>,
        capacity: usize,
        n_threads: usize,
        bytes: &[u8],
    ) -> Result<Vec<DataFrame>> {
        let projection = self
            .projection
            .take()
            .unwrap_or_else(|| (0..self.schema.fields().len()).collect());
        let mut parsed_dfs = Vec::with_capacity(128);

        let bytes = self.find_starting_point(bytes)?;

        let file_chunks = get_file_chunks(bytes, n_threads);

        let scopes: Result<_> = thread::scope(|s| {
            let mut handlers = Vec::with_capacity(n_threads);

            for (thread_no, (mut total_bytes_offset, stop_at_nbytes)) in
                file_chunks.into_iter().enumerate()
            {
                let delimiter = self.delimiter;
                let batch_size = self.batch_size;
                let schema = self.schema.clone();
                let ignore_parser_errors = self.ignore_parser_errors;
                let encoding = self.encoding;
                let projection = &projection;

                let h: ScopedJoinHandle<Result<_>> = s.spawn(move |_| {
                    // container to ammortize allocs
                    let mut rows = Vec::with_capacity(batch_size);
                    rows.resize_with(batch_size, Default::default);

                    let mut builders = init_builders(&projection, capacity, &schema).unwrap();

                    #[cfg(target_os = "linux")]
                    let has_utf8 = builders
                        .iter()
                        .any(|b| matches!(b, super::chunked_parser::Builder::Utf8(_)));

                    let mut local_parsed_dfs = Vec::with_capacity(16);

                    let mut local_bytes;
                    let mut core_reader =
                        csv_core::ReaderBuilder::new().delimiter(delimiter).build();

                    let mut count = 0;
                    loop {
                        count += 1;
                        local_bytes = &bytes[total_bytes_offset..stop_at_nbytes];
                        let (correctly_parsed, bytes_read) =
                            next_rows_core(&mut rows, local_bytes, &mut core_reader, batch_size);
                        total_bytes_offset += bytes_read;

                        if correctly_parsed < batch_size {
                            if correctly_parsed == 0 {
                                break;
                            }
                            // this only happens at the last batch if it doesn't fit a whole batch.
                            rows.truncate(correctly_parsed);
                        }
                        add_to_builders_core(
                            &mut builders,
                            &projection,
                            &rows,
                            &schema,
                            ignore_parser_errors,
                            encoding,
                        )?;

                        if total_bytes_offset >= stop_at_nbytes {
                            break;
                        }

                        if count % CAPACITY_MULTIPLIER == 0 {
                            let mut builders_tmp =
                                init_builders(&projection, capacity, &schema).unwrap();
                            std::mem::swap(&mut builders_tmp, &mut builders);
                            finish_builder(
                                builders_tmp,
                                &mut local_parsed_dfs,
                                predicate,
                                aggregate,
                            )
                            .unwrap();

                            #[cfg(target_os = "linux")]
                            {
                                // linux global allocators don't return freed memory immediately to the OS.
                                // macos and windows return more aggressively.
                                // We choose this location to do trim heap memory as this will be called after CSV read
                                // which may have some over-allocated utf8
                                // This is an expensive operation therefore we don't want to call it too often, and only when
                                // there are utf8 arrays.
                                if has_utf8
                                    && thread_no == 0
                                    && count % (CAPACITY_MULTIPLIER * 16) == 0
                                {
                                    use polars_core::utils::malloc_trim;
                                    unsafe { malloc_trim(0) };
                                }
                            }
                        }
                    }
                    finish_builder(builders, &mut local_parsed_dfs, predicate, aggregate)?;

                    Ok(local_parsed_dfs)
                });
                handlers.push(h)
            }
            for h in handlers {
                let local_parsed_dfs = h.join().expect("thread panicked")?;
                parsed_dfs.extend(local_parsed_dfs.into_iter())
            }

            Ok(())
        })
        .unwrap();
        let _ = scopes?;

        Ok(parsed_dfs)
    }

    fn parse_csv_fast(&mut self, n_threads: usize, bytes: &[u8]) -> Result<DataFrame> {
        // Make the variable mutable so that we can reassign the sliced file to this variable.
        let mut bytes = self.find_starting_point(bytes)?;

        // initial row guess. We use the line statistic to guess the number of rows to allocate
        let mut total_rows = 128;

        // if None, there are less then 128 rows in the file and the statistics don't matter that much
        if let Some((mean, std)) = get_line_stats(bytes, self.sample_size) {
            // x % upper bound of byte length per line assuming normally distributed
            let line_length_upper_bound = mean + 1.1 * std;
            total_rows = (bytes.len() as f32 / (mean - 0.01 * std)) as usize;

            // if we only need to parse n_rows,
            // we first try to use the line statistics the total bytes we need to process
            if let Some(n_rows) = self.n_rows {
                total_rows = std::cmp::min(n_rows, total_rows);

                // the guessed upper bound of  the no. of bytes in the file
                let n_bytes = (line_length_upper_bound * (n_rows as f32)) as usize;

                if n_bytes < bytes.len() {
                    if let Some(pos) = next_line_position(&bytes[n_bytes..]) {
                        bytes = &bytes[..n_bytes + pos]
                    }
                }
            }
        }

        let projection = self
            .projection
            .take()
            .unwrap_or_else(|| (0..self.schema.fields().len()).collect());

        // split the file by the nearest new line characters such that every thread processes
        // approximately the same number of rows.
        let file_chunks = get_file_chunks(bytes, n_threads);
        let local_capacity = total_rows / n_threads;

        let scopes: Result<_> = thread::scope(|s| {
            let mut handlers = Vec::with_capacity(n_threads);

            for (bytes_offset_thread, stop_at_nbytes) in file_chunks {
                let delimiter = self.delimiter;
                let schema = self.schema.clone();
                let ignore_parser_errors = self.ignore_parser_errors;
                let projection = &projection;

                let h: ScopedJoinHandle<Result<_>> = s.spawn(move |_| {
                    let mut buffers = init_buffers(&projection, local_capacity, &schema)?;
                    let local_bytes = &bytes[bytes_offset_thread..stop_at_nbytes];
                    let read = bytes_offset_thread;

                    parse_lines(
                        local_bytes,
                        read,
                        delimiter,
                        projection,
                        &mut buffers,
                        ignore_parser_errors,
                    )?;
                    Ok(buffers)
                });
                handlers.push(h)
            }

            let mut buffers = Vec::with_capacity(handlers.len());
            for h in handlers {
                let local_buffer = h.join().expect("thread panicked")?;
                buffers.push(local_buffer);
            }

            Ok(buffers)
        })
        .unwrap();
        // all the buffers returned from the threads
        // Structure:
        //      the inner vec has got buffers from all the columns.
        let mut buffers = scopes?;

        // restructure the buffers so that they can be dropped as soon as processed;
        // Structure:
        //      the inner vec has got buffers from a single column
        let buffers = projection
            .iter()
            .map(|&idx| {
                buffers
                    .iter_mut()
                    .map(|buffers| std::mem::take(&mut buffers[idx]))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let columns = buffers
            .into_par_iter()
            .enumerate()
            .map(|(idx, buffers)| {
                let iter = buffers.into_iter();
                let mut s = buffers_to_series(iter, bytes, self.ignore_parser_errors)?;
                let name = self.schema.field(idx).unwrap().name();
                s.rename(name);
                Ok(s)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(DataFrame::new_no_checks(columns))
    }

    /// Read the csv into a DataFrame. The predicate can come from a lazy physical plan.
    pub fn as_df(
        &mut self,
        predicate: Option<Arc<dyn PhysicalIOExpr>>,
        aggregate: Option<&[ScanAggregation]>,
    ) -> Result<DataFrame> {
        let n_threads = self.n_threads.unwrap_or_else(num_cpus::get);

        let mut df = if predicate.is_some() || self.stable_parser {
            let mut capacity = self.batch_size * CAPACITY_MULTIPLIER;
            if let Some(n) = self.n_rows {
                self.batch_size = std::cmp::min(self.batch_size, n);
                capacity = std::cmp::min(n, capacity);
            }

            let path = self.path.as_ref().unwrap();
            let file = std::fs::File::open(path).expect("path should be set");
            let mmap = unsafe { memmap::Mmap::map(&file).unwrap() };
            let bytes = mmap[..].as_ref();

            let parsed_dfs =
                self.parse_csv_chunked(predicate.as_ref(), aggregate, capacity, n_threads, bytes)?;
            polars_core::utils::accumulate_dataframes_vertical(parsed_dfs)?
        } else {
            match (&self.path, self.record_iter.is_some()) {
                (Some(p), _) => {
                    let file = std::fs::File::open(p).unwrap();
                    let mmap = unsafe { memmap::Mmap::map(&file).unwrap() };
                    let bytes = mmap[..].as_ref();
                    self.parse_csv_fast(n_threads, bytes)?
                }
                (None, true) => {
                    let mut r = std::mem::take(&mut self.record_iter).unwrap().into_reader();
                    let mut bytes = Vec::with_capacity(1024 * 128);
                    r.get_mut().read_to_end(&mut bytes)?;
                    if bytes[bytes.len() - 1] != b'\n' || bytes[bytes.len() - 1] != b'\r' {
                        bytes.push(b'\n')
                    }
                    self.parse_csv_fast(n_threads, &bytes)?
                }
                _ => return Err(PolarsError::Other("file or reader must be set".into())),
            }
        };

        if let Some(aggregate) = aggregate {
            let cols = aggregate
                .iter()
                .map(|scan_agg| scan_agg.finish(&df).unwrap())
                .collect();
            df = DataFrame::new_no_checks(cols)
        }

        // if multi-threaded the n_rows was probabilistically determined.
        // Let's slice to correct number of rows if possible.
        if let Some(n_rows) = self.n_rows {
            if n_rows < df.height() {
                df = df.slice(0, n_rows).unwrap()
            }
        }
        Ok(df)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_csv_reader<R: 'static + Read + Seek + Sync + Send>(
    mut reader: R,
    n_rows: Option<usize>,
    skip_rows: usize,
    mut projection: Option<Vec<usize>>,
    batch_size: usize,
    max_records: Option<usize>,
    delimiter: Option<u8>,
    has_header: bool,
    ignore_parser_errors: bool,
    schema: Option<SchemaRef>,
    columns: Option<Vec<String>>,
    encoding: CsvEncoding,
    n_threads: Option<usize>,
    path: Option<String>,
    schema_overwrite: Option<&Schema>,
    sample_size: usize,
    stable_parser: bool
) -> Result<SequentialReader<R>> {
    // check if schema should be inferred
    let delimiter = delimiter.unwrap_or(b',');
    let schema = match schema {
        Some(schema) => schema,
        None => {
            let (inferred_schema, _) = infer_file_schema(
                &mut reader,
                delimiter,
                max_records,
                has_header,
                schema_overwrite,
            )?;
            Arc::new(inferred_schema)
        }
    };

    if let Some(cols) = columns {
        let mut prj = Vec::with_capacity(cols.len());
        for col in cols {
            let i = schema.index_of(&col)?;
            prj.push(i);
        }
        projection = Some(prj);
    }

    Ok(SequentialReader::from_reader(
        reader,
        schema,
        has_header,
        delimiter,
        batch_size,
        projection,
        ignore_parser_errors,
        n_rows,
        skip_rows,
        encoding,
        n_threads,
        path,
        sample_size,
        stable_parser
    ))
}
