use crate::csv::{CsvEncoding, NullValues};
use crate::csv_core::utils::*;
use crate::csv_core::{buffer::*, parser::*};
use crate::mmap::MmapBytesReader;
use crate::utils::resolve_homedir;
use crate::PhysicalIoExpr;
use crate::ScanAggregation;
use polars_arrow::array::*;
use polars_core::utils::accumulate_dataframes_vertical;
use polars_core::{prelude::*, POOL};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::borrow::Cow;
use std::fmt;
#[cfg(feature = "decompress")]
use std::io::SeekFrom;
use std::io::{Read, Seek};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{atomic::AtomicUsize, Arc};

/// CSV file reader
pub struct CoreReader<R: Read + MmapBytesReader> {
    /// Explicit schema for the CSV file
    schema: SchemaRef,
    /// Optional projection for which columns to load (zero-based column indices)
    projection: Option<Vec<usize>>,
    /// File reader
    reader: Option<R>,
    /// Current line number, used in error reporting
    line_number: usize,
    ignore_parser_errors: bool,
    skip_rows: usize,
    n_rows: Option<usize>,
    encoding: CsvEncoding,
    n_threads: Option<usize>,
    path: Option<PathBuf>,
    has_header: bool,
    delimiter: u8,
    sample_size: usize,
    chunk_size: usize,
    low_memory: bool,
    comment_char: Option<u8>,
    null_values: Option<Vec<String>>,
    /// If schema was given by user or inferred
    #[cfg(feature = "decompress")]
    inferred_schema: bool,
}

impl<R> fmt::Debug for CoreReader<R>
where
    R: Read + MmapBytesReader,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Reader")
            .field("schema", &self.schema)
            .field("projection", &self.projection)
            .field("line_number", &self.line_number)
            .finish()
    }
}

pub(crate) struct RunningSize {
    max: AtomicUsize,
    sum: AtomicUsize,
    count: AtomicUsize,
    last: AtomicUsize,
}

fn compute_size_hint(max: usize, sum: usize, count: usize, last: usize) -> usize {
    let avg = (sum as f32 / count as f32) as usize;
    let size = std::cmp::max(last, avg) as f32;
    if (max as f32) < (size * 1.5) {
        max
    } else {
        size as usize
    }
}
impl RunningSize {
    fn new(size: usize) -> Self {
        Self {
            max: AtomicUsize::new(size),
            sum: AtomicUsize::new(size),
            count: AtomicUsize::new(1),
            last: AtomicUsize::new(size),
        }
    }

    pub(crate) fn update(&self, size: usize) -> (usize, usize, usize, usize) {
        let max = self.max.fetch_max(size, Ordering::Release);
        let sum = self.sum.fetch_add(size, Ordering::Release);
        let count = self.count.fetch_add(1, Ordering::Release);
        let last = self.last.fetch_add(size, Ordering::Release);
        (
            max,
            sum / count,
            last,
            compute_size_hint(max, sum, count, last),
        )
    }

    pub(crate) fn size_hint(&self) -> usize {
        let max = self.max.load(Ordering::Acquire);
        let sum = self.sum.load(Ordering::Acquire);
        let count = self.count.load(Ordering::Acquire);
        let last = self.last.load(Ordering::Acquire);
        compute_size_hint(max, sum, count, last)
    }
}

impl<R: Read + Sync + Send + MmapBytesReader> CoreReader<R> {
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

    fn find_starting_point<'a>(&self, mut bytes: &'a [u8]) -> Result<&'a [u8]> {
        // Skip all leading white space and the occasional utf8-bom
        bytes = skip_line_ending(skip_whitespace(skip_bom(bytes)).0).0;

        // If there is a header we skip it.
        if self.has_header {
            bytes = skip_header(bytes).0;
        }

        if self.skip_rows > 0 {
            for _ in 0..self.skip_rows {
                // This does not check embedding of new line chars in string quotes.
                // TODO create a state machine/ or use that of csv crate to skip lines with proper
                // escaping
                let pos = next_line_position_naive(bytes)
                    .ok_or_else(|| PolarsError::NoData("not enough lines to skip".into()))?;
                bytes = &bytes[pos..];
            }
        }
        Ok(bytes)
    }

    fn parse_csv(
        &mut self,
        mut n_threads: usize,
        bytes: &[u8],
        predicate: Option<&Arc<dyn PhysicalIoExpr>>,
    ) -> Result<DataFrame> {
        let logging = std::env::var("POLARS_VERBOSE").is_ok();
        // Make the variable mutable so that we can reassign the sliced file to this variable.
        let mut bytes = self.find_starting_point(bytes)?;

        // initial row guess. We use the line statistic to guess the number of rows to allocate
        let mut total_rows = 128;

        // if None, there are less then 128 rows in the file and the statistics don't matter that much
        if let Some((mean, std)) = get_line_stats(bytes, self.sample_size) {
            if logging {
                eprintln!("avg line length: {}\nstd. dev. line length: {}", mean, std);
            }

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
                    if let Some(pos) = next_line_position(
                        &bytes[n_bytes..],
                        self.schema.fields().len(),
                        self.delimiter,
                    ) {
                        bytes = &bytes[..n_bytes + pos]
                    }
                }
            }
            if logging {
                eprintln!("initial row estimate: {}", total_rows)
            }
        }
        if total_rows == 128 {
            n_threads = 1;

            if logging {
                eprintln!("file < 128 rows, no statistics determined")
            }
        }

        // we also need to sort the projection to have predictable output.
        // the `parse_lines` function expects this.
        let projection = self
            .projection
            .take()
            .map(|mut v| {
                v.sort_unstable();
                v
            })
            .unwrap_or_else(|| (0..self.schema.fields().len()).collect());

        let chunk_size = std::cmp::min(self.chunk_size, total_rows);
        let n_chunks = total_rows / chunk_size;
        if logging {
            eprintln!(
                "no. of chunks: {} processed by: {} threads at 1 chunk/thread",
                n_chunks, n_threads
            );
        }

        // keep track of the maximum capacity that needs to be allocated for the utf8-builder
        // Per string column we keep a statistic of the maximum length of string bytes per chunk
        // We must the names, not the indexes, (the indexes are incorrect due to projection
        // pushdown)
        let mut str_columns = Vec::with_capacity(projection.len());
        for i in &projection {
            let fld = self.schema.field(*i).ok_or_else(||
                PolarsError::ValueError(
                    format!("the given projection index: {} is out of bounds for csv schema with {} columns", i, self.schema.len()).into())
                )?;

            if fld.data_type() == &DataType::Utf8 {
                str_columns.push(fld.name())
            }
        }

        // split the file by the nearest new line characters such that every thread processes
        // approximately the same number of rows.
        let file_chunks =
            get_file_chunks(bytes, n_threads, self.schema.fields().len(), self.delimiter);

        // If the number of threads given by the user is lower than our global thread pool we create
        // new one.
        let owned_pool;
        let pool = if POOL.current_num_threads() != n_threads {
            owned_pool = Some(
                ThreadPoolBuilder::new()
                    .num_threads(n_threads)
                    .build()
                    .unwrap(),
            );
            owned_pool.as_ref().unwrap()
        } else {
            &POOL
        };

        // all the buffers returned from the threads
        // Structure:
        //      the inner vec has got buffers from all the columns.
        if predicate.is_some() {
            // assume 10 chars per str
            // this is not updated in low memory mode
            let init_str_bytes = chunk_size * 10;
            let str_capacities: Vec<_> = str_columns
                .iter()
                .map(|_| RunningSize::new(init_str_bytes))
                .collect();

            // An empty file with a schema should return an empty DataFrame with that schema
            if bytes.is_empty() {
                let buffers = init_buffers(
                    &projection,
                    0,
                    &self.schema,
                    &str_capacities,
                    self.delimiter,
                )?;
                let df = DataFrame::new_no_checks(
                    buffers.into_iter().map(|buf| buf.into_series()).collect(),
                );
                return Ok(df);
            }

            let dfs = pool.install(|| {
                file_chunks
                    .into_par_iter()
                    .map(|(bytes_offset_thread, stop_at_nbytes)| {
                        let delimiter = self.delimiter;
                        let schema = self.schema.clone();
                        let ignore_parser_errors = self.ignore_parser_errors;
                        let projection = &projection;

                        let mut read = bytes_offset_thread;
                        let mut df: Option<DataFrame> = None;

                        loop {
                            if read >= stop_at_nbytes {
                                break;
                            }

                            let mut buffers = init_buffers(
                                projection,
                                chunk_size,
                                &schema,
                                &str_capacities,
                                self.delimiter,
                            )?;

                            let local_bytes = &bytes[read..stop_at_nbytes];

                            read = parse_lines(
                                local_bytes,
                                read,
                                delimiter,
                                self.comment_char,
                                self.null_values.as_ref(),
                                projection,
                                &mut buffers,
                                ignore_parser_errors,
                                self.encoding,
                                chunk_size,
                            )?;

                            let mut local_df = DataFrame::new_no_checks(
                                buffers.into_iter().map(|buf| buf.into_series()).collect(),
                            );
                            if let Some(predicate) = predicate {
                                let s = predicate.evaluate(&local_df)?;
                                let mask =
                                    s.bool().expect("filter predicates was not of type boolean");
                                local_df = local_df.filter(mask)?;
                            }

                            // update the running str bytes statistics
                            for (str_index, name) in str_columns.iter().enumerate() {
                                let ca = local_df.column(name)?.utf8()?;
                                let str_bytes_len = ca.get_values_size();

                                // don't update running statistics if we try to reduce string memory usage.
                                if self.low_memory {
                                    local_df.shrink_to_fit();
                                    let (max, avg, last, size_hint) =
                                        str_capacities[str_index].update(str_bytes_len);
                                    if logging {
                                        if size_hint < str_bytes_len {
                                            eprintln!(
                                                "probably needed to reallocate column: {}\
                                    \nprevious capacity was: {}\
                                    \nneeded capacity was: {}",
                                                name, size_hint, str_bytes_len
                                            );
                                        }
                                        eprintln!(
                                            "column {} statistics: \nmax: {}\navg: {}\nlast: {}",
                                            name, max, avg, last
                                        )
                                    }
                                }
                            }
                            match &mut df {
                                None => df = Some(local_df),
                                Some(df) => {
                                    df.vstack_mut(&local_df).unwrap();
                                }
                            }
                        }

                        Ok(df)
                    })
                    .collect::<Result<Vec<_>>>()
            })?;
            accumulate_dataframes_vertical(dfs.into_iter().flatten())
        } else {
            // let exponential growth solve the needed size. This leads to less memory overhead
            // in the later rechunk. Because we have large chunks they are easier reused for the
            // large final contiguous memory needed at the end.
            let rows_per_thread = total_rows / n_threads;
            let max_proxy = bytes.len() / n_threads / 2;
            let capacity = if self.low_memory {
                chunk_size
            } else {
                std::cmp::min(rows_per_thread, max_proxy)
            };

            // assume 10 chars per str
            let init_str_bytes = capacity * 10;
            let str_capacities: Vec<_> = str_columns
                .iter()
                .map(|_| RunningSize::new(init_str_bytes))
                .collect();

            let dfs = pool.install(|| {
                file_chunks
                    .into_par_iter()
                    .map(|(bytes_offset_thread, stop_at_nbytes)| {
                        let delimiter = self.delimiter;
                        let schema = self.schema.clone();
                        let ignore_parser_errors = self.ignore_parser_errors;
                        let projection = &projection;

                        let mut read = bytes_offset_thread;
                        let mut buffers = init_buffers(
                            projection,
                            capacity,
                            &schema,
                            &str_capacities,
                            self.delimiter,
                        )?;

                        loop {
                            if read >= stop_at_nbytes {
                                break;
                            }
                            let local_bytes = &bytes[read..stop_at_nbytes];

                            read = parse_lines(
                                local_bytes,
                                read,
                                delimiter,
                                self.comment_char,
                                self.null_values.as_ref(),
                                projection,
                                &mut buffers,
                                ignore_parser_errors,
                                self.encoding,
                                // chunk size doesn't really matter anymore,
                                // less calls if we increase the size
                                chunk_size * 320000,
                            )?;
                        }
                        Ok(DataFrame::new_no_checks(
                            buffers.into_iter().map(|buf| buf.into_series()).collect(),
                        ))
                    })
                    .collect::<Result<Vec<_>>>()
            })?;
            accumulate_dataframes_vertical(dfs.into_iter())
        }
    }

    #[cfg(feature = "decompress")]
    fn decompress_and_parse(
        &mut self,
        predicate: Option<Arc<dyn PhysicalIoExpr>>,
        n_threads: usize,
        bytes: &[u8],
    ) -> Result<DataFrame> {
        if let Some(bytes) = decompress(bytes) {
            if self.inferred_schema {
                self.schema = bytes_to_schema(
                    &bytes,
                    self.delimiter,
                    self.has_header,
                    self.skip_rows,
                    self.comment_char,
                )?;
            }

            self.parse_csv(n_threads, &bytes, predicate.as_ref())
        } else {
            self.parse_csv(n_threads, bytes, predicate.as_ref())
        }
    }

    #[cfg(feature = "decompress")]
    fn decompress<'a>(&mut self, bytes: &'a [u8]) -> Result<Cow<'a, [u8]>> {
        if let Some(bytes) = decompress(bytes) {
            if self.inferred_schema {
                self.schema = bytes_to_schema(
                    &bytes,
                    self.delimiter,
                    self.has_header,
                    self.skip_rows,
                    self.comment_char,
                )?;
            }

            Ok(Cow::Owned(bytes))
        } else {
            Ok(Cow::Borrowed(bytes))
        }
    }

    /// Read the csv into a DataFrame. The predicate can come from a lazy physical plan.
    pub fn as_df(
        &mut self,
        predicate: Option<Arc<dyn PhysicalIoExpr>>,
        aggregate: Option<&[ScanAggregation]>,
    ) -> Result<DataFrame> {
        let n_threads = self.n_threads.unwrap_or_else(|| POOL.current_num_threads());

        let mut df = match &self.path {
            // we have a path so we can mmap
            Some(p) => {
                let p = resolve_homedir(p);
                let file = std::fs::File::open(&p)?;
                let mmap = unsafe { memmap::Mmap::map(&file)? };
                let bytes = mmap[..].as_ref();

                #[cfg(feature = "decompress")]
                {
                    self.decompress_and_parse(predicate, n_threads, bytes)?
                }

                #[cfg(not(feature = "decompress"))]
                self.parse_csv(n_threads, bytes, predicate.as_ref())?
            }
            None => {
                let mut reader = self.reader.take().unwrap();

                // we have a file so we can mmap
                if let Some(file) = reader.to_file() {
                    let mmap = unsafe { memmap::Mmap::map(file)? };
                    let bytes = mmap[..].as_ref();

                    #[cfg(feature = "decompress")]
                    {
                        self.decompress_and_parse(predicate, n_threads, bytes)?
                    }

                    #[cfg(not(feature = "decompress"))]
                    self.parse_csv(n_threads, bytes, predicate.as_ref())?
                } else {
                    // we can get the bytes for free
                    let bytes = if let Some(bytes) = reader.to_bytes() {
                        #[cfg(feature = "decompress")]
                        {
                            self.decompress(bytes)?
                        }

                        #[cfg(not(feature = "decompress"))]
                        Cow::Borrowed(bytes)
                    } else {
                        // we have to read to an owned buffer to get the bytes.
                        let mut bytes = Vec::with_capacity(1024 * 128);
                        reader.read_to_end(&mut bytes)?;
                        if !bytes.is_empty()
                            && (bytes[bytes.len() - 1] != b'\n' || bytes[bytes.len() - 1] != b'\r')
                        {
                            bytes.push(b'\n')
                        }

                        #[cfg(feature = "decompress")]
                        {
                            match self.decompress(&bytes)? {
                                // always return the owned version
                                Cow::Borrowed(_) => Cow::Owned(bytes),
                                Cow::Owned(bytes) => Cow::Owned(bytes),
                            }
                        }

                        #[cfg(not(feature = "decompress"))]
                        Cow::Owned(bytes)
                    };

                    let out = self.parse_csv(n_threads, &bytes, predicate.as_ref())?;
                    // restore reader for a potential second run
                    self.reader = Some(reader);
                    out
                }
            }
        };

        if let Some(aggregate) = aggregate {
            let cols = aggregate
                .iter()
                .map(|scan_agg| scan_agg.finish(&df))
                .collect::<Result<Vec<_>>>()?;
            df = DataFrame::new_no_checks(cols)
        }

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

#[allow(clippy::too_many_arguments)]
pub fn build_csv_reader<R: Read + Seek + Sync + Send + MmapBytesReader>(
    mut reader: R,
    n_rows: Option<usize>,
    skip_rows: usize,
    mut projection: Option<Vec<usize>>,
    max_records: Option<usize>,
    delimiter: Option<u8>,
    has_header: bool,
    ignore_parser_errors: bool,
    schema: Option<SchemaRef>,
    columns: Option<Vec<String>>,
    encoding: CsvEncoding,
    n_threads: Option<usize>,
    path: Option<PathBuf>,
    schema_overwrite: Option<&Schema>,
    sample_size: usize,
    chunk_size: usize,
    low_memory: bool,
    comment_char: Option<u8>,
    null_values: Option<NullValues>,
) -> Result<CoreReader<R>> {
    // check if schema should be inferred
    let delimiter = delimiter.unwrap_or(b',');

    #[cfg(feature = "decompress")]
    let mut inferred_schema = false;

    let schema = match schema {
        Some(schema) => schema,
        None => {
            #[cfg(feature = "decompress")]
            {
                // We keep track of the inferred schema bool
                // In case the file is compressed this schema inference is wrong and has to be done
                // again after decompression.
                inferred_schema = true;

                const N: usize = 5;
                let mut bytes = [0u8; N];
                reader.read_exact(&mut bytes)?;
                // restore position
                reader.seek(SeekFrom::Current(-(N as i64)))?;
                if decompress(&bytes).is_some() {
                    Arc::new(Schema::default())
                } else {
                    let (inferred_schema, _) = infer_file_schema(
                        &mut reader,
                        delimiter,
                        max_records,
                        has_header,
                        schema_overwrite,
                        skip_rows,
                        comment_char,
                    )?;
                    Arc::new(inferred_schema)
                }
            }
            #[cfg(not(feature = "decompress"))]
            {
                let (inferred_schema, _) = infer_file_schema(
                    &mut reader,
                    delimiter,
                    max_records,
                    has_header,
                    schema_overwrite,
                    skip_rows,
                    comment_char,
                )?;
                Arc::new(inferred_schema)
            }
        }
    };

    let null_values = null_values.map(|nv| nv.process(&schema)).transpose()?;

    if let Some(cols) = columns {
        let mut prj = Vec::with_capacity(cols.len());
        for col in cols {
            let i = schema.index_of(&col)?;
            prj.push(i);
        }
        projection = Some(prj);
    }

    Ok(CoreReader {
        schema,
        projection,
        reader: Some(reader),
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
        chunk_size,
        low_memory,
        comment_char,
        null_values,
        #[cfg(feature = "decompress")]
        inferred_schema,
    })
}
