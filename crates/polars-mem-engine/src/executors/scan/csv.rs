use std::path::PathBuf;
use std::sync::Arc;

use polars_core::config;
use polars_core::utils::{
    accumulate_dataframes_vertical, accumulate_dataframes_vertical_unchecked,
};

use super::*;

pub struct CsvExec {
    pub paths: Arc<[PathBuf]>,
    pub file_info: FileInfo,
    pub options: CsvReadOptions,
    pub file_options: FileScanOptions,
    pub predicate: Option<Arc<dyn PhysicalExpr>>,
}

impl CsvExec {
    fn read(&self) -> PolarsResult<DataFrame> {
        let with_columns = self
            .file_options
            .with_columns
            .clone()
            // Interpret selecting no columns as selecting all columns.
            .filter(|columns| !columns.is_empty());

        let n_rows = _set_n_rows_for_scan(self.file_options.n_rows);
        let predicate = self.predicate.clone().map(phys_expr_to_io_expr);
        let options_base = self
            .options
            .clone()
            .with_schema(Some(
                self.file_info.reader_schema.clone().unwrap().unwrap_right(),
            ))
            .with_columns(with_columns)
            .with_rechunk(
                // We rechunk at the end to avoid rechunking multiple times in the
                // case of reading multiple files.
                false,
            )
            .with_row_index(None)
            .with_path::<&str>(None);

        if self.paths.is_empty() {
            let out = if let Some(schema) = options_base.schema {
                DataFrame::from_rows_and_schema(&[], schema.as_ref())?
            } else {
                Default::default()
            };
            return Ok(out);
        }

        let verbose = config::verbose();
        let force_async = config::force_async();
        let run_async = force_async || is_cloud_url(self.paths.first().unwrap());

        if force_async && verbose {
            eprintln!("ASYNC READING FORCED");
        }

        let finish_read =
            |i: usize, options: CsvReadOptions, predicate: Option<Arc<dyn PhysicalIoExpr>>| {
                let path = &self.paths[i];
                let mut df = if run_async {
                    #[cfg(feature = "cloud")]
                    {
                        let file = polars_io::file_cache::FILE_CACHE
                            .get_entry(path.to_str().unwrap())
                            // Safety: This was initialized by schema inference.
                            .unwrap()
                            .try_open_assume_latest()?;
                        let owned = &mut vec![];
                        let mmap = unsafe { memmap::Mmap::map(&file).unwrap() };

                        options
                            .into_reader_with_file_handle(std::io::Cursor::new(unsafe {
                                maybe_decompress_bytes(mmap.as_ref(), owned)
                            }?))
                            ._with_predicate(predicate.clone())
                            .finish()
                    }
                    #[cfg(not(feature = "cloud"))]
                    {
                        panic!("required feature `cloud` is not enabled")
                    }
                } else {
                    let file = polars_utils::open_file(path)?;
                    let mmap = unsafe { memmap::Mmap::map(&file).unwrap() };
                    let owned = &mut vec![];

                    options
                        .into_reader_with_file_handle(std::io::Cursor::new(unsafe {
                            maybe_decompress_bytes(mmap.as_ref(), owned)
                        }?))
                        ._with_predicate(predicate.clone())
                        .finish()
                }?;

                if let Some(col) = &self.file_options.include_file_paths {
                    let path = path.to_str().unwrap();
                    unsafe {
                        df.with_column_unchecked(
                            StringChunked::full(col, path, df.height()).into_series(),
                        )
                    };
                }

                Ok(df)
            };

        let mut df = if n_rows.is_some()
            || (predicate.is_some() && self.file_options.row_index.is_some())
        {
            // Basic sequential read
            // predicate must be done after n_rows and row_index, so we must read sequentially
            if verbose {
                eprintln!("read per-file to apply n_rows or (predicate + row_index)");
            }

            let mut n_rows_read = 0usize;
            let mut out = Vec::with_capacity(self.paths.len());
            // If we have n_rows or row_index then we need to count how many rows we read, so we need
            // to delay applying the predicate.
            let predicate_during_read = predicate
                .clone()
                .filter(|_| n_rows.is_none() && self.file_options.row_index.is_none());

            for i in 0..self.paths.len() {
                let opts = options_base
                    .clone()
                    .with_row_index(self.file_options.row_index.clone().map(|mut ri| {
                        ri.offset += n_rows_read as IdxSize;
                        ri
                    }))
                    .with_n_rows(n_rows.map(|n| n - n_rows_read));

                let mut df = finish_read(i, opts, predicate_during_read.clone())?;

                n_rows_read = n_rows_read.saturating_add(df.height());

                let df = if predicate.is_some() && predicate_during_read.is_none() {
                    let predicate = predicate.clone().unwrap();

                    // We should have a chunked df since we read with rechunk false,
                    // so we parallelize over row-wise batches.
                    // Safety: We can accumulate unchecked here as these DataFrames
                    // all come from the same file.
                    accumulate_dataframes_vertical_unchecked(
                        POOL.install(|| {
                            df.split_chunks()
                                .collect::<Vec<_>>()
                                .into_par_iter()
                                .map(|df| {
                                    let s = predicate.evaluate_io(&df)?;
                                    let mask = s
                                        .bool()
                                        .expect("filter predicates was not of type boolean");
                                    df.filter(mask)
                                })
                                .collect::<PolarsResult<Vec<_>>>()
                        })?
                        .into_iter(),
                    )
                } else {
                    df
                };

                out.push(df);

                if n_rows.is_some() && n_rows_read == n_rows.unwrap() {
                    if verbose {
                        eprintln!(
                            "reached n_rows = {} at file {} / {}",
                            n_rows.unwrap(),
                            1 + i,
                            self.paths.len()
                        )
                    }
                    break;
                }
            }

            accumulate_dataframes_vertical(out.into_iter())?
        } else {
            // Basic parallel read
            assert!(
                n_rows.is_none()
                    && !(
                        // We can do either but not both because we are doing them
                        // out-of-order for parallel.
                        predicate.is_some() && self.file_options.row_index.is_some()
                    )
            );
            if verbose {
                eprintln!("read files in parallel")
            }

            let dfs = POOL.install(|| {
                let step = std::cmp::min(POOL.current_num_threads(), 128);

                (0..self.paths.len())
                    .step_by(step)
                    .map(|start| {
                        (start..std::cmp::min(start.saturating_add(step), self.paths.len()))
                            .into_par_iter()
                            .map(|i| finish_read(i, options_base.clone(), predicate.clone()))
                            .collect::<PolarsResult<Vec<_>>>()
                    })
                    .collect::<PolarsResult<Vec<_>>>()
            })?;

            let mut df =
                accumulate_dataframes_vertical(dfs.into_iter().flat_map(|dfs| dfs.into_iter()))?;

            if let Some(row_index) = self.file_options.row_index.clone() {
                df.with_row_index_mut(row_index.name.as_ref(), Some(row_index.offset));
            }

            df
        };

        if self.file_options.rechunk {
            df.as_single_chunk_par();
        };

        Ok(df)
    }
}

impl Executor for CsvExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        let profile_name = if state.has_node_timer() {
            let mut ids = vec![self.paths[0].to_string_lossy().into()];
            if self.predicate.is_some() {
                ids.push("predicate".into())
            }
            let name = comma_delimited("csv".to_string(), &ids);
            Cow::Owned(name)
        } else {
            Cow::Borrowed("")
        };

        state.record(|| self.read(), profile_name)
    }
}
