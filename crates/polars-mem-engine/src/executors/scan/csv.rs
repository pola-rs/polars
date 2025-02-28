use std::sync::Arc;

use polars_core::config;
use polars_core::utils::{
    accumulate_dataframes_vertical, accumulate_dataframes_vertical_unchecked,
};
use polars_io::predicates::SkipBatchPredicate;
use polars_io::utils::compression::maybe_decompress_bytes;

use super::*;
use crate::ScanPredicate;

pub struct CsvExec {
    pub sources: ScanSources,
    pub file_info: FileInfo,
    pub options: CsvReadOptions,
    pub file_options: FileScanOptions,
    pub predicate: Option<ScanPredicate>,
}

impl CsvExec {
    fn read_impl(&self) -> PolarsResult<DataFrame> {
        let with_columns = self
            .file_options
            .with_columns
            .clone()
            // Interpret selecting no columns as selecting all columns.
            .filter(|columns| !columns.is_empty());

        let n_rows = _set_n_rows_for_scan(self.file_options.slice.map(|x| {
            assert_eq!(x.0, 0);
            x.1
        }));
        let predicate = self
            .predicate
            .as_ref()
            .map(|p| phys_expr_to_io_expr(p.predicate.clone()));
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

        if self.sources.is_empty() {
            let out = if let Some(schema) = options_base.schema {
                DataFrame::from_rows_and_schema(&[], schema.as_ref())?
            } else {
                Default::default()
            };
            return Ok(out);
        }

        let verbose = config::verbose();
        let force_async = config::force_async();
        let run_async = (self.sources.is_paths() && force_async) || self.sources.is_cloud_url();

        if self.sources.is_paths() && force_async && verbose {
            eprintln!("ASYNC READING FORCED");
        }

        let finish_read =
            |i: usize, options: CsvReadOptions, predicate: Option<Arc<dyn PhysicalIoExpr>>| {
                let source = self.sources.at(i);
                let owned = &mut vec![];

                let memslice = source.to_memslice_async_assume_latest(run_async)?;

                let reader = std::io::Cursor::new(maybe_decompress_bytes(&memslice, owned)?);
                let mut df = options
                    .into_reader_with_file_handle(reader)
                    ._with_predicate(predicate.clone())
                    .finish()?;

                if let Some(col) = &self.file_options.include_file_paths {
                    let name = source.to_include_path_name();

                    unsafe {
                        df.with_column_unchecked(Column::new_scalar(
                            col.clone(),
                            Scalar::new(DataType::String, AnyValue::StringOwned(name.into())),
                            df.height(),
                        ))
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
            let mut out = Vec::with_capacity(self.sources.len());
            // If we have n_rows or row_index then we need to count how many rows we read, so we need
            // to delay applying the predicate.
            let predicate_during_read = predicate
                .clone()
                .filter(|_| n_rows.is_none() && self.file_options.row_index.is_none());

            for i in 0..self.sources.len() {
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
                            "reached n_rows = {} at source {} / {}",
                            n_rows.unwrap(),
                            1 + i,
                            self.sources.len()
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

                (0..self.sources.len())
                    .step_by(step)
                    .map(|start| {
                        (start..std::cmp::min(start.saturating_add(step), self.sources.len()))
                            .into_par_iter()
                            .map(|i| finish_read(i, options_base.clone(), predicate.clone()))
                            .collect::<PolarsResult<Vec<_>>>()
                    })
                    .collect::<PolarsResult<Vec<_>>>()
            })?;

            let mut df =
                accumulate_dataframes_vertical(dfs.into_iter().flat_map(|dfs| dfs.into_iter()))?;

            if let Some(row_index) = self.file_options.row_index.clone() {
                df.with_row_index_mut(row_index.name.clone(), Some(row_index.offset));
            }

            df
        };

        if self.file_options.rechunk {
            df.as_single_chunk_par();
        };

        Ok(df)
    }
}

impl ScanExec for CsvExec {
    fn read(
        &mut self,
        with_columns: Option<Arc<[PlSmallStr]>>,
        slice: Option<(usize, usize)>,
        predicate: Option<ScanPredicate>,
        _skip_batch_predicate: Option<Arc<dyn SkipBatchPredicate>>,
        row_index: Option<polars_io::RowIndex>,
    ) -> PolarsResult<DataFrame> {
        self.file_options.with_columns = with_columns;
        self.file_options.slice = slice.map(|(o, l)| (o as i64, l));
        self.predicate = predicate;
        self.file_options.row_index = row_index;

        if self.file_info.reader_schema.is_none() {
            self.schema()?;
        }
        self.read_impl()
    }

    fn schema(&mut self) -> PolarsResult<&SchemaRef> {
        let mut schema = self.file_info.reader_schema.take();
        if schema.is_none() {
            let force_async = config::force_async();
            let run_async = (self.sources.is_paths() && force_async) || self.sources.is_cloud_url();

            let source = self.sources.at(0);
            let owned = &mut vec![];

            let memslice = source.to_memslice_async_assume_latest(run_async)?;

            // @TODO!: Cache the decompression
            let bytes = maybe_decompress_bytes(&memslice, owned)?;

            schema = Some(arrow::Either::Right(Arc::new(
                infer_file_schema(
                    &get_reader_bytes(&mut std::io::Cursor::new(bytes))?,
                    self.options.parse_options.as_ref(),
                    self.options.infer_schema_length,
                    self.options.has_header,
                    self.options.schema_overwrite.as_deref(),
                    self.options.skip_rows,
                    self.options.skip_lines,
                    self.options.skip_rows_after_header,
                    self.options.raise_if_empty,
                    &mut self.options.n_threads,
                )?
                .0,
            )));
        }

        Ok(self
            .file_info
            .reader_schema
            .insert(schema.unwrap())
            .as_ref()
            .unwrap_right())
    }

    fn num_unfiltered_rows(&mut self) -> PolarsResult<IdxSize> {
        let (lb, ub) = self.file_info.row_estimation;
        if lb.is_some_and(|lb| lb == ub) {
            return Ok(ub as IdxSize);
        }

        let force_async = config::force_async();
        let run_async = (self.sources.is_paths() && force_async) || self.sources.is_cloud_url();

        let source = self.sources.at(0);
        let owned = &mut vec![];

        // @TODO!: Cache the decompression
        let memslice = source.to_memslice_async_assume_latest(run_async)?;

        let popt = self.options.parse_options.as_ref();

        let bytes = maybe_decompress_bytes(&memslice, owned)?;

        let num_rows = count_rows_from_slice(
            bytes,
            popt.separator,
            popt.quote_char,
            popt.comment_prefix.as_ref(),
            popt.eol_char,
            self.options.has_header,
        )?;

        self.file_info.row_estimation = (Some(num_rows), num_rows);
        Ok(num_rows as IdxSize)
    }
}

impl Executor for CsvExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        let profile_name = if state.has_node_timer() {
            let mut ids = vec![self.sources.id()];
            if self.predicate.is_some() {
                ids.push("predicate".into())
            }
            let name = comma_delimited("csv".to_string(), &ids);
            Cow::Owned(name)
        } else {
            Cow::Borrowed("")
        };

        state.record(|| self.read_impl(), profile_name)
    }
}
