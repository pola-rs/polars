use std::borrow::Cow;
use std::io::Cursor;
use std::sync::Arc;

use polars_core::config;
use polars_core::prelude::{AnyValue, Column};
use polars_core::scalar::Scalar;
use polars_core::schema::SchemaRef;
use polars_core::utils::accumulate_dataframes_vertical;
use polars_expr::prelude::phys_expr_to_io_expr;
use polars_io::SerReader;
use polars_io::avro::AvroReader;
use polars_io::predicates::SkipBatchPredicate;
use polars_plan::plans::FileInfo;
use polars_plan::prelude::FileScanOptions;
use polars_utils::pl_str::PlSmallStr;

use super::{DataFrame, DataType, IdxSize, PolarsResult, ScanSources, comma_delimited};
use crate::executors::ScanExec;
use crate::{Executor, ScanPredicate};

pub struct AvroExec {
    sources: ScanSources,
    file_info: FileInfo,
    file_options: Box<FileScanOptions>,
    predicate: Option<ScanPredicate>,
}

impl AvroExec {
    pub fn new(
        sources: ScanSources,
        file_info: FileInfo,
        file_options: Box<FileScanOptions>,
        predicate: Option<ScanPredicate>,
    ) -> Self {
        Self {
            sources,
            file_options,
            file_info,
            predicate,
        }
    }

    fn read_impl(&mut self) -> PolarsResult<DataFrame> {
        let schema = self
            .file_info
            .reader_schema
            .as_ref()
            .unwrap()
            .as_ref()
            .unwrap_right();

        let verbose = config::verbose();
        let force_async = config::force_async();
        let run_async = (self.sources.is_paths() && force_async) || self.sources.is_cloud_url();

        if self.sources.is_paths() && force_async && verbose {
            eprintln!("ASYNC READING FORCED");
        }

        // TODO just store projection here
        let with_columns = self.file_options.with_columns.clone();
        let mut n_rows = self.file_options.pre_slice.map(|(start, len)| {
            assert_eq!(start, 0);
            len
        });

        let mut dfs = Vec::new();
        let mut row_index = self.file_options.row_index.clone();
        for source in self.sources.iter() {
            if n_rows == Some(0) {
                break;
            }
            let memslice = source.to_memslice_async_assume_latest(run_async)?;
            let reader = AvroReader::new(Cursor::new(memslice));

            let mut df = reader
                .set_rechunk(self.file_options.rechunk)
                .with_predicate(
                    self.predicate
                        .as_ref()
                        .map(|p| phys_expr_to_io_expr(p.predicate.clone())),
                )
                .with_columns(with_columns.clone())
                .with_n_rows(n_rows)
                .with_row_index(row_index.as_mut())
                .finish()?;

            if let Some(ref mut n_rows) = n_rows {
                *n_rows -= df.height();
            }

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

            if !df.is_empty() {
                dfs.push(df);
            }
        }

        // Avoid panicking if there are no rows
        if dfs.is_empty() {
            let mut df = DataFrame::empty_with_schema(schema);
            if let Some(col) = &self.file_options.include_file_paths {
                unsafe {
                    df.with_column_unchecked(Column::new_empty(col.clone(), &DataType::String))
                };
            }
            if let Some(row_index) = &self.file_options.row_index {
                df.with_row_index_mut(row_index.name.clone(), Some(row_index.offset));
            }
            Ok(df)
        } else {
            accumulate_dataframes_vertical(dfs)
        }
    }
}

impl ScanExec for AvroExec {
    fn read(
        &mut self,
        with_columns: Option<Arc<[PlSmallStr]>>,
        slice: Option<(usize, usize)>,
        predicate: Option<ScanPredicate>,
        _skip_batch_predicate: Option<Arc<dyn SkipBatchPredicate>>,
        row_index: Option<polars_io::RowIndex>,
    ) -> polars_error::PolarsResult<polars_core::prelude::DataFrame> {
        self.file_options.with_columns = with_columns;
        self.file_options.pre_slice = slice.map(|(s, l)| (s as i64, l));
        self.predicate = predicate;
        self.file_options.row_index = row_index;

        if self.file_info.reader_schema.is_none() {
            self.schema()?;
        }
        self.read_impl()
    }

    fn schema(&mut self) -> PolarsResult<&SchemaRef> {
        let either = match &mut self.file_info.reader_schema {
            Some(schema) => schema.as_ref(),
            reader_schema => {
                let force_async = config::force_async();
                let run_async =
                    (self.sources.is_paths() && force_async) || self.sources.is_cloud_url();
                let memslice = self
                    .sources
                    .at(0)
                    .to_memslice_async_assume_latest(run_async)?;

                let mut reader = AvroReader::new(Cursor::new(memslice));
                let schema = arrow::Either::Right(Arc::new(reader.schema()?));
                reader_schema.insert(schema).as_ref()
            },
        };
        Ok(either.unwrap_right())
    }

    fn num_unfiltered_rows(&mut self) -> polars_error::PolarsResult<polars_utils::IdxSize> {
        let (lb, ub) = self.file_info.row_estimation;
        let rows = if lb.is_some_and(|lb| lb == ub) {
            ub
        } else {
            let force_async = config::force_async();
            let run_async = (self.sources.is_paths() && force_async) || self.sources.is_cloud_url();

            let mut num_rows = 0;
            for source in self.sources.iter() {
                let memslice = source.to_memslice_async_assume_latest(run_async)?;
                let reader = AvroReader::new(Cursor::new(memslice));
                num_rows += reader.unfiltered_count()?;
            }

            // cache for future calls
            self.file_info.row_estimation = (Some(num_rows), num_rows);
            num_rows
        };
        Ok(rows as IdxSize)
    }
}

impl Executor for AvroExec {
    fn execute(
        &mut self,
        state: &mut polars_expr::prelude::ExecutionState,
    ) -> polars_error::PolarsResult<polars_core::prelude::DataFrame> {
        let profile_name = if state.has_node_timer() {
            let ids = vec![self.sources.id()];
            let name = comma_delimited("avro".to_string(), &ids);
            Cow::Owned(name)
        } else {
            Cow::Borrowed("")
        };

        state.record(|| self.read_impl(), profile_name)
    }
}
