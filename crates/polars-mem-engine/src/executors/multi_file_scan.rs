use std::borrow::Cow;

use hive::HivePartitions;
use polars_core::config;
use polars_core::frame::column::ScalarColumn;
use polars_core::utils::{
    accumulate_dataframes_vertical, accumulate_dataframes_vertical_unchecked,
};
use polars_io::predicates::BatchStats;
use polars_io::RowIndex;

use super::Executor;
#[cfg(feature = "csv")]
use crate::executors::CsvExec;
#[cfg(feature = "parquet")]
use crate::executors::ParquetExec;
use crate::prelude::*;

pub trait IOFileMetadata: Send + Sync {
    fn as_any(&self) -> &dyn std::any::Any;
}

pub trait ScanExec {
    fn read(
        &mut self,
        with_columns: Option<Arc<[PlSmallStr]>>,
        slice: Option<(usize, usize)>,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        row_index: Option<RowIndex>,
    ) -> PolarsResult<DataFrame>;

    fn schema(&mut self) -> PolarsResult<&SchemaRef>;
    fn num_unfiltered_rows(&mut self) -> PolarsResult<IdxSize>;
}

fn source_to_exec(
    source: ScanSourceRef,
    scan_type: &FileScan,
    file_info: &FileInfo,
    file_options: &FileScanOptions,
    allow_missing_columns: bool,
    metadata: Option<&dyn IOFileMetadata>,
) -> PolarsResult<Box<dyn ScanExec>> {
    let source = match source {
        ScanSourceRef::Path(path) => ScanSources::Paths([path.to_path_buf()].into()),
        ScanSourceRef::File(_) | ScanSourceRef::Buffer(_) => {
            ScanSources::Buffers([source.to_memslice()?].into())
        },
    };

    Ok(match scan_type {
        #[cfg(feature = "parquet")]
        FileScan::Parquet {
            options,
            cloud_options,
            ..
        } => {
            let mut options = options.clone();
            let mut file_info = file_info.clone();
            if allow_missing_columns {
                options.schema.take();
                file_info.reader_schema.take();
            }

            let mut exec = ParquetExec::new(
                source,
                file_info,
                None,
                None,
                options,
                cloud_options.clone(),
                file_options.clone(),
                metadata.map(|md| {
                    md.as_any()
                        .downcast_ref::<Arc<polars_io::parquet::read::FileMetadata>>()
                        .unwrap()
                        .clone()
                }),
            );

            if allow_missing_columns {
                // Fixes the file_info.schema
                exec.schema()?;
            }

            Box::new(exec)
        },
        #[cfg(feature = "csv")]
        FileScan::Csv { options, .. } => {
            let mut file_info = file_info.clone();
            let mut options = options.clone();
            let file_options = file_options.clone();

            if allow_missing_columns {
                options.schema.take();
                file_info.reader_schema.take();
            }

            let mut exec = CsvExec {
                sources: source,
                file_info,
                options,
                file_options,
                predicate: None,
            };

            if allow_missing_columns {
                // Fixes the file_info.schema
                exec.schema()?;
            }

            Box::new(exec)
        },
        _ => todo!(),
    })
}

/// Scan over multiple sources and combine their results.
pub struct MultiScanExec {
    sources: ScanSources,
    file_info: FileInfo,
    hive_parts: Option<Arc<Vec<HivePartitions>>>,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    file_options: FileScanOptions,
    scan_type: FileScan,

    first_file_metadata: Option<Box<dyn IOFileMetadata>>,
}

impl MultiScanExec {
    pub fn new(
        sources: ScanSources,
        file_info: FileInfo,
        hive_parts: Option<Arc<Vec<HivePartitions>>>,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        file_options: FileScanOptions,
        mut scan_type: FileScan,
    ) -> Self {
        let first_file_metadata = match &mut scan_type {
            #[cfg(feature = "parquet")]
            FileScan::Parquet { metadata, .. } => metadata.take().map(|md| Box::new(md) as _),
            _ => None,
        };

        Self {
            sources,
            file_info,
            hive_parts,
            predicate,
            file_options,
            scan_type,
            first_file_metadata,
        }
    }

    pub fn resolve_negative_slice(
        &mut self,
        offset: i64,
        length: usize,
    ) -> PolarsResult<(usize, usize)> {
        // Walk the files in reverse until we find the first file, and then translate the
        // slice into a positive-offset equivalent.
        let mut offset_remaining = -offset as usize;

        for i in (0..self.sources.len()).rev() {
            let source = self.sources.get(i).unwrap();
            let mut exec_source = source_to_exec(
                source,
                &self.scan_type,
                &self.file_info,
                &self.file_options,
                self.file_options.allow_missing_columns,
                self.first_file_metadata.as_deref().filter(|_| i == 0),
            )?;

            let num_rows = exec_source.num_unfiltered_rows()? as usize;

            if num_rows >= offset_remaining {
                return Ok((i, num_rows - offset_remaining));
            }
            offset_remaining -= num_rows;
        }

        Ok((0, length - offset_remaining))
    }

    pub fn read(&mut self) -> PolarsResult<DataFrame> {
        let include_file_paths = self.file_options.include_file_paths.take();
        let predicate = self.predicate.take();

        // Create a index set of the hive columns.
        let mut hive_column_set = PlIndexSet::default();
        if let Some(hive_parts) = &self.hive_parts {
            assert_eq!(self.sources.len(), hive_parts.len());

            if let Some(fst_hive_part) = hive_parts.first() {
                hive_column_set.extend(
                    fst_hive_part
                        .get_statistics()
                        .column_stats()
                        .iter()
                        .map(|c| c.field_name().clone()),
                );
            }
        }

        // Look through the predicate and assess whether hive columns are being used in it.
        let mut has_live_hive_columns = false;
        if let Some(predicate) = &predicate {
            let mut live_columns = PlIndexSet::new();
            predicate.collect_live_columns(&mut live_columns);

            for hive_column in &hive_column_set {
                has_live_hive_columns |= live_columns.contains(hive_column);
            }
        }

        // Remove the hive columns for each file load.
        let mut file_with_columns = self.file_options.with_columns.take();
        if self.hive_parts.is_some() {
            if let Some(with_columns) = &self.file_options.with_columns {
                file_with_columns = Some(
                    with_columns
                        .iter()
                        .filter(|&c| !hive_column_set.contains(c))
                        .cloned()
                        .collect(),
                );
            }
        }

        let allow_missing_columns = self.file_options.allow_missing_columns;
        self.file_options.allow_missing_columns = false;
        let mut row_index = self.file_options.row_index.take();
        let slice = self.file_options.slice.take();

        let file_output_schema = &self.file_info.schema;
        let file_output_schema = if let Some(file_with_columns) = file_with_columns.as_ref() {
            Arc::new(file_output_schema.try_project(file_with_columns.as_ref())?)
        } else {
            file_output_schema.clone()
        };
        let mut missing_columns = Vec::new();

        let mut first_slice_file = None;
        let mut slice = match slice {
            None => None,
            Some((offset, length)) => Some({
                if offset >= 0 {
                    (offset as usize, length)
                } else {
                    let (first_file, offset) = self.resolve_negative_slice(offset, length)?;
                    first_slice_file = Some(first_file);
                    (offset, length)
                }
            }),
        };

        let verbose = config::verbose();
        let mut dfs = Vec::with_capacity(self.sources.len());

        let mut const_columns = PlHashMap::new();

        // @TODO: This should be moved outside of the FileScan::Parquet
        let use_statistics = match &self.scan_type {
            #[cfg(feature = "parquet")]
            FileScan::Parquet { options, .. } => options.use_statistics,
            _ => true,
        };

        for (i, source) in self.sources.iter().enumerate() {
            let hive_part = self.hive_parts.as_ref().and_then(|h| h.get(i));
            if slice.is_some_and(|s| s.1 == 0) {
                break;
            }

            let mut exec_source = source_to_exec(
                source,
                &self.scan_type,
                &self.file_info,
                &self.file_options,
                allow_missing_columns,
                self.first_file_metadata.as_deref().filter(|_| i == 0),
            )?;

            if verbose {
                eprintln!(
                    "Multi-file / Hive read: currently reading '{}'",
                    source.to_include_path_name()
                );
            }

            let mut do_skip_file = false;
            if let Some(slice) = &slice {
                let allow_slice_skip = match first_slice_file {
                    None => slice.0 as IdxSize >= exec_source.num_unfiltered_rows()?,
                    Some(f) => i < f,
                };

                if allow_slice_skip && verbose {
                    eprintln!(
                        "Slice allows skipping of '{}'",
                        source.to_include_path_name()
                    );
                }
                do_skip_file |= allow_slice_skip;
            }

            // Insert the hive partition values into the predicate. This allows the predicate
            // to function even when there is a combination of hive and non-hive columns being
            // used.
            let mut file_predicate = predicate.clone();
            if has_live_hive_columns {
                let hive_part = hive_part.unwrap();
                let predicate = predicate.as_ref().unwrap();
                const_columns.clear();
                for (idx, column) in hive_column_set.iter().enumerate() {
                    let value = hive_part.get_statistics().column_stats()[idx]
                        .to_min()
                        .unwrap()
                        .get(0)
                        .unwrap()
                        .into_static();
                    const_columns.insert(column.clone(), value);
                }
                // @TODO: It would be nice to get this somehow.
                // for (_, (missing_column, _)) in &missing_columns {
                //     const_columns.insert((*missing_column).clone(), AnyValue::Null);
                // }

                file_predicate = predicate.replace_elementwise_const_columns(&const_columns);

                // @TODO: Set predicate to `None` if it's constant evaluated to true.

                // At this point the file_predicate should not contain any references to the
                // hive columns anymore.
                //
                // Note that, replace_elementwise_const_columns does not actually guarantee the
                // replacement of all reference to the const columns. But any expression which
                // does not guarantee this should not be pushed down as an IO predicate.
                if cfg!(debug_assertions) {
                    let mut live_columns = PlIndexSet::new();
                    file_predicate
                        .as_ref()
                        .unwrap()
                        .collect_live_columns(&mut live_columns);
                    for hive_column in hive_part.get_statistics().column_stats() {
                        assert!(
                            !live_columns.contains(hive_column.field_name()),
                            "Predicate still contains hive column"
                        );
                    }
                }
            }


            let stats_evaluator = file_predicate.as_ref().and_then(|p| p.as_stats_evaluator());
            let stats_evaluator = stats_evaluator.filter(|_| use_statistics);

            if let Some(stats_evaluator) = stats_evaluator {
                let allow_predicate_skip = !stats_evaluator
                    .should_read(&BatchStats::default())
                    .unwrap_or(true);
                if allow_predicate_skip && verbose {
                    eprintln!(
                        "File statistics allows skipping of '{}'",
                        source.to_include_path_name()
                    );
                }
                do_skip_file |= allow_predicate_skip;
            }

            if do_skip_file {
                // Update the row_index to the proper offset.
                if let Some(row_index) = row_index.as_mut() {
                    row_index.offset += exec_source.num_unfiltered_rows()?;
                }
                // Update the slice offset.
                if let Some(slice) = slice.as_mut() {
                    if first_slice_file.is_none_or(|f| i >= f) {
                        slice.0 = slice
                            .0
                            .saturating_sub(exec_source.num_unfiltered_rows()? as usize);
                    }
                }

                continue;
            }

            // @TODO: There are cases where we can ignore reading. E.g. no row index + empty with columns + no predicate
            let mut source_schema = exec_source.schema()?.clone();
            let mut extra_columns = Vec::new();

            if let Some(file_with_columns) = &file_with_columns {
                if allow_missing_columns {
                    source_schema = Arc::new(
                        source_schema.as_ref().try_project(
                            file_with_columns
                                .iter()
                                .filter(|c| source_schema.contains(c.as_str())),
                        )?,
                    );
                } else {
                    source_schema = Arc::new(source_schema.try_project(file_with_columns.iter())?);
                }
            }

            if allow_missing_columns {
                missing_columns.clear();
                extra_columns.clear();

                file_output_schema.as_ref().field_compare(
                    &source_schema,
                    &mut missing_columns,
                    &mut extra_columns,
                );

                if !extra_columns.is_empty() {
                    // @TODO: Better error
                    polars_bail!(InvalidOperation: "More schema in file after first");
                }
            }

            let with_columns = if allow_missing_columns {
                file_with_columns
                    .as_ref()
                    .map(|_| source_schema.iter_names().cloned().collect())
            } else {
                file_with_columns.clone()
            };

            // Read the DataFrame.
            let mut df =
                exec_source.read(with_columns, slice, file_predicate, row_index.clone())?;

            // Update the row_index to the proper offset.
            if let Some(row_index) = row_index.as_mut() {
                row_index.offset += exec_source.num_unfiltered_rows()?;
            }
            // Update the slice.
            if let Some(slice) = slice.as_mut() {
                if first_slice_file.is_none_or(|f| i >= f) {
                    let num_unfiltered_rows = exec_source.num_unfiltered_rows()?;
                    slice.1 = slice
                        .1
                        .saturating_sub(num_unfiltered_rows as usize - slice.0);
                    slice.0 = slice.0.saturating_sub(num_unfiltered_rows as usize);
                }
            }

            // Add all the missing columns.
            if allow_missing_columns && !missing_columns.is_empty() {
                for (_, (name, field)) in &missing_columns {
                    df.with_column(Column::full_null((*name).clone(), df.height(), field))?;
                }
            }
            // Materialize the hive columns and add them back in.
            if let Some(hive_part) = hive_part {
                for hive_col in hive_part.get_statistics().column_stats() {
                    df.with_column(
                        ScalarColumn::from_single_value_series(
                            hive_col
                                .to_min()
                                .unwrap()
                                .clone()
                                .with_name(hive_col.field_name().clone()),
                            df.height(),
                        )
                        .into_column(),
                    )?;
                }
            }
            // Add the `include_file_paths` column
            if let Some(include_file_paths) = &include_file_paths {
                df.with_column(ScalarColumn::new(
                    include_file_paths.clone(),
                    PlSmallStr::from_str(source.to_include_path_name()).into(),
                    df.height(),
                ))?;
            }

            // Project to ensure that all DataFrames have the proper order.
            df = df.select(file_output_schema.iter_names().cloned())?;
            dfs.push(df);
        }

        let out = if cfg!(debug_assertions) {
            accumulate_dataframes_vertical(dfs)?
        } else {
            accumulate_dataframes_vertical_unchecked(dfs)
        };

        Ok(out)
    }
}

impl Executor for MultiScanExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        let profile_name = if state.has_node_timer() {
            let mut ids = vec![self.sources.id()];
            if self.predicate.is_some() {
                ids.push("predicate".into())
            }
            let name = comma_delimited("hive".to_string(), &ids);
            Cow::Owned(name)
        } else {
            Cow::Borrowed("")
        };

        state.record(|| self.read(), profile_name)
    }
}
