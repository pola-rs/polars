use std::borrow::Cow;

use hive::HivePartitions;
use polars_core::config;
use polars_core::frame::column::ScalarColumn;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_io::predicates::BatchStats;
use polars_io::RowIndex;

use super::Executor;
#[cfg(feature = "csv")]
use crate::executors::CsvExec;
#[cfg(feature = "ipc")]
use crate::executors::IpcExec;
#[cfg(feature = "json")]
use crate::executors::JsonExec;
#[cfg(feature = "parquet")]
use crate::executors::ParquetExec;
use crate::prelude::*;

pub struct PhysicalExprWithConstCols {
    constants: Vec<(PlSmallStr, Scalar)>,
    child: Arc<dyn PhysicalExpr>,
}

impl PhysicalExpr for PhysicalExprWithConstCols {
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let mut df = df.clone();
        for (name, scalar) in &self.constants {
            df.with_column(Column::new_scalar(
                name.clone(),
                scalar.clone(),
                df.height(),
            ))?;
        }

        self.child.evaluate(&df, state)
    }

    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut df = df.clone();
        for (name, scalar) in &self.constants {
            df.with_column(Column::new_scalar(
                name.clone(),
                scalar.clone(),
                df.height(),
            ))?;
        }

        self.child.evaluate_on_groups(&df, groups, state)
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.child.to_field(input_schema)
    }

    fn collect_live_columns(&self, lv: &mut PlIndexSet<PlSmallStr>) {
        self.child.collect_live_columns(lv)
    }
    fn is_scalar(&self) -> bool {
        self.child.is_scalar()
    }
}

/// An [`Executor`] that scans over some IO.
pub trait ScanExec {
    /// Read the source.
    fn read(
        &mut self,
        with_columns: Option<Arc<[PlSmallStr]>>,
        slice: Option<(usize, usize)>,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        row_index: Option<RowIndex>,
    ) -> PolarsResult<DataFrame>;

    /// Get the full schema for the source behind this [`Executor`].
    ///
    /// Note that this might be called several times so attempts should be made to cache the result.
    fn schema(&mut self) -> PolarsResult<&SchemaRef>;
    /// Get the number of rows for the source behind this [`Executor`].
    ///
    /// Note that this might be called several times so attempts should be made to cache the result.
    fn num_unfiltered_rows(&mut self) -> PolarsResult<IdxSize>;
}

fn source_to_exec(
    source: ScanSourceRef,
    scan_type: &FileScan,
    file_info: &FileInfo,
    file_options: &FileScanOptions,
    allow_missing_columns: bool,
    file_index: usize,
) -> PolarsResult<Box<dyn ScanExec>> {
    let source = match source {
        ScanSourceRef::Path(path) => ScanSources::Paths([path.to_path_buf()].into()),
        ScanSourceRef::File(_) | ScanSourceRef::Buffer(_) => {
            ScanSources::Buffers([source.to_memslice()?].into())
        },
    };

    let is_first_file = file_index == 0;

    let mut file_info = file_info.clone();

    if allow_missing_columns && !is_first_file {
        file_info.reader_schema.take();
    }

    Ok(match scan_type {
        #[cfg(feature = "parquet")]
        FileScan::Parquet {
            options,
            cloud_options,
            metadata,
        } => {
            let metadata = metadata.as_ref().take_if(|_| is_first_file);

            let mut options = options.clone();

            if allow_missing_columns && !is_first_file {
                options.schema.take();
            }

            Box::new(ParquetExec::new(
                source,
                file_info,
                None,
                None,
                options,
                cloud_options.clone(),
                file_options.clone(),
                metadata.cloned(),
            ))
        },
        #[cfg(feature = "csv")]
        FileScan::Csv { options, .. } => {
            let mut options = options.clone();
            let file_options = file_options.clone();

            if allow_missing_columns && !is_first_file {
                options.schema.take();
            }

            Box::new(CsvExec {
                sources: source,
                file_info,
                options,
                file_options,
                predicate: None,
            })
        },
        #[cfg(feature = "ipc")]
        FileScan::Ipc {
            options,
            cloud_options,
            metadata,
        } => {
            let metadata = metadata.as_ref().take_if(|_| is_first_file);

            let options = options.clone();
            let file_options = file_options.clone();
            let cloud_options = cloud_options.clone();

            Box::new(IpcExec {
                sources: source,
                file_info,
                options,
                file_options,
                predicate: None,
                hive_parts: None,
                cloud_options,
                metadata: metadata.cloned(),
            })
        },
        #[cfg(feature = "json")]
        FileScan::NDJson {
            options,
            cloud_options,
            ..
        } => {
            let options = options.clone();
            let file_options = file_options.clone();
            _ = cloud_options; // @TODO: Use these?

            Box::new(JsonExec::new(
                source,
                options,
                file_options,
                file_info,
                None,
            ))
        },
        FileScan::Anonymous { .. } => unreachable!(),
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
}

impl MultiScanExec {
    pub fn new(
        sources: ScanSources,
        file_info: FileInfo,
        hive_parts: Option<Arc<Vec<HivePartitions>>>,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        file_options: FileScanOptions,
        scan_type: FileScan,
    ) -> Self {
        Self {
            sources,
            file_info,
            hive_parts,
            predicate,
            file_options,
            scan_type,
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
                i,
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

        let final_per_source_schema = &self.file_info.schema;
        let file_output_schema = if let Some(file_with_columns) = file_with_columns.as_ref() {
            let mut schema = final_per_source_schema.try_project(file_with_columns.as_ref())?;

            if let Some(v) = include_file_paths.clone() {
                schema.extend([(v, DataType::String)]);
            }

            Arc::new(schema)
        } else {
            final_per_source_schema.clone()
        };

        if slice.is_some_and(|x| x.1 == 0) {
            return Ok(DataFrame::empty_with_schema(final_per_source_schema));
        }

        let mut missing_columns = Vec::new();

        let verbose = config::verbose();
        let mut dfs = Vec::with_capacity(self.sources.len());

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
                i,
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

            let mut file_predicate = predicate.clone();

            // Insert the hive partition values into the predicate. This allows the predicate
            // to function even when there is a combination of hive and non-hive columns being
            // used.
            if has_live_hive_columns {
                let hive_part = hive_part.unwrap();
                let child = file_predicate.unwrap();

                file_predicate = Some(Arc::new(PhysicalExprWithConstCols {
                    constants: hive_column_set
                        .iter()
                        .enumerate()
                        .map(|(idx, column)| {
                            let series = hive_part.get_statistics().column_stats()[idx]
                                .to_min()
                                .unwrap();
                            (
                                column.clone(),
                                Scalar::new(
                                    series.dtype().clone(),
                                    series.get(0).unwrap().into_static(),
                                ),
                            )
                        })
                        .collect(),
                    child,
                }));
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
            let mut current_source_with_columns = Cow::Borrowed(&file_with_columns);

            // If we allow missing columns, we need to determine the set of missing columns and
            // possibly update the with_columns to reflect that.
            if allow_missing_columns {
                let current_source_schema = exec_source.schema()?;

                missing_columns.clear();

                let mut extra_columns = Vec::new();
                final_per_source_schema.as_ref().field_compare(
                    current_source_schema.as_ref(),
                    &mut missing_columns,
                    &mut extra_columns,
                );

                if !extra_columns.is_empty() {
                    let source_name = match source {
                        ScanSourceRef::Path(path) => path.to_string_lossy().into_owned(),
                        ScanSourceRef::File(_) => format!("file descriptor #{}", i + 1),
                        ScanSourceRef::Buffer(_) => format!("in-memory buffer #{}", i + 1),
                    };
                    let columns = extra_columns
                        .iter()
                        .map(|(_, (name, _))| format!("'{}'", name))
                        .collect::<Vec<_>>()
                        .join(", ");
                    polars_bail!(
                        SchemaMismatch:
                        "'{source_name}' contains column(s) {columns}, which are not present in the first scanned file"
                    );
                }

                // Update `with_columns` to not include any columns not present in the file.
                current_source_with_columns =
                    Cow::Owned(current_source_with_columns.as_deref().map(|with_columns| {
                        with_columns
                            .iter()
                            .filter(|c| current_source_schema.contains(c))
                            .cloned()
                            .collect()
                    }));
            }

            // Read the DataFrame.
            let mut df = exec_source.read(
                current_source_with_columns.into_owned(),
                slice,
                file_predicate,
                row_index.clone(),
            )?;

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

        if dfs.is_empty() {
            Ok(DataFrame::empty_with_schema(final_per_source_schema))
        } else {
            Ok(accumulate_dataframes_vertical_unchecked(dfs))
        }
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
