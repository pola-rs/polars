use std::borrow::Cow;

use hive::HivePartitions;
use polars_core::config;
use polars_core::frame::column::ScalarColumn;
use polars_core::utils::{
    accumulate_dataframes_vertical, accumulate_dataframes_vertical_unchecked,
};
use polars_io::predicates::BatchStats;

use super::Executor;
use crate::executors::ParquetExec;
use crate::prelude::*;

struct LazyTryCell<T, E, F> {
    f: F,
    evaluated: Option<T>,
    _pd: std::marker::PhantomData<E>,
}

impl<T, E, F> LazyTryCell<T, E, F> {
    fn new(f: F) -> Self {
        Self {
            f,
            evaluated: None,
            _pd: std::marker::PhantomData,
        }
    }
}

impl<T: Copy, E, F: FnMut() -> Result<T, E>> LazyTryCell<T, E, F> {
    fn get(&mut self) -> Result<T, E> {
        if let Some(evaluated) = self.evaluated.as_ref() {
            return Ok(*evaluated);
        }

        match (self.f)() {
            Ok(v) => {
                self.evaluated = Some(v);
                self.get()
            },
            Err(e) => Err(e),
        }
    }
}

pub trait ScanExec {
    fn read(&mut self) -> PolarsResult<DataFrame>;
    fn num_unfiltered_rows(&mut self) -> PolarsResult<IdxSize>;
    fn schema(&mut self) -> PolarsResult<Schema>;
}

pub struct HiveExec {
    sources: ScanSources,
    file_info: FileInfo,
    hive_parts: Option<Arc<Vec<HivePartitions>>>,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    file_options: FileScanOptions,
    scan_type: FileScan,
}

impl HiveExec {
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
        if let Some(with_columns) = &self.file_options.with_columns {
            file_with_columns = Some(
                with_columns
                    .iter()
                    .filter(|&c| !hive_column_set.contains(c))
                    .cloned()
                    .collect(),
            );
        }

        let allow_missing_columns = self.file_options.allow_missing_columns;
        self.file_options.allow_missing_columns = false;
        let mut row_index = self.file_options.row_index.take();
        let mut slice = self.file_options.slice.take();

        let current_schema = self.file_info.schema.clone();
        let output_schema = current_schema.clone();
        let mut missing_columns = Vec::new();

        assert!(slice.is_none_or(|s| s.0 >= 0), "NYI");

        #[cfg(feature = "parquet")]
        {
            let FileScan::Parquet {
                options,
                cloud_options,
                metadata: _,
            } = self.scan_type.clone()
            else {
                todo!()
            };

            let verbose = config::verbose();
            let mut dfs = Vec::with_capacity(self.sources.len());

            let mut const_columns = PlHashMap::new();

            for (i, source) in self.sources.iter().enumerate() {
                let hive_part = self.hive_parts.as_ref().and_then(|h| h.get(i));
                if slice.is_some_and(|s| s.1 == 0) {
                    break;
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

                let part_source = match source {
                    ScanSourceRef::Path(path) => ScanSources::Paths([path.to_path_buf()].into()),
                    ScanSourceRef::File(_) | ScanSourceRef::Buffer(_) => {
                        ScanSources::Buffers([source.to_memslice()?].into())
                    },
                };

                if verbose {
                    eprintln!(
                        "Multi-file / Hive read: currently reading '{}'",
                        source.to_include_path_name()
                    );
                }

                // @TODO: There are cases where we can ignore reading. E.g. no row index + empty with columns + no predicate
                let mut exec = ParquetExec::new(
                    part_source.clone(),
                    self.file_info.clone(),
                    None,
                    file_predicate.clone(),
                    options.clone(),
                    cloud_options.clone(),
                    self.file_options.clone(),
                    None,
                );

                let mut schema = exec.schema()?;
                let mut extra_columns = Vec::new();

                if let Some(file_with_columns) = &file_with_columns {
                    if allow_missing_columns {
                        schema = schema.try_project(
                            file_with_columns
                                .iter()
                                .filter(|c| schema.contains(c.as_str())),
                        )?;
                    } else {
                        schema = schema.try_project(file_with_columns.iter())?;
                    }
                }

                if allow_missing_columns {
                    missing_columns.clear();
                    extra_columns.clear();

                    current_schema.as_ref().field_compare(
                        &schema,
                        &mut missing_columns,
                        &mut extra_columns,
                    );

                    if !extra_columns.is_empty() {
                        // @TODO: Better error
                        polars_bail!(InvalidOperation: "More schema in file after first");
                    }
                }

                let mut file_options = self.file_options.clone();
                // @TODO: We should really supply a bitmask to the readers instead of supplying
                // the column names. It just leads to way less confusion about what the reader is
                // supposed to do.
                file_options.with_columns = if allow_missing_columns {
                    file_with_columns
                        .as_ref()
                        .map(|_| schema.iter_names().cloned().collect())
                } else {
                    file_with_columns.clone()
                };
                file_options.row_index = row_index.clone();

                let mut file_info = self.file_info.clone();
                // @TODO: Directly read parquet arrow schema
                file_info.reader_schema = Some(arrow::Either::Left(Arc::new(
                    schema.to_arrow(CompatLevel::newest()),
                )));
                file_info.schema = Arc::new(schema);

                let mut exec = ParquetExec::new(
                    part_source,
                    file_info,
                    None,
                    file_predicate.clone(),
                    options.clone(),
                    cloud_options.clone(),
                    file_options,
                    None,
                );
                let mut num_unfiltered_rows = LazyTryCell::new(|| exec.num_unfiltered_rows());

                let mut do_skip_file = false;
                if let Some(slice) = &slice {
                    let allow_slice_skip = slice.0 >= num_unfiltered_rows.get()? as i64;
                    if allow_slice_skip && verbose {
                        eprintln!(
                            "Slice allows skipping of '{}'",
                            source.to_include_path_name()
                        );
                    }
                    do_skip_file |= allow_slice_skip;
                }

                let stats_evaluator = file_predicate.as_ref().and_then(|p| p.as_stats_evaluator());
                let stats_evaluator = stats_evaluator.filter(|_| options.use_statistics);

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
                        row_index.offset += num_unfiltered_rows.get()?;
                    }
                    // Update the slice offset.
                    if let Some(slice) = slice.as_mut() {
                        slice.0 = slice.0.saturating_sub(num_unfiltered_rows.get()? as i64);
                    }

                    continue;
                }

                // Read the DataFrame and needed metadata.
                // @TODO: these should be merged into one call
                let num_unfiltered_rows = num_unfiltered_rows.get()?;
                let mut df = exec.read()?;

                // Update the row_index to the proper offset.
                if let Some(row_index) = row_index.as_mut() {
                    row_index.offset += num_unfiltered_rows;
                }
                // Update the slice.
                if let Some(slice) = slice.as_mut() {
                    slice.0 = slice.0.saturating_sub(num_unfiltered_rows as i64);
                    slice.1 = slice.1.saturating_sub(num_unfiltered_rows as usize);
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
                df = df.select(output_schema.iter_names().cloned())?;
                dfs.push(df);
            }

            let out = if cfg!(debug_assertions) {
                accumulate_dataframes_vertical(dfs)?
            } else {
                accumulate_dataframes_vertical_unchecked(dfs)
            };

            Ok(out)
        }

        #[cfg(not(feature = "parquet"))]
        {
            todo!()
        }
    }
}

impl Executor for HiveExec {
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
