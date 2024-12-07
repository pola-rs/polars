use std::borrow::Cow;
use std::cell::LazyCell;

use hive::HivePartitions;
use polars_core::frame::column::ScalarColumn;
use polars_core::utils::{
    accumulate_dataframes_vertical, accumulate_dataframes_vertical_unchecked,
};

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
                self.evaluated.insert(v);
                self.get()
            },
            Err(e) => Err(e),
        }
    }
}

pub trait ScanExec {
    fn read(&mut self) -> PolarsResult<DataFrame> {
        self.read_with_num_unfiltered_rows().map(|(_, df)| df)
    }
    fn num_unfiltered_rows(&mut self) -> PolarsResult<IdxSize>;
    fn read_with_num_unfiltered_rows(&mut self) -> PolarsResult<(IdxSize, DataFrame)>;
}

pub struct HiveExec {
    sources: ScanSources,
    file_info: FileInfo,
    hive_parts: Arc<Vec<HivePartitions>>,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    file_options: FileScanOptions,
    scan_type: FileScan,
}

impl HiveExec {
    pub fn new(
        sources: ScanSources,
        file_info: FileInfo,
        hive_parts: Arc<Vec<HivePartitions>>,
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

        state.record(
            || {
                let include_file_paths = self.file_options.include_file_paths.take();
                let mut row_index = self.file_options.row_index.take();
                let mut slice = self.file_options.slice.take();

                assert_eq!(self.sources.len(), self.hive_parts.len());

                assert!(!self.file_options.allow_missing_columns, "NYI");
                assert!(slice.is_none_or(|s| s.0 >= 0), "NYI");

                #[cfg(feature = "parquet")]
                {
                    let FileScan::Parquet {
                        options,
                        cloud_options,
                        metadata,
                    } = self.scan_type.clone()
                    else {
                        todo!()
                    };

                    let mut dfs = Vec::with_capacity(self.sources.len());

                    for (source, hive_part) in self.sources.iter().zip(self.hive_parts.iter()) {
                        if slice.is_some_and(|s| s.1 == 0) {
                            break;
                        }

                        let part_source = match source {
                            ScanSourceRef::Path(path) => {
                                ScanSources::Paths([path.to_path_buf()].into())
                            },
                            ScanSourceRef::File(_) | ScanSourceRef::Buffer(_) => {
                                ScanSources::Buffers([source.to_memslice()?].into())
                            },
                        };

                        let mut file_options = self.file_options.clone();
                        file_options.row_index = row_index.clone();

                        // @TODO: There are cases where we can ignore reading. E.g. no row index + empty with columns + no predicate
                        let mut exec = ParquetExec::new(
                            part_source,
                            self.file_info.clone(),
                            None,
                            self.predicate.clone(),
                            options.clone(),
                            cloud_options.clone(),
                            file_options,
                            metadata.clone(),
                        );

                        let mut num_unfiltered_rows =
                            LazyTryCell::new(|| exec.num_unfiltered_rows());

                        let mut do_skip_file = false;
                        if let Some(slice) = &slice {
                            do_skip_file |= slice.0 >= num_unfiltered_rows.get()? as i64;
                        }
                        // @TODO: Skipping based on the predicate

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

                        let (num_unfiltered_rows, mut df) = exec.read_with_num_unfiltered_rows()?;

                        // Update the row_index to the proper offset.
                        if let Some(row_index) = row_index.as_mut() {
                            row_index.offset += num_unfiltered_rows;
                        }

                        // Update the slice.
                        if let Some(slice) = slice.as_mut() {
                            slice.0 = slice.0.saturating_sub(num_unfiltered_rows as i64);
                            slice.1 = slice.1.saturating_sub(num_unfiltered_rows as usize);
                        }

                        if let Some(with_columns) = &self.file_options.with_columns {
                            df = match &row_index {
                                None => df.select(with_columns.iter().cloned())?,
                                Some(ri) => df.select(
                                    std::iter::once(ri.name.clone())
                                        .chain(with_columns.iter().cloned()),
                                )?,
                            }
                        }

                        // Materialize the hive columns and add them basic in.
                        let hive_df: DataFrame = hive_part
                            .get_statistics()
                            .column_stats()
                            .iter()
                            .map(|hive_col| {
                                ScalarColumn::from_single_value_series(
                                    hive_col
                                        .to_min()
                                        .unwrap()
                                        .clone()
                                        .with_name(hive_col.field_name().clone()),
                                    df.height(),
                                )
                                .into_column()
                            })
                            .collect();
                        let mut df = hive_df.hstack(df.get_columns())?;

                        if let Some(include_file_paths) = &include_file_paths {
                            df.with_column(ScalarColumn::new(
                                include_file_paths.clone(),
                                PlSmallStr::from_str(source.to_include_path_name()).into(),
                                df.height(),
                            ))?;
                        }

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
            },
            profile_name,
        )
    }
}
