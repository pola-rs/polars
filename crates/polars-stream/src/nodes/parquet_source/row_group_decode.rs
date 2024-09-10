use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::prelude::{ChunkFull, IdxCa, StringChunked};
use polars_core::series::{IntoSeries, IsSorted, Series};
use polars_error::{polars_bail, PolarsResult};
use polars_io::predicates::PhysicalIoExpr;
use polars_io::RowIndex;
use polars_plan::plans::hive::HivePartitions;
use polars_plan::plans::ScanSources;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::IdxSize;

use super::row_group_data_fetch::RowGroupData;
use crate::async_executor;
use crate::nodes::TaskPriority;

/// Turns row group data into DataFrames.
pub(super) struct RowGroupDecoder {
    pub(super) scan_sources: ScanSources,
    pub(super) hive_partitions: Option<Arc<Vec<HivePartitions>>>,
    pub(super) hive_partitions_width: usize,
    pub(super) include_file_paths: Option<PlSmallStr>,
    pub(super) projected_arrow_fields: Arc<[polars_core::prelude::ArrowField]>,
    pub(super) row_index: Option<RowIndex>,
    pub(super) physical_predicate: Option<Arc<dyn PhysicalIoExpr>>,
    pub(super) ideal_morsel_size: usize,
}

impl RowGroupDecoder {
    pub(super) async fn row_group_data_to_df(
        &self,
        row_group_data: RowGroupData,
    ) -> PolarsResult<Vec<DataFrame>> {
        let row_group_data = Arc::new(row_group_data);

        let out_width = self.row_index.is_some() as usize
            + self.projected_arrow_fields.len()
            + self.hive_partitions_width
            + self.include_file_paths.is_some() as usize;

        let mut out_columns = Vec::with_capacity(out_width);

        if self.row_index.is_some() {
            // Add a placeholder so that we don't have to shift the entire vec
            // later.
            out_columns.push(Series::default());
        }

        let slice_range = row_group_data
            .slice
            .map(|(offset, len)| offset..offset + len)
            .unwrap_or(0..row_group_data.row_group_metadata.num_rows());

        let projected_arrow_fields = &self.projected_arrow_fields;
        let projected_arrow_fields = projected_arrow_fields.clone();

        let row_group_data_2 = row_group_data.clone();
        let slice_range_2 = slice_range.clone();

        // Minimum number of values to amortize the overhead of spawning tasks.
        // This value is arbitrarily chosen.
        const VALUES_PER_THREAD: usize = 16_777_216;
        let n_rows = row_group_data.row_group_metadata.num_rows();
        let cols_per_task = 1 + VALUES_PER_THREAD / n_rows;

        let decode_fut_iter = (0..self.projected_arrow_fields.len())
            .step_by(cols_per_task)
            .map(move |offset| {
                let row_group_data = row_group_data_2.clone();
                let slice_range = slice_range_2.clone();
                let projected_arrow_fields = projected_arrow_fields.clone();

                async move {
                    (offset
                        ..offset
                            .saturating_add(cols_per_task)
                            .min(projected_arrow_fields.len()))
                        .map(|i| {
                            let arrow_field = projected_arrow_fields[i].clone();

                            let columns_to_deserialize = row_group_data
                                .row_group_metadata
                                .columns_under_root_iter(&arrow_field.name)
                                .map(|col_md| {
                                    let byte_range = col_md.byte_range();

                                    (
                                        col_md,
                                        row_group_data.byte_source.get_range(
                                            byte_range.start as usize..byte_range.end as usize,
                                        ),
                                    )
                                })
                                .collect::<Vec<_>>();

                            assert!(
                                slice_range.end <= row_group_data.row_group_metadata.num_rows()
                            );

                            let array = polars_io::prelude::_internal::to_deserializer(
                                columns_to_deserialize,
                                arrow_field.clone(),
                                Some(polars_parquet::read::Filter::Range(slice_range.clone())),
                            )?;

                            let series = Series::try_from((&arrow_field, array))?;

                            // TODO: Also load in the metadata.

                            PolarsResult::Ok(series)
                        })
                        .collect::<PolarsResult<Vec<_>>>()
                }
            });

        if decode_fut_iter.len() > 1 {
            for handle in decode_fut_iter.map(|fut| {
                async_executor::AbortOnDropHandle::new(async_executor::spawn(
                    TaskPriority::Low,
                    fut,
                ))
            }) {
                out_columns.extend(handle.await?);
            }
        } else {
            for fut in decode_fut_iter {
                out_columns.extend(fut.await?);
            }
        }

        let projection_height = if self.projected_arrow_fields.is_empty() {
            slice_range.len()
        } else {
            debug_assert!(out_columns.len() > self.row_index.is_some() as usize);
            out_columns.last().unwrap().len()
        };

        if let Some(RowIndex { name, offset }) = self.row_index.as_ref() {
            let Some(offset) = (|| {
                let offset = offset
                    .checked_add((row_group_data.row_offset + slice_range.start) as IdxSize)?;
                offset.checked_add(projection_height as IdxSize)?;

                Some(offset)
            })() else {
                let msg = format!(
                    "adding a row index column with offset {} overflows at {} rows",
                    offset,
                    row_group_data.row_offset + slice_range.end
                );
                polars_bail!(ComputeError: msg)
            };

            // The DataFrame can be empty at this point if no columns were projected from the file,
            // so we create the row index column manually instead of using `df.with_row_index` to
            // ensure it has the correct number of rows.
            let mut ca = IdxCa::from_vec(
                name.clone(),
                (offset..offset + projection_height as IdxSize).collect(),
            );
            ca.set_sorted_flag(IsSorted::Ascending);

            out_columns[0] = ca.into_series();
        }

        let shared_file_state = row_group_data
            .shared_file_state
            .get_or_init(|| async {
                let path_index = row_group_data.path_index;

                let hive_series = if let Some(hp) = self.hive_partitions.as_deref() {
                    let mut v = hp[path_index].materialize_partition_columns();
                    for s in v.iter_mut() {
                        *s = s.new_from_index(0, row_group_data.file_max_row_group_height);
                    }
                    v
                } else {
                    vec![]
                };

                let file_path_series = self.include_file_paths.clone().map(|file_path_col| {
                    StringChunked::full(
                        file_path_col,
                        self.scan_sources
                            .get(path_index)
                            .unwrap()
                            .to_include_path_name(),
                        row_group_data.file_max_row_group_height,
                    )
                    .into_series()
                });

                SharedFileState {
                    path_index,
                    hive_series,
                    file_path_series,
                }
            })
            .await;

        assert_eq!(shared_file_state.path_index, row_group_data.path_index);

        for s in &shared_file_state.hive_series {
            debug_assert!(s.len() >= projection_height);
            out_columns.push(s.slice(0, projection_height));
        }

        if let Some(file_path_series) = &shared_file_state.file_path_series {
            debug_assert!(file_path_series.len() >= projection_height);
            out_columns.push(file_path_series.slice(0, projection_height));
        }

        let df = unsafe { DataFrame::new_no_checks(out_columns) };

        // Re-calculate: A slice may have been applied.
        let cols_per_task = 1 + VALUES_PER_THREAD / df.height();

        let df = if let Some(predicate) = self.physical_predicate.as_deref() {
            let mask = predicate.evaluate_io(&df)?;
            let mask = mask.bool().unwrap();

            if cols_per_task <= df.width() {
                df._filter_seq(mask)?
            } else {
                let mask = mask.clone();
                let cols = Arc::new(df.take_columns());
                let mut out_cols = Vec::with_capacity(cols.len());

                for handle in (0..cols.len())
                    .step_by(cols_per_task)
                    .map(move |offset| {
                        let cols = cols.clone();
                        let mask = mask.clone();
                        async move {
                            cols[offset..offset.saturating_add(cols_per_task).min(cols.len())]
                                .iter()
                                .map(|s| s.filter(&mask))
                                .collect::<PolarsResult<Vec<_>>>()
                        }
                    })
                    .map(|fut| {
                        async_executor::AbortOnDropHandle::new(async_executor::spawn(
                            TaskPriority::Low,
                            fut,
                        ))
                    })
                {
                    out_cols.extend(handle.await?);
                }

                unsafe { DataFrame::new_no_checks(out_cols) }
            }
        } else {
            df
        };

        assert_eq!(df.width(), out_width);

        let n_morsels = if df.height() > 3 * self.ideal_morsel_size / 2 {
            // num_rows > (1.5 * ideal_morsel_size)
            (df.height() / self.ideal_morsel_size).max(2)
        } else {
            1
        } as u64;

        if n_morsels == 1 {
            return Ok(vec![df]);
        }

        let rows_per_morsel = 1 + df.height() / n_morsels as usize;

        let out = (0..i64::try_from(df.height()).unwrap())
            .step_by(rows_per_morsel)
            .map(|offset| df.slice(offset, rows_per_morsel))
            .collect::<Vec<_>>();

        Ok(out)
    }
}

/// State shared across row groups for a single file.
pub(super) struct SharedFileState {
    path_index: usize,
    hive_series: Vec<Series>,
    file_path_series: Option<Series>,
}
