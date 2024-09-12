use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::prelude::{
    ArrowField, ArrowSchema, BooleanChunked, ChunkFull, IdxCa, StringChunked,
};
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
    pub(super) projected_arrow_schema: Arc<ArrowSchema>,
    pub(super) row_index: Option<RowIndex>,
    pub(super) physical_predicate: Option<Arc<dyn PhysicalIoExpr>>,
    pub(super) use_prefiltered: bool,
    pub(super) ideal_morsel_size: usize,
    pub(super) min_values_per_thread: usize,
}

impl RowGroupDecoder {
    pub(super) async fn row_group_data_to_df(
        &self,
        row_group_data: RowGroupData,
    ) -> PolarsResult<Vec<DataFrame>> {
        if self.use_prefiltered {
            self.row_group_data_to_df_prefiltered(row_group_data).await
        } else {
            self.row_group_data_to_df_impl(row_group_data).await
        }
    }

    async fn row_group_data_to_df_impl(
        &self,
        row_group_data: RowGroupData,
    ) -> PolarsResult<Vec<DataFrame>> {
        let row_group_data = Arc::new(row_group_data);

        let out_width = self.row_index.is_some() as usize
            + self.projected_arrow_schema.len()
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

        assert!(slice_range.end <= row_group_data.row_group_metadata.num_rows());

        self.decode_all_columns(
            &mut out_columns,
            &row_group_data,
            Some(polars_parquet::read::Filter::Range(slice_range.clone())),
        )
        .await?;

        let projection_height = if self.projected_arrow_schema.is_empty() {
            slice_range.len()
        } else {
            debug_assert!(out_columns.len() > self.row_index.is_some() as usize);
            out_columns.last().unwrap().len()
        };

        if let Some(s) = self.materialize_row_index(row_group_data.as_ref(), slice_range)? {
            out_columns[0] = s;
        }

        let shared_file_state = row_group_data
            .shared_file_state
            .get_or_init(|| self.shared_file_state_init_func(&row_group_data))
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

        let df = if let Some(predicate) = self.physical_predicate.as_deref() {
            let mask = predicate.evaluate_io(&df)?;
            let mask = mask.bool().unwrap();

            unsafe {
                DataFrame::new_no_checks(
                    filter_cols(df.take_columns(), mask, self.min_values_per_thread).await?,
                )
            }
        } else {
            df
        };

        assert_eq!(df.width(), out_width); // `out_width` should have been calculated correctly

        Ok(self.split_to_morsels(df))
    }

    async fn shared_file_state_init_func(&self, row_group_data: &RowGroupData) -> SharedFileState {
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
    }

    fn materialize_row_index(
        &self,
        row_group_data: &RowGroupData,
        slice_range: core::ops::Range<usize>,
    ) -> PolarsResult<Option<Series>> {
        if let Some(RowIndex { name, offset }) = self.row_index.as_ref() {
            let projection_height = row_group_data.row_group_metadata.num_rows();

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

            Ok(Some(ca.into_series()))
        } else {
            Ok(None)
        }
    }

    /// Potentially parallelizes based on number of rows & columns. Decoded columns are appended to
    /// `out_vec`.
    async fn decode_all_columns(
        &self,
        out_vec: &mut Vec<Series>,
        row_group_data: &Arc<RowGroupData>,
        filter: Option<polars_parquet::read::Filter>,
    ) -> PolarsResult<()> {
        let projected_arrow_schema = &self.projected_arrow_schema;

        let Some((cols_per_thread, remainder)) = calc_cols_per_thread(
            row_group_data.row_group_metadata.num_rows(),
            projected_arrow_schema.len(),
            self.min_values_per_thread,
        ) else {
            // Single-threaded
            for s in projected_arrow_schema
                .iter_values()
                .map(|arrow_field| decode_column(arrow_field, row_group_data, filter.clone()))
            {
                out_vec.push(s?)
            }

            return Ok(());
        };

        let projected_arrow_schema = projected_arrow_schema.clone();
        let row_group_data_2 = row_group_data.clone();

        let task_handles = {
            let projected_arrow_schema = projected_arrow_schema.clone();
            let filter = filter.clone();

            (remainder..projected_arrow_schema.len())
                .step_by(cols_per_thread)
                .map(move |offset| {
                    let row_group_data = row_group_data_2.clone();
                    let projected_arrow_schema = projected_arrow_schema.clone();
                    let filter = filter.clone();

                    async move {
                        // This is exact as we have already taken out the remainder.
                        (offset..offset + cols_per_thread)
                            .map(|i| {
                                let (_, arrow_field) =
                                    projected_arrow_schema.get_at_index(i).unwrap();

                                decode_column(arrow_field, &row_group_data, filter.clone())
                            })
                            .collect::<PolarsResult<Vec<_>>>()
                    }
                })
                .map(|fut| {
                    async_executor::AbortOnDropHandle::new(async_executor::spawn(
                        TaskPriority::Low,
                        fut,
                    ))
                })
                .collect::<Vec<_>>()
        };

        for out in projected_arrow_schema
            .iter_values()
            .take(remainder)
            .map(|arrow_field| decode_column(arrow_field, row_group_data, filter.clone()))
        {
            out_vec.push(out?);
        }

        for handle in task_handles {
            out_vec.extend(handle.await?);
        }

        Ok(())
    }

    fn split_to_morsels(&self, df: DataFrame) -> Vec<DataFrame> {
        let n_morsels = if df.height() > 3 * self.ideal_morsel_size / 2 {
            // num_rows > (1.5 * ideal_morsel_size)
            (df.height() / self.ideal_morsel_size).max(2)
        } else {
            1
        } as u64;

        if n_morsels == 1 {
            return vec![df];
        }

        let rows_per_morsel = 1 + df.height() / n_morsels as usize;

        (0..i64::try_from(df.height()).unwrap())
            .step_by(rows_per_morsel)
            .map(|offset| df.slice(offset, rows_per_morsel))
            .collect::<Vec<_>>()
    }
}

fn decode_column(
    arrow_field: &ArrowField,
    row_group_data: &RowGroupData,
    filter: Option<polars_parquet::read::Filter>,
) -> PolarsResult<Series> {
    let columns_to_deserialize = row_group_data
        .row_group_metadata
        .columns_under_root_iter(&arrow_field.name)
        .map(|col_md| {
            let byte_range = col_md.byte_range();

            (
                col_md,
                row_group_data
                    .fetched_bytes
                    .get_range(byte_range.start as usize..byte_range.end as usize),
            )
        })
        .collect::<Vec<_>>();

    let array = polars_io::prelude::_internal::to_deserializer(
        columns_to_deserialize,
        arrow_field.clone(),
        filter,
    )?;

    let series = Series::try_from((arrow_field, array))?;

    // TODO: Also load in the metadata.

    Ok(series)
}

/// # Safety
/// All series in `cols` have the same length.
async unsafe fn filter_cols(
    mut cols: Vec<Series>,
    mask: &BooleanChunked,
    min_values_per_thread: usize,
) -> PolarsResult<Vec<Series>> {
    if cols.is_empty() {
        return Ok(cols);
    }

    let Some((cols_per_thread, remainder)) =
        calc_cols_per_thread(cols[0].len(), cols.len(), min_values_per_thread)
    else {
        for s in cols.iter_mut() {
            *s = s.filter(mask)?;
        }

        return Ok(cols);
    };

    let mut out_vec = Vec::with_capacity(cols.len());
    let cols = Arc::new(cols);
    let mask = mask.clone();

    let task_handles = {
        let cols = &cols;
        let mask = &mask;

        (remainder..cols.len())
            .step_by(cols_per_thread)
            .map(move |offset| {
                let cols = cols.clone();
                let mask = mask.clone();
                async move {
                    (offset..offset + cols_per_thread)
                        .map(|i| cols[i].filter(&mask))
                        .collect::<PolarsResult<Vec<_>>>()
                }
            })
            .map(|fut| {
                async_executor::AbortOnDropHandle::new(async_executor::spawn(
                    TaskPriority::Low,
                    fut,
                ))
            })
            .collect::<Vec<_>>()
    };

    for out in cols.iter().take(remainder).map(|s| s.filter(&mask)) {
        out_vec.push(out?);
    }

    for handle in task_handles {
        out_vec.extend(handle.await?)
    }

    Ok(out_vec)
}

/// Returns `Some((n_cols_per_thread, n_remainder))` if at least 2 tasks with >= `min_values_per_thread` can be created.
fn calc_cols_per_thread(
    n_rows_per_col: usize,
    n_cols: usize,
    min_values_per_thread: usize,
) -> Option<(usize, usize)> {
    let cols_per_thread = 1 + min_values_per_thread / n_rows_per_col.max(1);

    let cols_per_thread = if n_rows_per_col >= min_values_per_thread {
        1
    } else {
        cols_per_thread
    };

    // At least 2 fully saturated tasks according to floordiv.
    let parallel = n_cols / cols_per_thread >= 2;
    let remainder = n_cols % cols_per_thread;

    parallel.then_some((cols_per_thread, remainder))
}

/// State shared across row groups for a single file.
pub(super) struct SharedFileState {
    path_index: usize,
    hive_series: Vec<Series>,
    file_path_series: Option<Series>,
}

///
/// Pre-filtered
///

impl RowGroupDecoder {
    async fn row_group_data_to_df_prefiltered(
        &self,
        row_group_data: RowGroupData,
    ) -> PolarsResult<Vec<DataFrame>> {
        // TODO: actually prefilter
        self.row_group_data_to_df_impl(row_group_data).await
    }
}

mod tests {
    #[test]
    fn test_calc_cols_per_thread() {
        use super::calc_cols_per_thread;

        let n_rows = 3;
        let n_cols = 11;
        let min_vals = 5;
        assert_eq!(calc_cols_per_thread(n_rows, n_cols, min_vals), Some((2, 1)));

        let n_rows = 6;
        let n_cols = 11;
        let min_vals = 5;
        assert_eq!(calc_cols_per_thread(n_rows, n_cols, min_vals), Some((1, 0)));

        calc_cols_per_thread(0, 1, 1);
        calc_cols_per_thread(1, 0, 1);
    }
}
