use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::prelude::{
    AnyValue, ArrowField, ArrowSchema, BooleanChunked, Column, DataType, IdxCa, IntoColumn,
};
use polars_core::scalar::Scalar;
use polars_core::series::{IsSorted, Series};
use polars_core::utils::arrow::bitmap::{Bitmap, BitmapBuilder};
use polars_error::{polars_bail, PolarsResult};
use polars_io::hive;
use polars_io::predicates::PhysicalIoExpr;
use polars_io::prelude::_internal::calc_prefilter_cost;
pub use polars_io::prelude::_internal::PrefilterMaskSetting;
use polars_io::prelude::try_set_sorted_flag;
use polars_plan::plans::hive::HivePartitions;
use polars_plan::plans::ScanSources;
use polars_utils::index::AtomicIdxSize;
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
    pub(super) reader_schema: Arc<ArrowSchema>,
    pub(super) projected_arrow_schema: Arc<ArrowSchema>,
    pub(super) row_index: Option<Arc<(PlSmallStr, AtomicIdxSize)>>,
    pub(super) physical_predicate: Option<Arc<dyn PhysicalIoExpr>>,
    pub(super) use_prefiltered: Option<PrefilterMaskSetting>,
    /// Indices into `projected_arrow_schema. This must be sorted.
    pub(super) predicate_arrow_field_indices: Vec<usize>,
    /// Indices into `projected_arrow_schema. This must be sorted.
    pub(super) non_predicate_arrow_field_indices: Vec<usize>,
    pub(super) min_values_per_thread: usize,
}

impl RowGroupDecoder {
    pub(super) async fn row_group_data_to_df(
        &self,
        row_group_data: RowGroupData,
    ) -> PolarsResult<DataFrame> {
        if self.use_prefiltered.is_some() {
            self.row_group_data_to_df_prefiltered(row_group_data).await
        } else {
            self.row_group_data_to_df_impl(row_group_data).await
        }
    }

    async fn row_group_data_to_df_impl(
        &self,
        row_group_data: RowGroupData,
    ) -> PolarsResult<DataFrame> {
        let row_group_data = Arc::new(row_group_data);

        let out_width = self.row_index.is_some() as usize
            + self.projected_arrow_schema.len()
            + self.hive_partitions_width
            + self.include_file_paths.is_some() as usize;

        let mut out_columns = Vec::with_capacity(out_width);

        let slice_range = row_group_data
            .slice
            .map(|(offset, len)| offset..offset + len)
            .unwrap_or(0..row_group_data.row_group_metadata.num_rows());

        assert!(slice_range.end <= row_group_data.row_group_metadata.num_rows());

        if let Some(s) = self.materialize_row_index(row_group_data.as_ref(), slice_range.clone())? {
            out_columns.push(s);
        }

        let mut decoded_cols = Vec::with_capacity(row_group_data.row_group_metadata.n_columns());
        self.decode_projected_columns(
            &mut decoded_cols,
            &row_group_data,
            Some(polars_parquet::read::Filter::Range(slice_range.clone())),
        )
        .await?;

        let projection_height = slice_range.len();

        let shared_file_state = row_group_data
            .shared_file_state
            .get_or_init(|| self.shared_file_state_init_func(&row_group_data))
            .await;

        assert_eq!(shared_file_state.path_index, row_group_data.path_index);

        let mut hive_cols_iter = shared_file_state.hive_series.iter().map(|s| {
            debug_assert!(s.len() >= projection_height);
            s.slice(0, projection_height)
        });

        hive::merge_sorted_to_schema_order(
            &mut decoded_cols.into_iter(),
            &mut hive_cols_iter,
            &self.reader_schema,
            &mut out_columns,
        );

        if let Some(file_path_series) = &shared_file_state.file_path_series {
            debug_assert!(file_path_series.len() >= projection_height);
            out_columns.push(file_path_series.slice(0, projection_height));
        }

        let df = unsafe { DataFrame::new_no_checks(projection_height, out_columns) };

        let df = if let Some(predicate) = self.physical_predicate.as_deref() {
            let mask = predicate.evaluate_io(&df)?;
            let mask = mask.bool().unwrap();

            let filtered =
                unsafe { filter_cols(df.take_columns(), mask, self.min_values_per_thread) }.await?;

            let height = if let Some(fst) = filtered.first() {
                fst.len()
            } else {
                mask.num_trues()
            };

            unsafe { DataFrame::new_no_checks(height, filtered) }
        } else {
            df
        };

        assert_eq!(df.width(), out_width); // `out_width` should have been calculated correctly

        Ok(df)
    }

    async fn shared_file_state_init_func(&self, row_group_data: &RowGroupData) -> SharedFileState {
        let path_index = row_group_data.path_index;

        let hive_series = if let Some(hp) = self.hive_partitions.as_deref() {
            let v = hp[path_index].materialize_partition_columns();
            v.into_iter()
                .map(|s| {
                    s.into_column()
                        .new_from_index(0, row_group_data.file_max_row_group_height)
                })
                .collect()
        } else {
            vec![]
        };

        // @scalar-opt
        let file_path_series = self.include_file_paths.clone().map(|file_path_col| {
            Column::new_scalar(
                file_path_col,
                Scalar::new(
                    DataType::String,
                    AnyValue::StringOwned(
                        self.scan_sources
                            .get(path_index)
                            .unwrap()
                            .to_include_path_name()
                            .into(),
                    ),
                ),
                row_group_data.file_max_row_group_height,
            )
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
    ) -> PolarsResult<Option<Column>> {
        if let Some((name, offset)) = self.row_index.as_deref() {
            let offset = offset.load(std::sync::atomic::Ordering::Relaxed);
            let projection_height = slice_range.len();

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

            Ok(Some(ca.into_column()))
        } else {
            Ok(None)
        }
    }

    /// Potentially parallelizes based on number of rows & columns. Decoded columns are appended to
    /// `out_vec`.
    async fn decode_projected_columns(
        &self,
        out_vec: &mut Vec<Column>,
        row_group_data: &Arc<RowGroupData>,
        filter: Option<polars_parquet::read::Filter>,
    ) -> PolarsResult<()> {
        let projected_arrow_schema = &self.projected_arrow_schema;
        let expected_num_rows = filter
            .as_ref()
            .map_or(row_group_data.row_group_metadata.num_rows(), |x| {
                x.num_rows()
            });

        let Some((cols_per_thread, remainder)) = calc_cols_per_thread(
            row_group_data.row_group_metadata.num_rows(),
            projected_arrow_schema.len(),
            self.min_values_per_thread,
        ) else {
            // Single-threaded
            for s in projected_arrow_schema.iter_values().map(|arrow_field| {
                decode_column(
                    arrow_field,
                    row_group_data,
                    filter.clone(),
                    expected_num_rows,
                )
            }) {
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

                                decode_column(
                                    arrow_field,
                                    &row_group_data,
                                    filter.clone(),
                                    expected_num_rows,
                                )
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
            .map(|arrow_field| {
                decode_column(
                    arrow_field,
                    row_group_data,
                    filter.clone(),
                    expected_num_rows,
                )
            })
        {
            out_vec.push(out?);
        }

        for handle in task_handles {
            out_vec.extend(handle.await?);
        }

        Ok(())
    }
}

fn decode_column(
    arrow_field: &ArrowField,
    row_group_data: &RowGroupData,
    filter: Option<polars_parquet::read::Filter>,
    expected_num_rows: usize,
) -> PolarsResult<Column> {
    let Some(iter) = row_group_data
        .row_group_metadata
        .columns_under_root_iter(&arrow_field.name)
    else {
        return Ok(Column::full_null(
            arrow_field.name.clone(),
            expected_num_rows,
            &DataType::from_arrow_field(arrow_field),
        ));
    };

    let columns_to_deserialize = iter
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

    assert_eq!(array.len(), expected_num_rows);

    let mut series = Series::try_from((arrow_field, array))?;

    if let Some(col_idxs) = row_group_data
        .row_group_metadata
        .columns_idxs_under_root_iter(&arrow_field.name)
    {
        if col_idxs.len() == 1 {
            try_set_sorted_flag(&mut series, col_idxs[0], &row_group_data.sorting_map);
        }
    }

    // TODO: Also load in the metadata.

    Ok(series.into_column())
}

/// # Safety
/// All series in `cols` have the same length.
async unsafe fn filter_cols(
    mut cols: Vec<Column>,
    mask: &BooleanChunked,
    min_values_per_thread: usize,
) -> PolarsResult<Vec<Column>> {
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
    hive_series: Vec<Column>,
    file_path_series: Option<Column>,
}

// Pre-filtered

impl RowGroupDecoder {
    async fn row_group_data_to_df_prefiltered(
        &self,
        row_group_data: RowGroupData,
    ) -> PolarsResult<DataFrame> {
        debug_assert!(row_group_data.slice.is_none()); // Invariant of the optimizer.
        assert!(self.predicate_arrow_field_indices.len() <= self.projected_arrow_schema.len());

        let prefilter_setting = self.use_prefiltered.as_ref().unwrap();
        let row_group_data = Arc::new(row_group_data);
        let projection_height = row_group_data.row_group_metadata.num_rows();

        let shared_file_state = row_group_data
            .shared_file_state
            .get_or_init(|| self.shared_file_state_init_func(&row_group_data))
            .await;

        assert_eq!(shared_file_state.path_index, row_group_data.path_index);

        let mut live_columns = Vec::with_capacity(
            self.row_index.is_some() as usize
                + self.predicate_arrow_field_indices.len()
                + self.hive_partitions_width
                + self.include_file_paths.is_some() as usize,
        );

        if let Some(s) = self.materialize_row_index(
            row_group_data.as_ref(),
            0..row_group_data.row_group_metadata.num_rows(),
        )? {
            live_columns.push(s);
        }

        // Materialize file and hive columns in sorted order - this is important for correct merging
        // later.
        //
        // We do a trick to turn `Iterator<Item = Result<Column>>` into `Iterator<Item = Column>`
        // for `hive::merge_sorted_to_schema_order`.
        let mut opt_decode_err = None;

        let decoded_live_cols_iter = self
            .predicate_arrow_field_indices
            .iter()
            .map(|&i| self.projected_arrow_schema.get_at_index(i).unwrap())
            .map(|(_, arrow_field)| {
                let res = decode_column(arrow_field, &row_group_data, None, projection_height);

                match res {
                    Ok(c) => c,
                    e @ Err(_) => {
                        opt_decode_err.replace(e);
                        Default::default()
                    },
                }
            });
        let hive_cols_iter = shared_file_state.hive_series.iter().map(|s| {
            debug_assert!(s.len() >= projection_height);
            s.slice(0, projection_height)
        });

        live_columns.extend(decoded_live_cols_iter);
        live_columns.extend(hive_cols_iter);
        opt_decode_err.transpose()?;

        if let Some(file_path_series) = &shared_file_state.file_path_series {
            debug_assert!(file_path_series.len() >= projection_height);
            live_columns.push(file_path_series.slice(0, projection_height));
        }

        let mut live_df = unsafe {
            DataFrame::new_no_checks(row_group_data.row_group_metadata.num_rows(), live_columns)
        };

        let mask = self
            .physical_predicate
            .as_deref()
            .unwrap()
            .evaluate_io(&live_df)?;
        let mask = mask.bool().unwrap();

        unsafe {
            live_df.get_columns_mut().truncate(
                self.row_index.is_some() as usize + self.predicate_arrow_field_indices.len(),
            )
        }

        let filtered =
            unsafe { filter_cols(live_df.take_columns(), mask, self.min_values_per_thread) }
                .await?;

        let filtered_height = if let Some(fst) = filtered.first() {
            fst.len()
        } else {
            mask.num_trues()
        };

        let mut live_df_filtered = unsafe { DataFrame::new_no_checks(filtered_height, filtered) };

        if self.non_predicate_arrow_field_indices.is_empty() {
            // User or test may have explicitly requested prefiltering

            hive::merge_sorted_to_schema_order(
                unsafe {
                    &mut live_df_filtered
                        .get_columns_mut()
                        .drain(..)
                        .collect::<Vec<_>>()
                        .into_iter()
                },
                &mut shared_file_state
                    .hive_series
                    .iter()
                    .map(|s| s.slice(0, filtered_height)),
                &self.reader_schema,
                unsafe { live_df_filtered.get_columns_mut() },
            );

            unsafe {
                live_df_filtered.get_columns_mut().extend(
                    shared_file_state
                        .file_path_series
                        .as_ref()
                        .map(|c| c.slice(0, filtered_height)),
                )
            }

            return Ok(live_df_filtered);
        }

        let mask_bitmap = {
            let mut mask_bitmap = BitmapBuilder::with_capacity(mask.len());

            for chunk in mask.downcast_iter() {
                match chunk.validity() {
                    None => mask_bitmap.extend_from_bitmap(chunk.values()),
                    Some(validity) => mask_bitmap.extend_from_bitmap(&(validity & chunk.values())),
                }
            }

            mask_bitmap.freeze()
        };

        assert_eq!(mask_bitmap.len(), projection_height);

        let prefilter_cost = calc_prefilter_cost(&mask_bitmap);
        let expected_num_rows = mask_bitmap.set_bits();

        let mut opt_decode_err = None;

        let mut dead_cols_decode_iter = self
            .non_predicate_arrow_field_indices
            .iter()
            .map(|&i| self.projected_arrow_schema.get_at_index(i).unwrap())
            .map(|(_, arrow_field)| {
                match decode_column_prefiltered(
                    arrow_field,
                    &row_group_data,
                    prefilter_cost,
                    prefilter_setting,
                    mask,
                    &mask_bitmap,
                    expected_num_rows,
                ) {
                    Ok(v) => v,
                    e @ Err(_) => {
                        opt_decode_err.replace(e);
                        Column::default()
                    },
                }
            });

        let live_columns = live_df_filtered.take_columns();

        // dead_columns
        // [ ..arrow_fields ]
        // live_df_filtered
        // [ row_index?, ..arrow_fields, ..hive_cols, file_path? ]
        // We re-use `hive::merge_sorted_to_schema_order()` as it performs most of the merge operation we want.
        // But we take out the `row_index` column as it isn't on the correct side.

        let mut merged = Vec::with_capacity(live_columns.len() + dead_cols_decode_iter.len());

        if self.row_index.is_some() {
            merged.push(live_columns[0].clone());
        };

        hive::merge_sorted_to_schema_order(
            &mut dead_cols_decode_iter, // df_columns
            &mut live_columns
                .into_iter()
                .skip(self.row_index.is_some() as usize), // hive_columns
            &self.projected_arrow_schema,
            &mut merged,
        );

        opt_decode_err.transpose()?;

        let mut out = Vec::with_capacity(
            merged.len()
                + shared_file_state.hive_series.len()
                + shared_file_state.file_path_series.is_some() as usize,
        );

        hive::merge_sorted_to_schema_order(
            &mut merged.into_iter(),
            &mut shared_file_state
                .hive_series
                .iter()
                .map(|s| s.slice(0, filtered_height)),
            &self.reader_schema,
            &mut out,
        );

        out.extend(
            shared_file_state
                .file_path_series
                .as_ref()
                .map(|c| c.slice(0, filtered_height)),
        );

        let df = unsafe { DataFrame::new_no_checks(expected_num_rows, out) };
        Ok(df)
    }
}

fn decode_column_prefiltered(
    arrow_field: &ArrowField,
    row_group_data: &RowGroupData,
    prefilter_cost: f64,
    prefilter_setting: &PrefilterMaskSetting,
    mask: &BooleanChunked,
    mask_bitmap: &Bitmap,
    expected_num_rows: usize,
) -> PolarsResult<Column> {
    let Some(iter) = row_group_data
        .row_group_metadata
        .columns_under_root_iter(&arrow_field.name)
    else {
        return Ok(Column::full_null(
            arrow_field.name.clone(),
            expected_num_rows,
            &DataType::from_arrow_field(arrow_field),
        ));
    };

    let columns_to_deserialize = iter
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

    let prefilter = prefilter_setting.should_prefilter(prefilter_cost, &arrow_field.dtype);

    let deserialize_filter =
        prefilter.then(|| polars_parquet::read::Filter::Mask(mask_bitmap.clone()));

    let array = polars_io::prelude::_internal::to_deserializer(
        columns_to_deserialize,
        arrow_field.clone(),
        deserialize_filter,
    )?;

    let mut series = Series::try_from((arrow_field, array))?;

    if let Some(col_idxs) = row_group_data
        .row_group_metadata
        .columns_idxs_under_root_iter(&arrow_field.name)
    {
        if col_idxs.len() == 1 {
            try_set_sorted_flag(&mut series, col_idxs[0], &row_group_data.sorting_map);
        }
    }

    let series = if !prefilter {
        series.filter(mask)?
    } else {
        series
    };

    assert_eq!(series.len(), expected_num_rows);

    Ok(series.into_column())
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
