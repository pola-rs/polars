use std::ops::Deref;
use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::prelude::{
    ArrowField, ArrowSchema, BooleanChunked, ChunkFilter, Column, DataType, IntoColumn,
};
use polars_core::series::Series;
use polars_core::utils::arrow::bitmap::{Bitmap, MutableBitmap};
use polars_error::PolarsResult;
use polars_io::RowIndex;
use polars_io::predicates::{
    ColumnPredicateExpr, ColumnPredicates, ScanIOPredicate, SpecializedColumnPredicateExpr,
};
pub use polars_io::prelude::_internal::PrefilterMaskSetting;
use polars_io::prelude::_internal::calc_prefilter_cost;
use polars_io::prelude::try_set_sorted_flag;
use polars_parquet::read::{Filter, ParquetType, PredicateFilter, PrimitiveLogicalType};
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;

use super::row_group_data_fetch::RowGroupData;
use crate::async_primitives::opt_spawned_future::parallelize_first_to_local;

/// Turns row group data into DataFrames.
pub(super) struct RowGroupDecoder {
    pub(super) num_pipelines: usize,
    pub(super) projected_arrow_schema: Arc<ArrowSchema>,
    pub(super) row_index: Option<RowIndex>,
    pub(super) predicate: Option<ScanIOPredicate>,
    pub(super) use_prefiltered: Option<PrefilterMaskSetting>,
    /// Indices into `projected_arrow_schema. This must be sorted.
    pub(super) predicate_arrow_field_indices: Arc<Vec<usize>>,
    /// Indices into `projected_arrow_schema. This must be sorted.
    pub(super) non_predicate_arrow_field_indices: Arc<Vec<usize>>,
    pub(super) min_values_per_thread: usize,
}

impl RowGroupDecoder {
    pub(super) async fn row_group_data_to_df(
        &self,
        mut row_group_data: RowGroupData,
    ) -> PolarsResult<DataFrame> {
        // If the slice consumes the entire row-group. Don't slice. This allows for prefiltering to
        // happen more often until we properly support prefiltering with pre-slices.
        row_group_data.slice.take_if(|slice| {
            slice.0 == 0 && slice.1 >= row_group_data.row_group_metadata.num_rows()
        });

        if self.use_prefiltered.is_some()
            && row_group_data.slice.is_none()
            && !self.predicate_arrow_field_indices.is_empty()
        {
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

        let out_width = self.row_index.is_some() as usize + self.projected_arrow_schema.len();

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

        out_columns.extend(decoded_cols);

        let df = unsafe { DataFrame::new_no_checks(projection_height, out_columns) };

        let df = if let Some(predicate) = self.predicate.as_ref() {
            let mask = predicate.predicate.evaluate_io(&df)?;
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

    fn materialize_row_index(
        &self,
        row_group_data: &RowGroupData,
        slice_range: core::ops::Range<usize>,
    ) -> PolarsResult<Option<Column>> {
        if let Some(RowIndex { name, offset }) = self.row_index.clone() {
            let projection_height = slice_range.len();

            let offset = offset.saturating_add(
                IdxSize::try_from(row_group_data.row_offset + slice_range.start)
                    .unwrap_or(IdxSize::MAX),
            );

            // The DataFrame can be empty at this point if no columns were projected from the file,
            // so we create the row index column manually instead of using `df.with_row_index` to
            // ensure it has the correct number of rows.
            Ok(Some(Column::new_row_index(
                name,
                offset,
                projection_height,
            )?))
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
                x.num_rows(row_group_data.row_group_metadata.num_rows())
            });

        let Some((cols_per_thread, _)) = calc_cols_per_thread(
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
                out_vec.push(s?.0)
            }

            return Ok(());
        };

        let projected_arrow_schema = projected_arrow_schema.clone();
        let row_group_data_2 = row_group_data.clone();

        let task_handles = {
            let projected_arrow_schema = projected_arrow_schema.clone();
            let filter = filter.clone();

            parallelize_first_to_local(
                (0..projected_arrow_schema.len())
                    .step_by(cols_per_thread)
                    .map(move |offset| {
                        let row_group_data = row_group_data_2.clone();
                        let projected_arrow_schema = projected_arrow_schema.clone();
                        let filter = filter.clone();

                        async move {
                            // This is exact as we have already taken out the remainder.
                            (offset
                                ..offset
                                    .saturating_add(cols_per_thread)
                                    .min(projected_arrow_schema.len()))
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
                    }),
            )
        };

        for fut in task_handles {
            out_vec.extend(fut.await?.into_iter().map(|(c, _)| c));
        }

        Ok(())
    }
}

fn decode_column(
    arrow_field: &ArrowField,
    row_group_data: &RowGroupData,
    filter: Option<polars_parquet::read::Filter>,
    expected_num_rows: usize,
) -> PolarsResult<(Column, Bitmap)> {
    let Some(iter) = row_group_data
        .row_group_metadata
        .columns_under_root_iter(&arrow_field.name)
    else {
        return Ok((
            Column::full_null(
                arrow_field.name.clone(),
                expected_num_rows,
                &DataType::from_arrow_field(arrow_field),
            ),
            Bitmap::default(),
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

    let skip_num_rows_check = matches!(filter, Some(Filter::Predicate(_)));

    let (array, pred_true_mask) = polars_io::prelude::_internal::to_deserializer(
        columns_to_deserialize,
        arrow_field.clone(),
        filter,
    )?;

    if !skip_num_rows_check {
        assert_eq!(array.len(), expected_num_rows);
    }

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

    Ok((series.into_column(), pred_true_mask))
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

    let Some((cols_per_thread, _)) =
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

        parallelize_first_to_local((0..cols.len()).step_by(cols_per_thread).map(move |offset| {
            let cols = cols.clone();
            let mask = mask.clone();
            async move {
                (offset..offset + cols_per_thread)
                    .map(|i| cols[i].filter(&mask))
                    .collect::<PolarsResult<Vec<_>>>()
            }
        }))
    };

    for fut in task_handles {
        out_vec.extend(fut.await?)
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

// Pre-filtered

fn decode_column_in_filter(
    arrow_field: &ArrowField,
    use_column_predicates: bool,
    column_predicates: &ColumnPredicates,
    row_group_data: &RowGroupData,
    projection_height: usize,
) -> PolarsResult<(Column, Bitmap)> {
    let mut filter = None;
    let mut constant = None;
    if use_column_predicates {
        if let Some((column_predicate, specialized)) =
            column_predicates.predicates.get(&arrow_field.name)
        {
            constant = specialized.as_ref().and_then(|s| match s {
                SpecializedColumnPredicateExpr::Eq(sc) if !sc.is_null() => Some(sc),
                SpecializedColumnPredicateExpr::EqMissing(sc) => Some(sc),
                _ => None,
            });

            let p = ColumnPredicateExpr::new(
                arrow_field.name.clone(),
                DataType::from_arrow_field(arrow_field),
                column_predicate.clone(),
                specialized.clone(),
            );
            filter = Some(Filter::Predicate(PredicateFilter {
                predicate: Arc::new(p) as _,
                include_values: constant.is_none(),
            }));
        }
    }
    let (mut c, m) = decode_column(arrow_field, row_group_data, filter, projection_height)?;

    if let Some(constant) = constant {
        c = Column::new_scalar(c.name().clone(), constant.clone(), m.set_bits());
    }

    Ok((c, m))
}

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

        let mut live_columns = Vec::with_capacity(
            self.row_index.is_some() as usize
                + self.predicate_arrow_field_indices.len()
                + self.non_predicate_arrow_field_indices.len(),
        );
        let mut masks = Vec::with_capacity(
            self.row_index.is_some() as usize + self.predicate_arrow_field_indices.len(),
        );

        if let Some(s) = self.materialize_row_index(
            row_group_data.as_ref(),
            0..row_group_data.row_group_metadata.num_rows(),
        )? {
            live_columns.push(s);
        }

        let scan_predicate = self.predicate.as_ref().unwrap();

        let use_column_predicates = scan_predicate.column_predicates.is_sumwise_complete
            && self.row_index.is_none()
            && !row_group_data
                .row_group_metadata
                .parquet_columns()
                .iter()
                .any(|c| {
                    let ParquetType::PrimitiveType(pt) = c.descriptor().base_type.deref() else {
                        return false;
                    };
                    matches!(pt.logical_type, Some(PrimitiveLogicalType::Float16))
                })
            && self
                .predicate_arrow_field_indices
                .iter()
                .map(|&i| self.projected_arrow_schema.get_at_index(i).unwrap())
                .all(|(_, arrow_field)| !arrow_field.dtype().is_nested());

        let cols_per_thread = (self
            .predicate_arrow_field_indices
            .len()
            .div_ceil(self.num_pipelines))
        .max(1);
        let task_handles = {
            let predicate_arrow_field_indices = self.predicate_arrow_field_indices.clone();
            let projected_arrow_schema = self.projected_arrow_schema.clone();
            let row_group_data = row_group_data.clone();

            parallelize_first_to_local(
                (0..self.predicate_arrow_field_indices.len())
                    .step_by(cols_per_thread)
                    .map(move |offset| {
                        let row_group_data = row_group_data.clone();
                        let predicate_arrow_field_indices = predicate_arrow_field_indices.clone();
                        let projected_arrow_schema = projected_arrow_schema.clone();
                        let column_predicates = scan_predicate.column_predicates.clone();

                        async move {
                            (offset
                                ..offset
                                    .saturating_add(cols_per_thread)
                                    .min(predicate_arrow_field_indices.len()))
                                .map(|i| {
                                    let (_, arrow_field) = projected_arrow_schema
                                        .get_at_index(predicate_arrow_field_indices[i])
                                        .unwrap();

                                    decode_column_in_filter(
                                        arrow_field,
                                        use_column_predicates,
                                        column_predicates.as_ref(),
                                        row_group_data.as_ref(),
                                        projection_height,
                                    )
                                })
                                .collect::<PolarsResult<Vec<_>>>()
                        }
                    }),
            )
        };

        for fut in task_handles {
            for (c, m) in fut.await? {
                live_columns.push(c);
                masks.push(m);
            }
        }

        let (live_df_filtered, mut mask) = if use_column_predicates {
            assert!(scan_predicate.column_predicates.is_sumwise_complete);
            if masks.len() == 1 {
                (
                    DataFrame::new(live_columns).unwrap(),
                    BooleanChunked::from_bitmap(PlSmallStr::EMPTY, masks[0].clone()),
                )
            } else {
                let mut mask = MutableBitmap::new();
                mask.extend_from_bitmap(masks.first().unwrap());
                for col_mask in &masks[1..] {
                    <&mut MutableBitmap as std::ops::BitAndAssign<&Bitmap>>::bitand_assign(
                        &mut &mut mask,
                        col_mask,
                    );
                }
                let mask = BooleanChunked::from_bitmap(PlSmallStr::EMPTY, mask.freeze());
                let live_columns = live_columns
                    .into_iter()
                    .zip(masks)
                    .map(|(col, col_mask)| {
                        let col_mask = BooleanChunked::from_bitmap(PlSmallStr::EMPTY, col_mask);
                        let col_mask = mask.filter(&col_mask).unwrap();
                        col.filter(&col_mask).unwrap()
                    })
                    .collect();

                (DataFrame::new(live_columns).unwrap(), mask)
            }
        } else {
            let mut live_df = unsafe {
                DataFrame::new_no_checks(row_group_data.row_group_metadata.num_rows(), live_columns)
            };

            let mask = scan_predicate.predicate.evaluate_io(&live_df)?;
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

            (
                unsafe { DataFrame::new_no_checks(filtered_height, filtered) },
                mask.clone(),
            )
        };

        if self.non_predicate_arrow_field_indices.is_empty() {
            // User or test may have explicitly requested prefiltering
            let iter = self.projected_arrow_schema.iter_names().cloned();
            return Ok(if let Some(ri) = self.row_index.as_ref() {
                live_df_filtered
                    .select(std::iter::once(ri.name.clone()).chain(iter))
                    .unwrap()
            } else {
                live_df_filtered.select(iter).unwrap()
            });
        }

        mask.rechunk_mut();
        let mask_bitmap = mask.downcast_as_array();
        let mask_bitmap = match mask_bitmap.validity() {
            None => mask_bitmap.values().clone(),
            Some(v) => mask_bitmap.values() & v,
        };

        assert_eq!(mask_bitmap.len(), projection_height);

        let prefilter_cost = calc_prefilter_cost(&mask_bitmap);
        let expected_num_rows = mask_bitmap.set_bits();

        let cols_per_thread = (self
            .predicate_arrow_field_indices
            .len()
            .div_ceil(self.num_pipelines))
        .max(1);
        let task_handles = {
            let non_predicate_arrow_field_indices = self.non_predicate_arrow_field_indices.clone();
            let non_predicate_len = non_predicate_arrow_field_indices.len();
            let projected_arrow_schema = self.projected_arrow_schema.clone();
            let row_group_data = row_group_data.clone();
            let prefilter_setting = *prefilter_setting;

            parallelize_first_to_local((0..non_predicate_len).step_by(cols_per_thread).map(
                move |offset| {
                    let row_group_data = row_group_data.clone();
                    let non_predicate_arrow_field_indices =
                        non_predicate_arrow_field_indices.clone();
                    let projected_arrow_schema = projected_arrow_schema.clone();
                    let mask = mask.clone();
                    let mask_bitmap = mask_bitmap.clone();
                    let max_col = offset
                        .saturating_add(cols_per_thread)
                        .min(non_predicate_len);

                    async move {
                        (offset..max_col)
                            .map(|i| {
                                let (_, arrow_field) = projected_arrow_schema
                                    .get_at_index(non_predicate_arrow_field_indices[i])
                                    .unwrap();

                                decode_column_prefiltered(
                                    arrow_field,
                                    row_group_data.as_ref(),
                                    prefilter_cost,
                                    &prefilter_setting,
                                    &mask,
                                    &mask_bitmap,
                                    expected_num_rows,
                                )
                            })
                            .collect::<PolarsResult<Vec<_>>>()
                    }
                },
            ))
        };

        let live_columns = live_df_filtered.take_columns();

        let mut dead_cols = Vec::with_capacity(self.non_predicate_arrow_field_indices.len());
        for fut in task_handles {
            dead_cols.extend(fut.await?);
        }

        let mut merged = live_columns;
        merged.extend(dead_cols);
        let df = unsafe { DataFrame::new_no_checks(expected_num_rows, merged) };
        Ok(df)
    }
}

fn decode_column_prefiltered(
    arrow_field: &ArrowField,
    row_group_data: &RowGroupData,
    _prefilter_cost: f64,
    _prefilter_setting: &PrefilterMaskSetting,
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

    let prefilter = !arrow_field.dtype.is_nested();

    let deserialize_filter =
        prefilter.then(|| polars_parquet::read::Filter::Mask(mask_bitmap.clone()));

    let (array, _) = polars_io::prelude::_internal::to_deserializer(
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
