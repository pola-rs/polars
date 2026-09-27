use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use polars_async::executor::TaskPriority;
use polars_async::primitives::opt_spawned_future::parallelize_first_to_local;
use polars_core::frame::DataFrame;
use polars_core::prelude::{ArrowField, BooleanChunked, ChunkFilter, Column, DataType, IntoColumn};
use polars_core::scalar::Scalar;
use polars_core::series::Series;
use polars_core::utils::polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_error::PolarsResult;
use polars_io::RowIndex;
use polars_io::predicates::{DynamicPredicateSource, PhysicalIoExpr, ScanIOPredicate};
use polars_io::prelude::_internal::canonicalize_parquet_maps;
use polars_io::prelude::try_set_sorted_flag;
use polars_parquet::read::{Filter, PredicateFilter, PrimitiveLogicalType};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::{IdxSize, UnitVec};

use super::passes::{Selectivity, keeps_most_rows, plan_passes, promote};
use super::row_group_data_fetch::RowGroupData;
use crate::nodes::io_sources::parquet::projection::ArrowFieldProjection;

/// Where a column of a row group comes from. Sorts in output order.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Source {
    RowIndex,
    /// Index into `RowGroupDecoder::projected_arrow_fields`.
    Field(usize),
}

/// The conjuncts of the predicate that read one column.
pub(super) struct PredicateColumn {
    pub(super) source: Source,
    /// The static conjuncts, conjoined.
    pub(super) predicate: Option<Arc<dyn PhysicalIoExpr>>,
    /// Evaluates `predicate` while decoding, when the decoder may.
    pub(super) decode_filter: Option<PredicateFilter>,
    /// The value of every kept row when `predicate` is an equality.
    pub(super) constant: Option<Scalar>,
    pub(super) dynamic: Vec<DynamicConjunct>,
}

/// A conjunct on a predicate column that a producer sets at run time. It is
/// evaluated once it is set and rejects rows, and no longer once it keeps most
/// of the rows it sees, when its source allows that.
pub(super) struct DynamicConjunct {
    pub(super) predicate: Arc<dyn PhysicalIoExpr>,
    pub(super) source: Arc<dyn DynamicPredicateSource>,
    /// Set once the conjunct was seen filtering rows, so the next row group
    /// measures it on every row.
    activated: AtomicBool,
    measured: Mutex<Selectivity>,
    bypassed: AtomicBool,
}

impl DynamicConjunct {
    pub(super) fn new(
        predicate: Arc<dyn PhysicalIoExpr>,
        source: Arc<dyn DynamicPredicateSource>,
    ) -> Self {
        Self {
            predicate,
            source,
            activated: AtomicBool::new(false),
            measured: Mutex::new(Selectivity::default()),
            bypassed: AtomicBool::new(false),
        }
    }

    fn is_active(&self) -> bool {
        !self.bypassed.load(Ordering::Relaxed) && self.source.filters_rows()
    }

    /// Whether the conjunct filters rows and was not seen doing so before.
    fn newly_active(&self) -> bool {
        self.source.filters_rows() && !self.activated.swap(true, Ordering::Relaxed)
    }

    /// Count the rows an evaluation saw and kept.
    fn measure(&self, input_rows: usize, kept_rows: usize) {
        let measured = {
            let mut measured = self.measured.lock().unwrap();
            measured.input_rows += input_rows;
            measured.kept_rows += kept_rows;
            *measured
        };
        if keeps_most_rows(measured.kept_rows, measured.input_rows)
            && self.source.can_bypass()
            && !self.bypassed.swap(true, Ordering::Relaxed)
            && polars_core::config::verbose()
        {
            eprintln!(
                "[ParquetFileReader]: Dynamic predicate bypassed, it kept {} of {} rows",
                measured.kept_rows, measured.input_rows
            );
        }
    }
}

/// Turns row group data into DataFrames.
pub(super) struct RowGroupDecoder {
    pub(super) num_pipelines: usize,
    pub(super) projected_arrow_fields: Arc<[ArrowFieldProjection]>,
    pub(super) row_index: Option<RowIndex>,
    pub(super) predicate: Option<ScanIOPredicate>,
    pub(super) use_prefiltered: bool,
    /// Indices into `projected_arrow_fields. This must be sorted.
    pub(super) predicate_field_indices: Arc<[usize]>,
    /// The conjuncts of the predicate that read one column.
    pub(super) predicate_columns: Arc<[PredicateColumn]>,
    /// The part of the predicate `predicate_columns` do not cover.
    pub(super) rest_predicate: Option<Arc<dyn PhysicalIoExpr>>,
    /// The predicate fields no predicate column reads. Sorted.
    pub(super) rest_field_indices: Arc<[usize]>,
    /// The predicate columns decoded in each pass over a row group, as indices into
    /// `predicate_columns`; the last pass also decodes `rest_field_indices`. Rows a
    /// pass rejects are not decoded by the next. Replanned after every row group.
    pub(super) passes: Mutex<Arc<Vec<Vec<usize>>>>,
    /// Indices into `projected_arrow_fields. This must be sorted.
    pub(super) non_predicate_field_indices: Arc<[usize]>,
    pub(super) target_values_per_thread: usize,
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

        let nothing_to_evaluate = self.nothing_to_evaluate();
        if self.use_prefiltered
            && row_group_data.slice.is_none()
            && !self.predicate_field_indices.is_empty()
            && !nothing_to_evaluate
        {
            self.row_group_data_to_df_prefiltered(row_group_data).await
        } else {
            self.row_group_data_to_df_impl(row_group_data, nothing_to_evaluate)
                .await
        }
    }

    /// Whether the predicate keeps every row for now: it has no static part
    /// and none of its dynamic conjuncts filters rows.
    fn nothing_to_evaluate(&self) -> bool {
        self.use_prefiltered
            && self.rest_predicate.is_none()
            && self
                .predicate_columns
                .iter()
                .all(|c| c.predicate.is_none() && c.dynamic.iter().all(|d| !d.is_active()))
    }

    async fn row_group_data_to_df_impl(
        &self,
        row_group_data: RowGroupData,
        nothing_to_evaluate: bool,
    ) -> PolarsResult<DataFrame> {
        let row_group_data = Arc::new(row_group_data);

        let out_width = self.row_index.is_some() as usize + self.projected_arrow_fields.len();

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

        drop(row_group_data);

        let projection_height = slice_range.len();

        out_columns.extend(decoded_cols);

        let df = unsafe { DataFrame::new_unchecked(projection_height, out_columns) };

        let df = if let Some(predicate) = self
            .predicate
            .as_ref()
            .filter(|p| p.filters_rows && !nothing_to_evaluate)
        {
            let mask = predicate.predicate.evaluate_io(&df)?;
            let mask = mask.bool().unwrap();

            let filtered =
                filter_cols(df.into_columns(), mask, self.target_values_per_thread).await?;

            let height = if let Some(fst) = filtered.first() {
                fst.len()
            } else {
                mask.num_trues()
            };

            unsafe { DataFrame::new_unchecked(height, filtered) }
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
        self.row_index
            .as_ref()
            .map(|row_index| row_index_column(row_index, row_group_data, slice_range))
            .transpose()
    }

    /// Potentially parallelizes based on number of rows & columns. Decoded columns are appended to
    /// `out_vec`.
    async fn decode_projected_columns(
        &self,
        out_vec: &mut Vec<Column>,
        row_group_data: &Arc<RowGroupData>,
        filter: Option<polars_parquet::read::Filter>,
    ) -> PolarsResult<()> {
        let projected_arrow_fields = &self.projected_arrow_fields;
        let expected_num_rows = filter
            .as_ref()
            .map_or(row_group_data.row_group_metadata.num_rows(), |x| {
                x.num_rows(row_group_data.row_group_metadata.num_rows())
            });

        // Ensure we provide the same output column order as the pre-filtered decode.
        let get_projected_field_at_output_index = {
            let predicate_field_indices = self.predicate_field_indices.clone();
            let non_predicate_field_indices = self.non_predicate_field_indices.clone();

            move |i: usize| {
                if predicate_field_indices.is_empty() {
                    i
                } else if i < predicate_field_indices.len() {
                    predicate_field_indices[i]
                } else {
                    non_predicate_field_indices[i - predicate_field_indices.len()]
                }
            }
        };

        let cols_per_thread = calc_cols_per_thread(
            row_group_data.row_group_metadata.num_rows(),
            self.target_values_per_thread,
        );

        let projected_arrow_fields = projected_arrow_fields.clone();
        let row_group_data_2 = row_group_data.clone();

        let task_handles = {
            let projected_arrow_fields = projected_arrow_fields.clone();
            let filter = filter.clone();

            parallelize_first_to_local(
                TaskPriority::Low,
                (0..projected_arrow_fields.len())
                    .step_by(cols_per_thread)
                    .map(move |offset| {
                        let row_group_data = row_group_data_2.clone();
                        let projected_arrow_fields = projected_arrow_fields.clone();
                        let filter = filter.clone();
                        let get_projected_field_at_output_index =
                            get_projected_field_at_output_index.clone();

                        async move {
                            // This is exact as we have already taken out the remainder.
                            (offset
                                ..offset
                                    .saturating_add(cols_per_thread)
                                    .min(projected_arrow_fields.len()))
                                .map(|i| {
                                    let projection = &projected_arrow_fields
                                        [get_projected_field_at_output_index(i)];

                                    let (col, pred_true_mask) = decode_column(
                                        projection.arrow_field(),
                                        &row_group_data,
                                        filter.clone(),
                                        expected_num_rows,
                                    )?;

                                    let col = projection.apply_transform(col)?;

                                    Ok((col, pred_true_mask))
                                })
                                .collect::<PolarsResult<UnitVec<_>>>()
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

fn row_index_column(
    row_index: &RowIndex,
    row_group_data: &RowGroupData,
    slice_range: core::ops::Range<usize>,
) -> PolarsResult<Column> {
    let offset = row_index.offset.saturating_add(
        IdxSize::try_from(row_group_data.row_offset + slice_range.start).unwrap_or(IdxSize::MAX),
    );
    // The DataFrame can be empty at this point if no columns were projected from the file,
    // so we create the row index column manually instead of using `df.with_row_index` to
    // ensure it has the correct number of rows.
    Column::new_row_index(row_index.name.clone(), offset, slice_range.len())
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

    let (arrays, pred_true_mask) = polars_io::prelude::_internal::to_deserializer(
        columns_to_deserialize,
        arrow_field.clone(),
        filter,
    )?;

    if !skip_num_rows_check {
        let num_rows = arrays.iter().map(|array| array.len()).sum::<usize>();
        assert_eq!(num_rows, expected_num_rows);
    }

    let mut series = Series::try_from((arrow_field, arrays))?;
    canonicalize_parquet_maps(&mut series)?;

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

/// Filters columns, in parallel depending number of rows / columns.
async fn filter_cols(
    cols: Vec<Column>,
    mask: &BooleanChunked,
    target_values_per_thread: usize,
) -> PolarsResult<Vec<Column>> {
    if cols.is_empty() {
        return Ok(cols);
    }

    let cols_per_thread = calc_cols_per_thread(cols[0].len(), target_values_per_thread);
    let mut out_vec = Vec::with_capacity(cols.len());
    let cols = Arc::new(cols);
    let mask = mask.clone();

    let task_handles = {
        let cols = &cols;
        let mask = &mask;

        parallelize_first_to_local(
            TaskPriority::Low,
            (0..cols.len()).step_by(cols_per_thread).map(move |offset| {
                let cols = cols.clone();
                let mask = mask.clone();
                async move {
                    (offset..offset.saturating_add(cols_per_thread).min(cols.len()))
                        .map(|i| cols[i].filter(&mask))
                        .collect::<PolarsResult<UnitVec<_>>>()
                }
            }),
        )
    };

    for fut in task_handles {
        out_vec.extend(fut.await?)
    }

    Ok(out_vec)
}

fn calc_cols_per_thread(n_rows_per_col: usize, target_n_rows_per_thread: usize) -> usize {
    if n_rows_per_col == 0 {
        return usize::MAX;
    }

    let n = target_n_rows_per_thread / n_rows_per_col;
    let floor_distance = target_n_rows_per_thread % n_rows_per_col;
    let ceil_distance = n_rows_per_col - floor_distance;

    if floor_distance <= ceil_distance {
        n.max(1)
    } else {
        n + 1
    }
}

// Pre-filtered

/// What a pass over a row group decodes.
#[derive(Clone, Copy)]
enum Item {
    /// Index into `RowGroupDecoder::predicate_columns`.
    PredicateColumn(usize),
    Source(Source),
}

/// The rows a predicate column keeps.
enum Kept {
    /// The column holds every row of the pass.
    Masked(Bitmap),
    /// The column holds the kept rows only.
    Filtered(Bitmap),
}

struct Decoded {
    source: Source,
    column: Column,
    /// For a predicate column: its index into `RowGroupDecoder::predicate_columns` and
    /// the rows it keeps.
    predicate: Option<(usize, Kept)>,
}

/// One pass over a row group.
struct Pass {
    predicate_columns: Arc<[PredicateColumn]>,
    projected_arrow_fields: Arc<[ArrowFieldProjection]>,
    row_index: Option<RowIndex>,
    row_group_data: Arc<RowGroupData>,
    /// The rows the passes before kept. `None` while every row is.
    mask: Option<Bitmap>,
    /// The rows of `mask`.
    num_rows: usize,
    use_decode_filters: bool,
}

impl Pass {
    fn decode(&self, item: Item) -> PolarsResult<Decoded> {
        let (source, predicate_column) = match item {
            Item::Source(source) => (source, None),
            Item::PredicateColumn(i) => {
                let c = &self.predicate_columns[i];
                (c.source, Some((i, c)))
            },
        };
        // The rows of `column` that its active dynamic conjuncts keep, `None` when
        // there is none.
        let dynamic_mask = |c: &PredicateColumn, column: &Column| -> PolarsResult<Option<Bitmap>> {
            let mut mask: Option<Bitmap> = None;
            let df = column.clone().into_frame();
            for d in c.dynamic.iter().filter(|d| d.is_active()) {
                let m = evaluate_mask(d.predicate.as_ref(), &df)?;
                d.measure(m.len(), m.set_bits());
                mask = Some(and_masks(mask, m));
            }
            Ok(mask)
        };
        let evaluate = |column: &Column| -> PolarsResult<Option<(usize, Kept)>> {
            let Some((i, c)) = predicate_column else {
                return Ok(None);
            };
            let mut mask = match &c.predicate {
                Some(p) => {
                    let df =
                        unsafe { DataFrame::new_unchecked(self.num_rows, vec![column.clone()]) };
                    Some(evaluate_mask(p.as_ref(), &df)?)
                },
                None => None,
            };
            if let Some(m) = dynamic_mask(c, column)? {
                mask = Some(and_masks(mask, m));
            }
            Ok(mask.map(|m| (i, Kept::Masked(m))))
        };

        let field_idx = match source {
            Source::Field(field_idx) => field_idx,
            Source::RowIndex => {
                let height = self.row_group_data.row_group_metadata.num_rows();
                let mut column = row_index_column(
                    self.row_index.as_ref().unwrap(),
                    &self.row_group_data,
                    0..height,
                )?;
                if let Some(mask) = &self.mask {
                    column = column.filter(&BooleanChunked::from_bitmap(
                        PlSmallStr::EMPTY,
                        mask.clone(),
                    ))?;
                }
                let predicate = evaluate(&column)?;
                return Ok(Decoded {
                    source,
                    column,
                    predicate,
                });
            },
        };
        let projection = &self.projected_arrow_fields[field_idx];
        let arrow_field = projection.arrow_field();

        if let Some((i, c)) = predicate_column
            && self.mask.is_none()
            && self.use_decode_filters
            && let Some(filter) = &c.decode_filter
        {
            let (column, mask) = decode_column(
                arrow_field,
                &self.row_group_data,
                Some(Filter::Predicate(filter.clone())),
                self.num_rows,
            )?;
            let column = match &c.constant {
                Some(v) => Column::new_scalar(column.name().clone(), v.clone(), mask.set_bits()),
                None => column,
            };
            let mut column = projection.apply_transform(column)?;
            let mut mask = mask;
            if let Some(m) = dynamic_mask(c, &column)?
                && m.unset_bits() > 0
            {
                column =
                    column.filter(&BooleanChunked::from_bitmap(PlSmallStr::EMPTY, m.clone()))?;
                mask = compose_masks(&mask, &m);
            }
            return Ok(Decoded {
                source,
                column,
                predicate: Some((i, Kept::Filtered(mask))),
            });
        }

        let column = match &self.mask {
            Some(mask) => {
                decode_column_prefiltered(arrow_field, &self.row_group_data, mask, self.num_rows)?
            },
            None => decode_column(arrow_field, &self.row_group_data, None, self.num_rows)?.0,
        };
        let column = projection.apply_transform(column)?;
        let predicate = evaluate(&column)?;
        Ok(Decoded {
            source,
            column,
            predicate,
        })
    }
}

impl RowGroupDecoder {
    async fn row_group_data_to_df_prefiltered(
        &self,
        row_group_data: RowGroupData,
    ) -> PolarsResult<DataFrame> {
        debug_assert!(row_group_data.slice.is_none()); // Invariant of the optimizer.
        assert!(self.predicate_field_indices.len() <= self.projected_arrow_fields.len());

        let row_group_data = Arc::new(row_group_data);
        let projection_height = row_group_data.row_group_metadata.num_rows();
        let mut passes = self.passes.lock().unwrap().clone();
        // A conjunct that just started filtering rows is measured on every row
        // of this row group.
        let newly_active: Vec<usize> = (0..self.predicate_columns.len())
            .filter(|&c| {
                self.predicate_columns[c]
                    .dynamic
                    .iter()
                    .any(|d| d.newly_active())
            })
            .collect();
        if !newly_active.is_empty() {
            passes = Arc::new(promote(&passes, &newly_active));
            *self.passes.lock().unwrap() = passes.clone();
        }

        let use_decode_filters = !row_group_data
            .row_group_metadata
            .parquet_columns()
            .iter()
            .any(|c| {
                matches!(
                    c.descriptor().descriptor.primitive_type.logical_type,
                    Some(PrimitiveLogicalType::Float16)
                )
            });

        // The row index and predicate columns decoded so far, holding the kept rows only.
        let mut live_columns: Vec<(Source, Column)> = Vec::with_capacity(
            self.row_index.is_some() as usize + self.predicate_field_indices.len(),
        );
        // The rows kept so far. `None` while every row is.
        let mut mask: Option<Bitmap> = None;
        let mut kept = projection_height;
        let mut selectivity = vec![Selectivity::default(); self.predicate_columns.len()];
        let mut pass_selectivity = Vec::with_capacity(passes.len());

        for (i, pass) in passes.iter().enumerate() {
            let is_last = i + 1 == passes.len();
            let mut items = Vec::with_capacity(pass.len() + 1 + self.rest_field_indices.len());
            if i == 0
                && self.row_index.is_some()
                && self
                    .predicate_columns
                    .iter()
                    .all(|c| c.source != Source::RowIndex)
            {
                items.push(Item::Source(Source::RowIndex));
            }
            items.extend(pass.iter().map(|&c| Item::PredicateColumn(c)));
            if is_last {
                items.extend(
                    self.rest_field_indices
                        .iter()
                        .map(|&f| Item::Source(Source::Field(f))),
                );
            }
            let decoded = self
                .decode_items(
                    items,
                    &row_group_data,
                    mask.as_ref(),
                    kept,
                    use_decode_filters,
                )
                .await?;

            let mut pass_mask: Option<Bitmap> = None;
            let mut filtered = Vec::new();
            for d in decoded {
                let Some((c, kept_rows)) = d.predicate else {
                    live_columns.push((d.source, d.column));
                    continue;
                };
                let m = match &kept_rows {
                    Kept::Masked(m) | Kept::Filtered(m) => m,
                };
                selectivity[c] = Selectivity {
                    input_rows: kept,
                    kept_rows: m.set_bits(),
                };
                pass_mask = Some(and_masks(pass_mask, m.clone()));
                match kept_rows {
                    Kept::Masked(_) => live_columns.push((d.source, d.column)),
                    Kept::Filtered(own) => filtered.push((d.source, d.column, own)),
                }
            }

            let kept_before = kept;
            match pass_mask.filter(|m| m.unset_bits() > 0) {
                None => live_columns.extend(filtered.into_iter().map(|(s, c, _)| (s, c))),
                Some(pass_mask) => {
                    let pass_mask_ck =
                        BooleanChunked::from_bitmap(PlSmallStr::EMPTY, pass_mask.clone());
                    live_columns = self.filter_kept(live_columns, &pass_mask_ck).await?;
                    // A column the decoder filtered by its own mask keeps the rows of the
                    // pass mask among those.
                    for (source, column, own) in filtered {
                        let column = if own.set_bits() == pass_mask.set_bits() {
                            column
                        } else {
                            let own = BooleanChunked::from_bitmap(PlSmallStr::EMPTY, own);
                            column.filter(&pass_mask_ck.filter(&own)?)?
                        };
                        live_columns.push((source, column));
                    }
                    kept = pass_mask.set_bits();
                    mask = Some(match mask {
                        None => pass_mask,
                        Some(mask) => compose_masks(&mask, &pass_mask),
                    });
                },
            }
            pass_selectivity.push(Selectivity {
                input_rows: kept_before,
                kept_rows: kept,
            });
        }

        if let Some(rest) = &self.rest_predicate {
            let columns = live_columns.iter().map(|(_, c)| c.clone()).collect();
            let df = unsafe { DataFrame::new_unchecked(kept, columns) };
            let rest_mask = evaluate_mask(rest.as_ref(), &df)?;
            if rest_mask.unset_bits() > 0 {
                let rest_mask_ck =
                    BooleanChunked::from_bitmap(PlSmallStr::EMPTY, rest_mask.clone());
                live_columns = self.filter_kept(live_columns, &rest_mask_ck).await?;
                kept = rest_mask.set_bits();
                mask = Some(match mask {
                    None => rest_mask,
                    Some(mask) => compose_masks(&mask, &rest_mask),
                });
            }
        }

        let dead_columns = self
            .decode_items(
                self.non_predicate_field_indices
                    .iter()
                    .map(|&f| Item::Source(Source::Field(f)))
                    .collect(),
                &row_group_data,
                mask.as_ref(),
                kept,
                false,
            )
            .await?;

        drop(row_group_data);

        let next_passes = plan_passes(
            &passes,
            projection_height,
            &selectivity,
            &pass_selectivity,
            !self.rest_field_indices.is_empty(),
        );
        if next_passes != *passes {
            if polars_core::config::verbose() {
                let names: Vec<Vec<&str>> = next_passes
                    .iter()
                    .map(|pass| {
                        pass.iter()
                            .map(|&c| match self.predicate_columns[c].source {
                                Source::RowIndex => self.row_index.as_ref().unwrap().name.as_str(),
                                Source::Field(f) => {
                                    self.projected_arrow_fields[f].output_name().as_str()
                                },
                            })
                            .collect()
                    })
                    .collect();
                eprintln!("[ParquetFileReader]: Predicate passes: {names:?}");
            }
            *self.passes.lock().unwrap() = Arc::new(next_passes);
        }

        // The row index, `predicate_field_indices`, then the other columns.
        live_columns.sort_unstable_by_key(|(source, _)| *source);
        let columns = live_columns
            .into_iter()
            .map(|(_, c)| c)
            .chain(dead_columns.into_iter().map(|d| d.column))
            .collect();
        let df = unsafe { DataFrame::new_unchecked(kept, columns) };
        Ok(df)
    }

    /// Decodes `items`, keeping only the rows set in the mask when one is given.
    async fn decode_items(
        &self,
        items: Vec<Item>,
        row_group_data: &Arc<RowGroupData>,
        mask: Option<&Bitmap>,
        num_rows: usize,
        use_decode_filters: bool,
    ) -> PolarsResult<Vec<Decoded>> {
        let n_items = items.len();
        let cols_per_thread = n_items.div_ceil(self.num_pipelines).max(1);
        let items: Arc<[Item]> = items.into();
        let pass = Arc::new(Pass {
            predicate_columns: self.predicate_columns.clone(),
            projected_arrow_fields: self.projected_arrow_fields.clone(),
            row_index: self.row_index.clone(),
            row_group_data: row_group_data.clone(),
            mask: mask.cloned(),
            num_rows,
            use_decode_filters,
        });

        let task_handles = {
            parallelize_first_to_local(
                TaskPriority::Low,
                (0..n_items).step_by(cols_per_thread).map(move |offset| {
                    let items = items.clone();
                    let pass = pass.clone();

                    async move {
                        (offset..offset.saturating_add(cols_per_thread).min(n_items))
                            .map(|i| pass.decode(items[i]))
                            .collect::<PolarsResult<UnitVec<_>>>()
                    }
                }),
            )
        };

        let mut out = Vec::with_capacity(n_items);
        for fut in task_handles {
            out.extend(fut.await?);
        }
        Ok(out)
    }

    /// Keeps the rows of `mask` in the columns decoded so far.
    async fn filter_kept(
        &self,
        live_columns: Vec<(Source, Column)>,
        mask: &BooleanChunked,
    ) -> PolarsResult<Vec<(Source, Column)>> {
        let (sources, columns): (Vec<Source>, Vec<Column>) = live_columns.into_iter().unzip();
        let columns = filter_cols(columns, mask, self.target_values_per_thread).await?;
        Ok(sources.into_iter().zip(columns).collect())
    }
}

/// Evaluates a row predicate and returns its set rows as a bitmap, treating null as
/// unset.
fn evaluate_mask(predicate: &dyn PhysicalIoExpr, df: &DataFrame) -> PolarsResult<Bitmap> {
    let mut mask = predicate.evaluate_io(df)?.bool().unwrap().clone();
    mask.rechunk_mut();
    let arr = mask.downcast_as_array();
    Ok(match arr.validity() {
        None => arr.values().clone(),
        Some(validity) => arr.values() & validity,
    })
}

/// `mask` narrowed by `m`, both over the same rows.
fn and_masks(mask: Option<Bitmap>, m: Bitmap) -> Bitmap {
    match mask {
        None => m,
        Some(mask) => &mask & &m,
    }
}

/// Narrows `outer` by `inner`, which holds one bit per set bit of `outer`.
fn compose_masks(outer: &Bitmap, inner: &Bitmap) -> Bitmap {
    assert_eq!(inner.len(), outer.set_bits());
    if inner.unset_bits() == 0 {
        return outer.clone();
    }
    let mut out = MutableBitmap::from_len_zeroed(outer.len());
    for (idx, keep) in outer.true_idx_iter().zip(inner.iter()) {
        if keep {
            // SAFETY: `idx` indexes `outer`, which has the same length as `out`.
            unsafe { out.set_unchecked(idx, true) };
        }
    }
    out.freeze()
}

fn decode_column_prefiltered(
    arrow_field: &ArrowField,
    row_group_data: &RowGroupData,
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

    let mut series = if !prefilter {
        let mask = BooleanChunked::from_bitmap(PlSmallStr::EMPTY, mask_bitmap.clone());
        series.filter(&mask)?
    } else {
        series
    };

    // Done after the filter so that discarded rows cost nothing.
    canonicalize_parquet_maps(&mut series)?;

    assert_eq!(series.len(), expected_num_rows);

    Ok(series.into_column())
}

mod tests {
    #[test]
    fn test_calc_cols_per_thread() {
        use super::calc_cols_per_thread;

        assert_eq!(
            [
                calc_cols_per_thread(0, 5),
                calc_cols_per_thread(1, 5),
                calc_cols_per_thread(2, 5),
                calc_cols_per_thread(3, 5),
                calc_cols_per_thread(4, 5),
                calc_cols_per_thread(5, 5),
            ],
            [usize::MAX, 5, 2, 2, 1, 1]
        );

        assert_eq!(
            [
                calc_cols_per_thread(11_184_810, 16_777_216),
                calc_cols_per_thread(11_184_811, 16_777_216),
            ],
            [2, 1]
        );

        assert_eq!(
            [
                calc_cols_per_thread(0, 0),
                calc_cols_per_thread(0, 99),
                calc_cols_per_thread(99, 0),
                calc_cols_per_thread(99, 99),
            ],
            [usize::MAX, usize::MAX, 1, 1],
        )
    }
}
