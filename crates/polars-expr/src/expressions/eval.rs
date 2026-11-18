use std::borrow::Cow;
use std::cell::LazyCell;
use std::collections::VecDeque;
use std::sync::Arc;

use polars_arrow::bitmap::{Bitmap, BitmapBuilder};
use polars_core::chunked_array::builder::AnonymousOwnedListBuilder;
use polars_core::error::{PolarsResult, feature_gated, polars_bail, polars_ensure, polars_err};
use polars_core::frame::DataFrame;
#[cfg(feature = "dtype-array")]
use polars_core::prelude::ArrayChunked;
use polars_core::prelude::{
    _set_check_length, BooleanChunked, ChunkCast, ChunkExplode, ChunkFilter, ChunkNestingUtils,
    Column, Field, GroupPositions, GroupsType, IdxCa, IntoColumn, ListBuilderTrait, ListChunked,
};
use polars_core::schema::{Schema, SchemaExt};
use polars_core::series::Series;
use polars_plan::dsl::{EvalVariant, Expr};
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;

use super::{AggState, AggregationContext, PhysicalExpr, UpdateGroups};
use crate::state::{EvalColumns, ExecutionState};

fn check_column_lengths(columns: &[Column], len: usize) -> PolarsResult<()> {
    if let Some(c) = columns.iter().find(|c| c.len() != len) {
        polars_bail!(
            ShapeMismatch: "column '{}' of length {} read in `eval` does not match the input of length {}",
            c.name(), c.len(), len
        );
    }
    Ok(())
}

/// The frame of an evaluation: `columns` with each row repeated `lengths[row]` times.
fn repeat_rows(
    columns: &[Column],
    lengths: impl Iterator<Item = usize>,
    height: usize,
) -> PolarsResult<DataFrame> {
    let idx = LazyCell::new(|| {
        let mut idx = Vec::with_capacity(height);
        for (row, length) in lengths.enumerate() {
            idx.extend(std::iter::repeat_n(row as IdxSize, length));
        }
        idx
    });
    let columns = columns
        .iter()
        .map(|c| match c {
            Column::Scalar(_) => c.new_from_index(0, height),
            // SAFETY: every row index is in bounds.
            _ => unsafe { c.take_slice_unchecked(&idx) },
        })
        .collect();
    DataFrame::new(height, columns)
}

/// The columns read in an evaluation with a row per group of `groups`, which skip the rows that
/// are null in `validity`.
fn eval_columns(
    columns: &[Column],
    validity: Option<&Bitmap>,
    groups: &GroupPositions,
) -> PolarsResult<Option<EvalColumns>> {
    let Some(first) = columns.first() else {
        return Ok(None);
    };
    let (height, columns) = match validity {
        Some(validity) if validity.unset_bits() > 0 => {
            (validity.set_bits(), filter_columns(columns, validity)?.1)
        },
        _ => (first.len(), columns.to_vec()),
    };
    Ok(Some(EvalColumns {
        columns: DataFrame::new(height, columns)?,
        groups: groups as *const GroupPositions as usize,
    }))
}

/// A column read in the evaluation of an `eval`: a single value per evaluated list.
pub struct EvalColumnExpr {
    name: PlSmallStr,
    expr: Expr,
}

impl EvalColumnExpr {
    pub fn new(name: PlSmallStr, expr: Expr) -> Self {
        Self { name, expr }
    }
}

impl PhysicalExpr for EvalColumnExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    /// The value of the list of each element, from the frame of the elementwise evaluation.
    fn evaluate_impl(&self, df: &DataFrame, _state: &ExecutionState) -> PolarsResult<Column> {
        df.column(&self.name).cloned().map_err(|_| {
            polars_err!(
                InvalidOperation: "column '{}' can not be read per element in this `eval`",
                self.name
            )
        })
    }

    /// The value of each group: that of the list for the groups of the evaluation, and that of its
    /// first element for other groups, such as those of a window.
    fn evaluate_on_groups_impl<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        if let Some(eval_columns) = state.eval_columns.as_ref()
            && eval_columns.groups == groups as *const GroupPositions as usize
        {
            let column = eval_columns.columns.column(&self.name)?.clone();
            return Ok(AggregationContext::from_agg_state(
                AggState::AggregatedScalar(column),
                Cow::Borrowed(groups),
            ));
        }
        // SAFETY: the groups are in bounds of the frame.
        let column = unsafe { self.evaluate(df, state)?.agg_first(groups) };
        Ok(AggregationContext::from_agg_state(
            AggState::AggregatedScalar(column),
            Cow::Borrowed(groups),
        ))
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        input_schema
            .get_field(&self.name)
            .ok_or_else(|| polars_err!(ColumnNotFound: "{}", self.name))
    }

    fn is_scalar(&self) -> bool {
        true
    }
}

/// The rows of `columns` that are set in `validity`.
fn filter_columns(
    columns: &[Column],
    validity: &Bitmap,
) -> PolarsResult<(BooleanChunked, Vec<Column>)> {
    let mask = BooleanChunked::from_bitmap(PlSmallStr::EMPTY, validity.clone());
    let columns = columns
        .iter()
        .map(|c| c.filter(&mask))
        .collect::<PolarsResult<Vec<_>>>()?;
    Ok((mask, columns))
}

#[derive(Clone)]
pub struct EvalExpr {
    input: Arc<dyn PhysicalExpr>,
    /// The columns of the frame that `evaluation` reads.
    columns: Vec<Arc<dyn PhysicalExpr>>,
    evaluation: Arc<dyn PhysicalExpr>,
    variant: EvalVariant,
    expr: Expr,
    output_field: Field,
    is_scalar: bool,
    evaluation_is_scalar: bool,
    evaluation_is_elementwise: bool,
    evaluation_is_fallible: bool,
    inputs_are_elementwise: bool,
}

impl EvalExpr {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        input: Arc<dyn PhysicalExpr>,
        columns: Vec<Arc<dyn PhysicalExpr>>,
        evaluation: Arc<dyn PhysicalExpr>,
        variant: EvalVariant,
        expr: Expr,
        output_field: Field,
        is_scalar: bool,
        evaluation_is_scalar: bool,
        evaluation_is_elementwise: bool,
        evaluation_is_fallible: bool,
        inputs_are_elementwise: bool,
    ) -> Self {
        Self {
            input,
            columns,
            evaluation,
            variant,
            expr,
            output_field,
            is_scalar,
            evaluation_is_scalar,
            evaluation_is_elementwise,
            evaluation_is_fallible,
            inputs_are_elementwise,
        }
    }

    /// Whether `evaluation` runs once on all elements. A scalar evaluation that reads columns
    /// instead gives a single value per list.
    fn use_elementwise_path(&self) -> bool {
        self.evaluation_is_elementwise && !(self.evaluation_is_scalar && !self.columns.is_empty())
    }

    /// `columns` are those `evaluation` reads, a value per list.
    fn evaluate_on_list_chunked(
        &self,
        ca: &ListChunked,
        columns: &[Column],
        state: &ExecutionState,
        is_agg: bool,
    ) -> PolarsResult<Column> {
        check_column_lengths(columns, ca.len())?;
        // Fast path: Empty or only nulls.
        if ca.null_count() == ca.len() {
            let name = self.output_field.name.clone();
            return Ok(Column::full_null(name, ca.len(), self.output_field.dtype()));
        }
        let ca = ca
            .trim_lists_to_normalized_offsets()
            .map_or(Cow::Borrowed(ca), Cow::Owned);

        // A fallible evaluation must not run on the elements of null rows.
        if self.evaluation_is_fallible
            && ca.has_masked_out_values()
            && let Some(validity) = ca.rechunk_validity()
        {
            let (mask, columns) = filter_columns(columns, &validity)?;
            let out = self.evaluate_on_list_chunked(&ca.filter(&mask)?, &columns, state, is_agg)?;
            return Ok(out.deposit(&validity));
        }

        // SAFETY:
        // We may temporarily create lengths that exceed IDXSIZE
        // If that happens we slice and process in batches.
        unsafe { _set_check_length(false) };
        let flattened = ca.get_inner().into_column();
        unsafe { _set_check_length(true) };
        let flattened_len = flattened.len();
        let validity = ca.rechunk_validity();

        // Batch when the total number of inner elements exceeds IdxSize::MAX to avoid
        // truncating offset/length casts below. Each batch covers a contiguous row-range
        // whose accumulated inner element count stays within IdxSize::MAX.
        const LIMIT: usize = (IdxSize::MAX - 1) as usize;
        if flattened_len > LIMIT {
            let offsets = ca.offsets()?;
            // offsets_slice[i] / offsets_slice[i+1] are the start/end of row i.
            let offsets_slice = offsets.as_slice();
            let mut batch_results: VecDeque<Column> = VecDeque::new();
            let mut batch_row_start = 0usize;
            let mut batch_inner_start: i64 = 0;

            loop {
                if batch_row_start >= ca.len() {
                    break;
                }
                let threshold = batch_inner_start + LIMIT as i64;
                // Binary search for the first row whose end offset exceeds the threshold.
                // offsets_slice[batch_row_start+1..] holds end offsets for rows
                // batch_row_start, batch_row_start+1, …; partition_point returns how many fit.
                let rel = offsets_slice[batch_row_start + 1..].partition_point(|&v| v <= threshold);
                if batch_row_start + rel >= ca.len() {
                    // All remaining rows fit in one batch.
                    break;
                }
                let flush_len = rel;
                polars_ensure!(flush_len > 0, ComputeError: "list elements larger than IdxSize::MAX are not supported");
                let batch = ca.slice(batch_row_start as i64, flush_len);
                let batch_columns = slice_columns(columns, batch_row_start, flush_len);
                batch_results.push_back(self.evaluate_on_list_chunked(
                    &batch,
                    &batch_columns,
                    state,
                    is_agg,
                )?);
                batch_row_start += flush_len;
                batch_inner_start = offsets_slice[batch_row_start];
            }
            // Flush the final batch.
            let batch_len = ca.len() - batch_row_start;
            let batch = ca.slice(batch_row_start as i64, batch_len);
            let batch_columns = slice_columns(columns, batch_row_start, batch_len);
            batch_results.push_back(self.evaluate_on_list_chunked(
                &batch,
                &batch_columns,
                state,
                is_agg,
            )?);

            let mut out = batch_results.pop_front().unwrap();
            for other in batch_results.into_iter() {
                out.append_owned(other)?;
            }
            return Ok(out);
        }

        let df = if columns.is_empty() {
            DataFrame::empty_with_height(ca.len())
        } else {
            let offsets = ca.offsets()?;
            let lengths = offsets.lengths();
            repeat_rows(columns, lengths, flattened_len)?
        };

        // Fast path: fully elementwise expression.
        if self.use_elementwise_path() {
            let mut state = state.clone();
            state.element = Arc::new(Some((flattened, validity.clone())));
            let mut column = self.evaluation.evaluate(&df, &state)?;

            // Since `lit` is marked as elementwise, this may lead to problems.
            column.broadcast_in_place_to(flattened_len)?;

            if !is_agg || !self.evaluation_is_scalar {
                column = ca
                    .with_inner_values(column.as_materialized_series())
                    .into_column();
            }

            return Ok(column);
        }

        let offsets = ca.offsets()?;
        // Detect accidental inclusion of sliced-out elements from chunks after the 1st (if present).
        assert_eq!(i64::try_from(flattened_len).unwrap(), *offsets.last());

        // Create groups for all valid array elements.
        let groups = if ca.has_nulls() {
            let validity = validity.as_ref().unwrap();
            offsets
                .offset_and_length_iter()
                .zip(validity.iter())
                .filter_map(|((offset, length), validity)| {
                    validity.then_some([offset as IdxSize, length as IdxSize])
                })
                .collect()
        } else {
            offsets
                .offset_and_length_iter()
                .map(|(offset, length)| [offset as IdxSize, length as IdxSize])
                .collect()
        };
        let groups = GroupsType::new_slice(groups, false, true);
        let groups = Cow::Owned(groups.into_sliceable());

        let mut state = state.clone();
        state.eval_columns = Arc::new(eval_columns(columns, validity.as_ref(), &groups)?);
        state.element = Arc::new(Some((flattened, validity.clone())));

        let mut ac = self.evaluation.evaluate_on_groups(&df, &groups, &state)?;

        // Update the groups.
        if self.evaluation_is_scalar || is_agg {
            ac.set_groups_for_undefined_agg_states();
        } else {
            ac.groups();
        }

        let flat_naive = ac.flat_naive();

        // Fast path. Groups are pointing to the same offsets in the data buffer.
        if flat_naive.len() == flattened_len
            && let Some(output_groups) = ac.groups.as_ref().as_unrolled_slice()
            && !(is_agg && self.evaluation_is_scalar)
        {
            let groups_are_unchanged = if let Some(validity) = &validity {
                assert_eq!(validity.set_bits(), output_groups.len());
                validity
                    .true_idx_iter()
                    .zip(output_groups)
                    .all(|(j, [start, len])| {
                        let (original_start, original_end) =
                            unsafe { offsets.start_end_unchecked(j) };
                        (*start == original_start as IdxSize)
                            & (*len == (original_end - original_start) as IdxSize)
                    })
            } else {
                output_groups
                    .iter()
                    .zip(offsets.offset_and_length_iter())
                    .all(|([start, len], (original_start, original_groups))| {
                        (*start == original_start as IdxSize) & (*len == original_groups as IdxSize)
                    })
            };

            if groups_are_unchanged {
                let values = flat_naive.as_materialized_series();
                return Ok(ca.with_inner_values(values).into_column());
            }
        }

        // Slow path. Groups have changed, so we need to gather data again.
        if is_agg && self.evaluation_is_scalar {
            let mut values = ac.finalize();

            // We didn't have any groups for the `null` values so we have to reinsert them.
            if let Some(validity) = validity {
                values = values.deposit(&validity);
            }

            Ok(values)
        } else {
            let mut ca = ac.aggregated_as_list();

            // We didn't have any groups for the `null` values so we have to reinsert them.
            if let Some(validity) = validity {
                ca = Cow::Owned(ca.deposit(&validity));
            }

            Ok(ca.into_owned().into_column())
        }
    }

    #[cfg(feature = "dtype-array")]
    fn evaluate_on_array_chunked(
        &self,
        ca: &ArrayChunked,
        columns: &[Column],
        state: &ExecutionState,
        as_list: bool,
        is_agg: bool,
    ) -> PolarsResult<Column> {
        check_column_lengths(columns, ca.len())?;
        // Fast path: Empty or only nulls.
        if ca.null_count() == ca.len() {
            let name = self.output_field.name.clone();
            return Ok(Column::full_null(name, ca.len(), self.output_field.dtype()));
        }

        let ca = ca
            .trim_lists_to_normalized_offsets()
            .map_or(Cow::Borrowed(ca), Cow::Owned);

        // A fallible evaluation must not run on the elements of null rows.
        if self.evaluation_is_fallible
            && ca.width() > 0
            && ca.has_nulls()
            && let Some(validity) = ca.rechunk_validity()
        {
            let (mask, columns) = filter_columns(columns, &validity)?;
            let out = self.evaluate_on_array_chunked(
                &ca.filter(&mask)?,
                &columns,
                state,
                as_list,
                is_agg,
            )?;
            return Ok(out.deposit(&validity));
        }

        // SAFETY:
        // We may temporarily create lengths that exceed IDXSIZE
        // If that happens we slice and process in batches.
        unsafe { _set_check_length(false) };
        let flattened = ca.get_inner().into_column();
        unsafe { _set_check_length(true) };
        let flattened_len = flattened.len();
        let validity = ca.rechunk_validity();
        let width = ca.width();

        let limit = if cfg!(debug_assertions) {
            std::env::var("POLARS_ARRAY_EVAL_IDX_SIZE_LIMIT")
                .map(|v| v.parse::<usize>().unwrap())
                .unwrap_or(IdxSize::MAX as usize - 1)
        } else {
            (IdxSize::MAX - 1) as usize
        };

        if flattened_len > limit && width > 0 {
            if state.verbose() {
                eprintln!("IdxSize limit hit; chunking branch hit");
            }

            let rows_per_batch = limit / width;
            polars_ensure!(rows_per_batch > 0, ComputeError: "array elements larger than IdxSize::MAX are not supported");
            let mut batch_results: VecDeque<Column> = VecDeque::new();
            let mut batch_row_start = 0usize;

            while batch_row_start < ca.len() {
                let batch_len = (ca.len() - batch_row_start).min(rows_per_batch);
                let batch = ca.slice(batch_row_start as i64, batch_len);
                let batch_columns = slice_columns(columns, batch_row_start, batch_len);
                batch_results.push_back(self.evaluate_on_array_chunked(
                    &batch,
                    &batch_columns,
                    state,
                    as_list,
                    is_agg,
                )?);
                batch_row_start += batch_len;
            }

            let mut out = batch_results.pop_front().unwrap();
            for other in batch_results {
                out.append_owned(other)?;
            }
            return Ok(out);
        }

        let df = if columns.is_empty() {
            DataFrame::empty_with_height(ca.len())
        } else {
            repeat_rows(columns, std::iter::repeat_n(width, ca.len()), flattened_len)?
        };

        // Fast path: fully elementwise expression.
        if self.use_elementwise_path() {
            assert!(!self.evaluation_is_scalar);

            let mut state = state.clone();
            state.element = Arc::new(Some((flattened, None)));

            let mut column = self.evaluation.evaluate(&df, &state)?;
            column.broadcast_in_place_to(flattened_len)?;
            assert_eq!(column.len(), ca.len() * ca.width());

            let dtype = column.dtype().clone();
            let mut out = ArrayChunked::from_aligned_values(
                self.output_field.name.clone(),
                &dtype,
                ca.width(),
                column.take_materialized_series().into_chunks(),
                ca.len(),
            );

            out.set_validity(validity);
            return Ok(if as_list {
                out.to_list().into_column()
            } else {
                out.clone().into_column()
            });
        }

        assert_eq!(flattened_len, ca.width() * ca.len());

        // Create groups for all valid array elements.
        let groups = if ca.has_nulls() {
            let validity = validity.as_ref().unwrap();
            (0..ca.len())
                .filter(|i| unsafe { validity.get_bit_unchecked(*i) })
                .map(|i| [(i * ca.width()) as IdxSize, ca.width() as IdxSize])
                .collect()
        } else {
            (0..ca.len())
                .map(|i| [(i * ca.width()) as IdxSize, ca.width() as IdxSize])
                .collect()
        };
        let groups = GroupsType::new_slice(groups, false, true);
        let groups = Cow::Owned(groups.into_sliceable());

        let mut state = state.clone();
        state.eval_columns = Arc::new(eval_columns(columns, validity.as_ref(), &groups)?);
        state.element = Arc::new(Some((flattened, validity.clone())));

        let mut ac = self.evaluation.evaluate_on_groups(&df, &groups, &state)?;

        ac.groups(); // Update the groups.

        let flat_naive = ac.flat_naive();

        // Fast path. Groups are pointing to the same offsets in the data buffer.
        if flat_naive.len() == ca.len() * ca.width()
            && let Some(output_groups) = ac.groups.as_ref().as_unrolled_slice()
            && !(is_agg && self.evaluation_is_scalar)
        {
            let ca_width = ca.width() as IdxSize;
            let groups_are_unchanged = if let Some(validity) = &validity {
                assert_eq!(validity.set_bits(), output_groups.len());
                validity
                    .true_idx_iter()
                    .zip(output_groups)
                    .all(|(j, [start, len])| {
                        (*start == j as IdxSize * ca_width) & (*len == ca_width)
                    })
            } else {
                use polars_utils::itertools::Itertools;

                output_groups
                    .iter()
                    .enumerate_idx()
                    .all(|(i, [start, len])| (*start == i * ca_width) & (*len == ca_width))
            };

            if groups_are_unchanged {
                let values = flat_naive;
                let dtype = values.dtype().clone();
                let mut out = ArrayChunked::from_aligned_values(
                    self.output_field.name.clone(),
                    &dtype,
                    ca.width(),
                    values.as_materialized_series().chunks().clone(),
                    ca.len(),
                );

                out.set_validity(validity);
                return Ok(if as_list {
                    out.to_list().into_column()
                } else {
                    out.into_column()
                });
            }
        }

        // Slow path. Groups have changed, so we need to gather data again.
        if is_agg && self.evaluation_is_scalar {
            let mut values = ac.finalize();

            // We didn't have any groups for the `null` values so we have to reinsert them.
            if let Some(validity) = validity {
                values = values.deposit(&validity);
            }

            Ok(values)
        } else {
            let mut ca = ac.aggregated_as_list();

            // We didn't have any groups for the `null` values so we have to reinsert them.
            if let Some(validity) = validity {
                ca = Cow::Owned(ca.deposit(&validity));
            }

            Ok(if as_list {
                ca.into_owned().into_column()
            } else {
                ca.cast(self.output_field.dtype()).unwrap().into_column()
            })
        }
    }

    fn evaluate_cumulative_eval(
        &self,
        input: &Series,
        columns: &[Column],
        min_samples: usize,
        state: &ExecutionState,
    ) -> PolarsResult<Column> {
        if input.is_empty() {
            return Ok(Column::new_empty(
                self.output_field.name().clone(),
                self.output_field.dtype(),
            ));
        }

        let flattened = input.clone().into_column();
        let validity = input.rechunk_validity();

        let mut deposit: Option<Bitmap> = None;

        let groups = if min_samples == 0 {
            (0..input.len() as IdxSize).map(|i| [0, i + 1]).collect()
        } else {
            let validity = validity
                .clone()
                .unwrap_or_else(|| Bitmap::new_with_value(true, input.len()));
            let mut count = 0;
            let mut deposit_builder = BitmapBuilder::with_capacity(input.len());
            let out = (0..input.len() as IdxSize)
                .filter(|i| {
                    count += usize::from(unsafe { validity.get_bit_unchecked(*i as usize) });
                    let is_selected = count >= min_samples;
                    unsafe { deposit_builder.push_unchecked(is_selected) };
                    is_selected
                })
                .map(|i| [0, i + 1])
                .collect();
            deposit = Some(deposit_builder.freeze());
            out
        };

        let groups = GroupsType::new_slice(groups, true, true);

        let groups = groups.into_sliceable();

        // Each window reads the columns at its last row.
        check_column_lengths(columns, input.len())?;
        let df = DataFrame::empty_with_height(input.len());

        let mut state = state.clone();
        state.eval_columns = Arc::new(eval_columns(columns, deposit.as_ref(), &groups)?);
        state.element = Arc::new(Some((flattened, validity)));

        let agg = self.evaluation.evaluate_on_groups(&df, &groups, &state)?;
        let (mut out, _) = agg.get_final_aggregation();

        // Since we only evaluated the expressions on the items that satisfied the min samples, we
        // need to fix it up here again.
        if let Some(deposit) = deposit {
            let mut i = 0;
            let gather_idxs = deposit
                .iter()
                .map(|v| {
                    let out = i;
                    i += IdxSize::from(v);
                    out
                })
                .collect::<Vec<IdxSize>>();
            let gather_idxs =
                IdxCa::from_vec_validity(PlSmallStr::EMPTY, gather_idxs, Some(deposit));
            out = unsafe { out.take_unchecked(&gather_idxs) };
        }

        Ok(out)
    }
}

impl PhysicalExpr for EvalExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate_impl(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let mut input = self.input.evaluate(df, state)?;
        let columns = self
            .columns
            .iter()
            .map(|c| c.evaluate(df, state))
            .collect::<PolarsResult<Vec<_>>>()?;
        if let Some(c) = columns.first()
            && input.len() == 1
        {
            input = input.new_from_index(0, c.len());
        }
        // @scalar-opt: every variant but `Cumulative` maps rows independently, so a scalar
        // input could be evaluated once and broadcast instead of expanded to full length.
        let out = match self.variant {
            EvalVariant::List => {
                let lst = input.list()?;
                self.evaluate_on_list_chunked(lst, &columns, state, false)
            },
            EvalVariant::ListAgg => {
                let lst = input.list()?;
                self.evaluate_on_list_chunked(lst, &columns, state, true)
            },
            EvalVariant::Array { as_list } => feature_gated!("dtype-array", {
                let arr = input.array()?;
                self.evaluate_on_array_chunked(arr, &columns, state, as_list, false)
            }),
            EvalVariant::ArrayAgg => feature_gated!("dtype-array", {
                let arr = input.array()?;
                self.evaluate_on_array_chunked(arr, &columns, state, true, true)
            }),
            EvalVariant::Cumulative { min_samples } => self.evaluate_cumulative_eval(
                input.as_materialized_series(),
                &columns,
                min_samples,
                state,
            ),
        }?;
        Ok(out.with_name(self.output_field.name().clone()))
    }

    fn evaluate_on_groups_impl<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut input = self.input.evaluate_on_groups(df, groups, state)?;
        input.groups();

        let unsupported = || {
            polars_err!(
                InvalidOperation: "an `eval` that reads columns is not supported on {} in an aggregation",
                self.input.as_expression().map_or_else(String::new, |e| format!("'{e}'"))
            )
        };

        if let EvalVariant::Cumulative { min_samples } = self.variant {
            if !self.columns.is_empty() {
                return Err(unsupported());
            }
            let mut builder = AnonymousOwnedListBuilder::new(
                self.output_field.name().clone(),
                input.groups().len(),
                Some(self.output_field.dtype.clone()),
            );
            for group in input.iter_groups(false) {
                match group {
                    None => {},
                    Some(group) => {
                        let out =
                            self.evaluate_cumulative_eval(group.as_ref(), &[], min_samples, state)?;
                        builder.append_series(out.as_materialized_series())?;
                    },
                }
            }

            input.with_values(builder.finish().into_column(), true, Some(&self.expr))?;
            input.rename(self.output_field.name().clone());
            return Ok(input);
        }

        // `values` are the rows the evaluation runs on, with a value per row in each of
        // `columns`. If `lists` is set, these are the rows of all groups in order, as in `lists`.
        let mut lists = None;
        let (values, columns) = if self.columns.is_empty() {
            (input.flat_naive().into_owned(), Vec::new())
        } else {
            match input.agg_state() {
                AggState::NotAggregated(_) | AggState::LiteralScalar(_)
                    if self.inputs_are_elementwise
                        && matches!(input.update_groups, UpdateGroups::No) =>
                {
                    let columns = self
                        .columns
                        .iter()
                        .map(|c| c.evaluate(df, state))
                        .collect::<PolarsResult<Vec<_>>>()?;
                    // A literal is the same for every row of `df`.
                    let values = match input.agg_state() {
                        AggState::LiteralScalar(c) => c.new_from_index(0, df.height()),
                        _ => input.flat_naive().into_owned(),
                    };
                    (values, columns)
                },
                AggState::AggregatedScalar(_) => return Err(unsupported()),
                AggState::NotAggregated(_)
                | AggState::AggregatedList(_)
                | AggState::LiteralScalar(_) => {
                    let input_lists = input
                        .aggregated_as_list()
                        .into_owned()
                        .rechunk()
                        .into_owned();
                    let lengths = input_lists.offsets()?;
                    let columns = self
                        .columns
                        .iter()
                        .map(|c| {
                            let mut ac = c.evaluate_on_groups(df, groups, state)?;
                            let c_lists = ac.aggregated_as_list().rechunk().into_owned();
                            if c_lists.offsets()?.lengths().ne(lengths.lengths()) {
                                return Err(unsupported());
                            }
                            Ok(c_lists.get_inner().into_column())
                        })
                        .collect::<PolarsResult<Vec<_>>>()?;
                    let values = input_lists.get_inner().into_column();
                    lists = Some(input_lists);
                    (values, columns)
                },
            }
        };

        let out = match self.variant {
            EvalVariant::List => {
                self.evaluate_on_list_chunked(values.list()?, &columns, state, false)?
            },
            EvalVariant::ListAgg => {
                self.evaluate_on_list_chunked(values.list()?, &columns, state, true)?
            },
            EvalVariant::Array { as_list } => feature_gated!("dtype-array", {
                self.evaluate_on_array_chunked(values.array()?, &columns, state, as_list, false)?
            }),
            EvalVariant::ArrayAgg => feature_gated!("dtype-array", {
                self.evaluate_on_array_chunked(values.array()?, &columns, state, true, true)?
            }),
            EvalVariant::Cumulative { .. } => unreachable!(),
        };
        match lists {
            Some(lists) => {
                let out = lists.with_inner_values(out.as_materialized_series());
                input.with_values_and_args(
                    out.into_column(),
                    true,
                    Some(&self.expr),
                    false,
                    false,
                )?;
            },
            None => {
                input.with_values_and_args(out, false, Some(&self.expr), true, self.is_scalar())?;
            },
        }

        input.rename(self.output_field.name().clone());
        Ok(input)
    }

    fn to_field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        Ok(self.output_field.clone())
    }

    fn is_scalar(&self) -> bool {
        self.is_scalar
    }
}

fn slice_columns(columns: &[Column], offset: usize, length: usize) -> Vec<Column> {
    columns
        .iter()
        .map(|c| c.slice(offset as i64, length))
        .collect()
}
