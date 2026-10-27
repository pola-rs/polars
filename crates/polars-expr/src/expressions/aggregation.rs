use std::borrow::Cow;

use polars_arrow::bitmap::Bitmap;
use polars_compute::nan::first_nan_idx;
use polars_core::frame::group_by::aggregations::{_use_rolling_kernels, rolling_numeric_minmax_by};
use polars_core::prelude::row_encode::_get_rows_encoded_ca;
use polars_core::prelude::*;
use polars_core::runtime::RAYON;
use polars_core::series::IsSorted;
use polars_core::utils::{_split_offsets, first_null};
use polars_core::with_match_physical_float_polars_type;
use polars_ops::prelude::ArgAgg;
#[cfg(feature = "propagate_nans")]
use polars_ops::prelude::nan_propagating_aggregate;
use rayon::prelude::*;

use super::*;
use crate::expressions::AggState::AggregatedScalar;
use crate::expressions::count::evaluate_count_on_ac;
use crate::expressions::{AggState, AggregationContext, PhysicalExpr};
use crate::reduce::GroupedReduction;

#[derive(Debug, Clone, Copy)]
pub struct AggregationType {
    pub(crate) groupby: GroupByMethod,
    pub(crate) allow_threading: bool,
    /// Only used by min and max.
    pub(crate) propagate_nulls: bool,
}

pub(crate) struct AggregationExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) agg_type: AggregationType,
    pub(crate) output_field: Field,
}

impl AggregationExpr {
    /// Sets the groups of `values` that contain a null to null, if the aggregation propagates
    /// nulls.
    ///
    /// # Safety
    /// `groups` must be in bounds for `values`.
    unsafe fn propagate_nulls(&self, agg: Column, values: &Column, groups: &GroupsType) -> Column {
        if !self.agg_type.propagate_nulls || !values.has_nulls() {
            return agg;
        }
        let has_nulls = unsafe { values.is_null().agg_any(groups, true) };
        let validity: Bitmap = has_nulls.iter().map(|b| b != Some(true)).collect();
        agg.mask(&validity)
    }

    pub fn new(
        expr: Arc<dyn PhysicalExpr>,
        agg_type: AggregationType,
        output_field: Field,
    ) -> Self {
        Self {
            input: expr,
            agg_type,
            output_field,
        }
    }
}

impl PhysicalExpr for AggregationExpr {
    fn as_expression(&self) -> Option<&Expr> {
        None
    }

    fn evaluate_impl(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let s = self.input.evaluate(df, state)?;

        let AggregationType {
            groupby,
            allow_threading,
            propagate_nulls,
        } = self.agg_type;

        if propagate_nulls && s.has_nulls() {
            return Ok(Column::full_null(s.name().clone(), 1, s.dtype()));
        }

        let is_float = s.dtype().is_float();
        let group_by = match groupby {
            GroupByMethod::NanMin if !is_float => GroupByMethod::Min,
            GroupByMethod::NanMax if !is_float => GroupByMethod::Max,
            gb => gb,
        };

        match group_by {
            GroupByMethod::Min => match s.is_sorted_flag() {
                IsSorted::Ascending | IsSorted::Descending => {
                    s.min_reduce().map(|sc| sc.into_column(s.name().clone()))
                },
                IsSorted::Not => parallel_op_columns(
                    |s| s.min_reduce().map(|sc| sc.into_column(s.name().clone())),
                    s,
                    allow_threading,
                ),
            },
            #[cfg(feature = "propagate_nans")]
            GroupByMethod::NanMin => parallel_op_columns(
                |s| {
                    Ok(polars_ops::prelude::nan_propagating_aggregate::nan_min_s(
                        s.as_materialized_series(),
                        s.name().clone(),
                    )
                    .into_column())
                },
                s,
                allow_threading,
            ),
            #[cfg(not(feature = "propagate_nans"))]
            GroupByMethod::NanMin => {
                panic!("activate 'propagate_nans' feature")
            },
            GroupByMethod::Max => match s.is_sorted_flag() {
                IsSorted::Ascending | IsSorted::Descending => {
                    s.max_reduce().map(|sc| sc.into_column(s.name().clone()))
                },
                IsSorted::Not => parallel_op_columns(
                    |s| s.max_reduce().map(|sc| sc.into_column(s.name().clone())),
                    s,
                    allow_threading,
                ),
            },
            #[cfg(feature = "propagate_nans")]
            GroupByMethod::NanMax => parallel_op_columns(
                |s| {
                    Ok(polars_ops::prelude::nan_propagating_aggregate::nan_max_s(
                        s.as_materialized_series(),
                        s.name().clone(),
                    )
                    .into_column())
                },
                s,
                allow_threading,
            ),
            #[cfg(not(feature = "propagate_nans"))]
            GroupByMethod::NanMax => {
                panic!("activate 'propagate_nans' feature")
            },
            GroupByMethod::Median => s.median_reduce().map(|sc| sc.into_column(s.name().clone())),
            GroupByMethod::Mean => s.mean_reduce().map(|sc| sc.into_column(s.name().clone())),
            GroupByMethod::First => Ok(if s.is_empty() {
                Column::full_null(s.name().clone(), 1, s.dtype())
            } else {
                s.head(Some(1))
            }),
            GroupByMethod::FirstNonNull => Ok(s
                .as_materialized_series_maintain_scalar()
                .first_non_null()
                .into_column(s.name().clone())),
            GroupByMethod::Last => Ok(if s.is_empty() {
                Column::full_null(s.name().clone(), 1, s.dtype())
            } else {
                s.tail(Some(1))
            }),
            GroupByMethod::LastNonNull => Ok(s
                .as_materialized_series_maintain_scalar()
                .last_non_null()
                .into_column(s.name().clone())),
            GroupByMethod::Item { allow_empty } => Ok(match s.len() {
                0 if allow_empty => Column::full_null(s.name().clone(), 1, s.dtype()),
                1 => s,
                n => polars_bail!(item_agg_count_not_one = n, allow_empty = allow_empty),
            }),
            GroupByMethod::Sum { null_on_empty } => parallel_op_columns(
                |s| {
                    let mut sc = s.sum_reduce()?;
                    if null_on_empty && s.null_count() == s.len() {
                        sc = Scalar::null(sc.dtype().clone());
                    }
                    Ok(sc.into_column(s.name().clone()))
                },
                s,
                allow_threading,
            ),
            GroupByMethod::Groups => unreachable!(),
            GroupByMethod::NUnique => s.n_unique().map(|count| {
                IdxCa::from_slice(s.name().clone(), &[count as IdxSize]).into_column()
            }),
            GroupByMethod::Count { include_nulls } => {
                let count = s.len() - s.null_count() * !include_nulls as usize;

                Ok(IdxCa::from_slice(s.name().clone(), &[count as IdxSize]).into_column())
            },
            GroupByMethod::Implode { maintain_order: _ } => s.implode().map(|ca| ca.into_column()),
            GroupByMethod::Std(ddof) => s
                .std_reduce(ddof)
                .map(|sc| sc.into_column(s.name().clone())),
            GroupByMethod::Var(ddof) => s
                .var_reduce(ddof)
                .map(|sc| sc.into_column(s.name().clone())),
            GroupByMethod::Quantile(_, _) => unimplemented!(),
            GroupByMethod::ArgMin => {
                let opt = s.as_materialized_series().arg_min();
                Ok(opt.map_or_else(
                    || Column::full_null(s.name().clone(), 1, &IDX_DTYPE),
                    |idx| {
                        Column::new_scalar(
                            s.name().clone(),
                            Scalar::new_idxsize(idx.try_into().unwrap()),
                            1,
                        )
                    },
                ))
            },
            GroupByMethod::ArgMax => {
                let opt = s.as_materialized_series().arg_max();
                Ok(opt.map_or_else(
                    || Column::full_null(s.name().clone(), 1, &IDX_DTYPE),
                    |idx| {
                        Column::new_scalar(
                            s.name().clone(),
                            Scalar::new_idxsize(idx.try_into().unwrap()),
                            1,
                        )
                    },
                ))
            },
        }
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups_impl<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut ac = self.input.evaluate_on_groups(df, groups, state)?;

        // don't change names by aggregations as is done in polars-core
        let keep_name = ac.get_values().name().clone();

        if let AggState::LiteralScalar(c) = &mut ac.state {
            *c = self.evaluate(df, state)?;
            return Ok(ac);
        }

        // AggregatedScalar has no defined group structure. We fix it up here, so that we can
        // reliably call `agg_*` functions with the groups.
        ac.set_groups_for_undefined_agg_states();

        // SAFETY:
        // groups must always be in bounds.
        let out = unsafe {
            match self.agg_type.groupby {
                GroupByMethod::Min => {
                    let (c, groups) = ac.get_final_aggregation();
                    let agg_c = c.agg_min(&groups);
                    let agg_c = self.propagate_nulls(agg_c, &c, &groups);
                    AggregatedScalar(agg_c.with_name(keep_name))
                },
                GroupByMethod::Max => {
                    let (c, groups) = ac.get_final_aggregation();
                    let agg_c = c.agg_max(&groups);
                    let agg_c = self.propagate_nulls(agg_c, &c, &groups);
                    AggregatedScalar(agg_c.with_name(keep_name))
                },
                GroupByMethod::ArgMin => {
                    let (c, groups) = ac.get_final_aggregation();
                    let agg_c = c.agg_arg_min(&groups);
                    AggregatedScalar(agg_c.with_name(keep_name))
                },
                GroupByMethod::ArgMax => {
                    let (c, groups) = ac.get_final_aggregation();
                    let agg_c = c.agg_arg_max(&groups);
                    AggregatedScalar(agg_c.with_name(keep_name))
                },
                GroupByMethod::Median => {
                    let (c, groups) = ac.get_final_aggregation();
                    let agg_c = c.agg_median(&groups);
                    AggregatedScalar(agg_c.with_name(keep_name))
                },
                GroupByMethod::Mean => {
                    let (c, groups) = ac.get_final_aggregation();
                    let agg_c = c.agg_mean(&groups);
                    AggregatedScalar(agg_c.with_name(keep_name))
                },
                GroupByMethod::Sum { null_on_empty } => {
                    let (c, groups) = ac.get_final_aggregation();
                    let mut agg_c = c.agg_sum(&groups);
                    if null_on_empty {
                        let counts = c.agg_valid_count(&groups);
                        let has_values: Bitmap =
                            counts.idx()?.into_no_null_iter().map(|n| n > 0).collect();
                        agg_c = agg_c.mask(&has_values);
                    }
                    AggregatedScalar(agg_c.with_name(keep_name))
                },
                GroupByMethod::Count { include_nulls } => {
                    let agg_c = evaluate_count_on_ac(ac, include_nulls)?;
                    AggregatedScalar(agg_c.with_name(keep_name))
                },
                GroupByMethod::First => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_first(&groups);
                    AggregatedScalar(agg_s.with_name(keep_name))
                },
                GroupByMethod::FirstNonNull => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_first_non_null(&groups);
                    AggregatedScalar(agg_s.with_name(keep_name))
                },
                GroupByMethod::Last => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_last(&groups);
                    AggregatedScalar(agg_s.with_name(keep_name))
                },
                GroupByMethod::LastNonNull => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_last_non_null(&groups);
                    AggregatedScalar(agg_s.with_name(keep_name))
                },
                GroupByMethod::Item { allow_empty } => {
                    let (s, groups) = ac.get_final_aggregation();
                    for gc in groups.group_count().iter() {
                        match gc {
                            Some(0) if allow_empty => continue,
                            None | Some(1) => continue,
                            Some(n) => {
                                polars_bail!(item_agg_count_not_one = n, allow_empty = allow_empty);
                            },
                        }
                    }
                    let agg_s = s.agg_first(&groups);
                    AggregatedScalar(agg_s.with_name(keep_name))
                },
                GroupByMethod::NUnique => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_n_unique(&groups);
                    AggregatedScalar(agg_s.with_name(keep_name))
                },
                GroupByMethod::Implode { maintain_order: _ } => {
                    let col = match ac.agg_state() {
                        AggState::LiteralScalar(_) => unreachable!(), // handled above
                        AggState::AggregatedScalar(c) => c.to_unit_list(),
                        AggState::AggregatedList(c) => c.clone(),
                        AggState::NotAggregated(_) => ac.aggregated(),
                    };
                    // TODO: Introduce `UpdateGroups::WithUnitLen` as a new lazy `groups()` method
                    // and move the groups constructor there. Then, set `UpdateGroups::WithUnitLen` to
                    // all AggregationExprs.
                    let groups = Cow::Owned({
                        let groups = (0..col.len() as IdxSize).map(|i| [i, 1]).collect();
                        GroupsType::new_slice(groups, false, true).into_sliceable()
                    });
                    let mut out = AggregationContext::from_agg_state(AggregatedScalar(col), groups);
                    out.set_original_groups(false);
                    return Ok(out);
                },
                GroupByMethod::Groups => {
                    let mut column: ListChunked = ac.groups().as_list_chunked();
                    column.rename(keep_name);
                    AggregatedScalar(column.into_column())
                },
                GroupByMethod::Std(ddof) => {
                    let (c, groups) = ac.get_final_aggregation();
                    let agg_c = c.agg_std(&groups, ddof);
                    AggregatedScalar(agg_c.with_name(keep_name))
                },
                GroupByMethod::Var(ddof) => {
                    let (c, groups) = ac.get_final_aggregation();
                    let agg_c = c.agg_var(&groups, ddof);
                    AggregatedScalar(agg_c.with_name(keep_name))
                },
                GroupByMethod::Quantile(_, _) => {
                    // implemented explicitly in AggQuantile struct
                    unimplemented!()
                },
                GroupByMethod::NanMin => {
                    #[cfg(feature = "propagate_nans")]
                    {
                        let (c, groups) = ac.get_final_aggregation();
                        let agg_c = if c.dtype().is_float() {
                            nan_propagating_aggregate::group_agg_nan_min_s(
                                c.as_materialized_series(),
                                &groups,
                            )
                            .into_column()
                        } else {
                            c.agg_min(&groups)
                        };
                        let agg_c = self.propagate_nulls(agg_c, &c, &groups);
                        AggregatedScalar(agg_c.with_name(keep_name))
                    }
                    #[cfg(not(feature = "propagate_nans"))]
                    {
                        panic!("activate 'propagate_nans' feature")
                    }
                },
                GroupByMethod::NanMax => {
                    #[cfg(feature = "propagate_nans")]
                    {
                        let (c, groups) = ac.get_final_aggregation();
                        let agg_c = if c.dtype().is_float() {
                            nan_propagating_aggregate::group_agg_nan_max_s(
                                c.as_materialized_series(),
                                &groups,
                            )
                            .into_column()
                        } else {
                            c.agg_max(&groups)
                        };
                        let agg_c = self.propagate_nulls(agg_c, &c, &groups);
                        AggregatedScalar(agg_c.with_name(keep_name))
                    }
                    #[cfg(not(feature = "propagate_nans"))]
                    {
                        panic!("activate 'propagate_nans' feature")
                    }
                },
            }
        };

        Ok(AggregationContext::from_agg_state(
            out,
            Cow::Borrowed(groups),
        ))
    }

    fn to_field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        Ok(self.output_field.clone())
    }

    fn is_scalar(&self) -> bool {
        true
    }
}

/// The index of the minimum or the maximum of `s` in the order `sort` uses: NaN is the largest
/// value, and null is the smallest or the largest value. Returns `None` only if `s` is empty.
pub(crate) fn arg_min_max_total(
    s: &Series,
    is_max: bool,
    nulls_largest: bool,
) -> PolarsResult<Option<usize>> {
    if s.is_empty() {
        return Ok(None);
    }
    if is_max == nulls_largest && s.has_nulls() {
        return Ok(first_null(s.chunks().iter().map(|arr| arr.as_ref())));
    }
    if is_max && s.dtype().is_float() {
        let nan_idx = with_match_physical_float_polars_type!(s.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
            first_nan(ca)
        });
        if nan_idx.is_some() {
            return Ok(nan_idx);
        }
    }
    let idx = if is_max { s.arg_max() } else { s.arg_min() };
    Ok(Some(idx.unwrap_or(0)))
}

/// The index of the first NaN that is not null.
pub(crate) fn first_nan<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> Option<usize> {
    let mut offset = 0;
    for arr in ca.downcast_iter() {
        let values = arr.values().as_slice();
        let mut start = 0;
        while let Some(i) = first_nan_idx(&values[start..]) {
            let i = start + i;
            if arr.validity().is_none_or(|validity| validity.get_bit(i)) {
                return Some(offset + i);
            }
            start = i + 1;
        }
        offset += arr.len();
    }
    None
}

pub struct AggMinMaxByExpr {
    input: Arc<dyn PhysicalExpr>,
    by: Arc<dyn PhysicalExpr>,
    is_max_by: bool,
    order: MinMaxByOrder,
}

impl AggMinMaxByExpr {
    pub fn new_min_by(
        input: Arc<dyn PhysicalExpr>,
        by: Arc<dyn PhysicalExpr>,
        order: MinMaxByOrder,
    ) -> Self {
        Self {
            input,
            by,
            is_max_by: false,
            order,
        }
    }

    pub fn new_max_by(
        input: Arc<dyn PhysicalExpr>,
        by: Arc<dyn PhysicalExpr>,
        order: MinMaxByOrder,
    ) -> Self {
        Self {
            input,
            by,
            is_max_by: true,
            order,
        }
    }
}

impl PhysicalExpr for AggMinMaxByExpr {
    fn as_expression(&self) -> Option<&Expr> {
        None
    }

    fn evaluate_impl(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let mut input = self.input.evaluate(df, state)?;
        let mut by = self.by.evaluate(df, state)?;
        if by.len() == 1 && self.by.is_scalar() {
            by = by.new_from_index(0, input.len());
        } else if input.len() == 1 && self.input.is_scalar() {
            input = input.new_from_index(0, by.len());
        }
        let name = if self.is_max_by { "max_by" } else { "min_by" };
        polars_ensure!(
            input.len() == by.len(),
            ShapeMismatch: "'by' column in {} expression has incorrect length: expected {}, got {}",
            name, input.len(), by.len()
        );
        let by = by.as_materialized_series_maintain_scalar();
        let arg_extremum = match self.order {
            MinMaxByOrder::SkipNulls if self.is_max_by => by.arg_max(),
            MinMaxByOrder::SkipNulls => by.arg_min(),
            MinMaxByOrder::Total { nulls_largest } => {
                arg_min_max_total(&by, self.is_max_by, nulls_largest)?
            },
        };
        let out = if let Some(idx) = arg_extremum {
            input.slice(idx as i64, 1)
        } else {
            let dtype = input.dtype().clone();
            Column::new_scalar(input.name().clone(), Scalar::null(dtype), 1)
        };
        Ok(out)
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups_impl<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut ac = self.input.evaluate_on_groups(df, groups, state)?;
        let mut ac_by = self.by.evaluate_on_groups(df, groups, state)?;
        assert!(ac.groups.len() == ac_by.groups.len());

        // Don't change names by aggregations as is done in polars-core
        let keep_name = ac.get_values().name().clone();

        ac.set_groups_for_undefined_agg_states();
        ac_by.set_groups_for_undefined_agg_states();
        if self.input.is_scalar() && ac.broadcast_unit_groups_to(&mut ac_by) {
            ac.normalize_values();
        }
        if self.by.is_scalar() && ac_by.broadcast_unit_groups_to(&mut ac) {
            ac_by.normalize_values();
        }

        let (input_col, input_groups) = ac.get_final_aggregation();
        let (by_col, by_groups) = ac_by.get_final_aggregation();
        GroupsType::check_lengths(&input_groups, &by_groups)?;

        let nulls_largest = match self.order {
            MinMaxByOrder::SkipNulls => None,
            MinMaxByOrder::Total { nulls_largest } => Some(nulls_largest),
        };
        // The arg max kernels may skip NaN, but `sort` puts NaN after all other values.
        let nan_mask = match nulls_largest {
            Some(_) if self.is_max_by && by_col.dtype().is_float() => {
                let is_nan = by_col
                    .as_materialized_series()
                    .is_nan()?
                    .rechunk()
                    .into_owned();
                is_nan.any().then_some(is_nan)
            },
            _ => None,
        };
        let is_exact = nulls_largest.is_none() || (!by_col.has_nulls() && nan_mask.is_none());

        // Fast path: rolling context with numeric by column — O(n) via deque.
        if let GroupsType::Slice {
            groups: slices,
            overlapping,
            monotonic,
        } = by_groups.as_ref().as_ref()
            && input_groups.is_same(&by_groups)
            && is_exact
        {
            let by_phys = by_col.as_materialized_series().to_physical_repr();
            if by_phys.dtype().is_primitive_numeric()
                && !by_col.dtype().is_categorical()
                && _use_rolling_kernels(slices, *overlapping, *monotonic, by_phys.chunks())
            {
                let gather_idxs = rolling_numeric_minmax_by(&by_col, slices, self.is_max_by);
                // SAFETY: The indices come from the same groups as `input_col`.
                let gathered = unsafe { input_col.take_unchecked(&gather_idxs) };
                let agg_state = AggregatedScalar(gathered.with_name(keep_name));
                return Ok(AggregationContext::from_agg_state(
                    agg_state,
                    Cow::Borrowed(groups),
                ));
            }
        }

        // Slow path: per-group agg_arg_min/agg_arg_max.
        let by_col = if by_col.dtype().is_nested() {
            let encoded =
                _get_rows_encoded_ca(by_col.name().clone(), &[by_col], &[false], &[false], true)?;
            encoded.cast(&DataType::Binary)?.into_column()
        } else if by_col.dtype() == &DataType::BinaryOffset {
            by_col.cast(&DataType::Binary)?
        } else {
            by_col
        };
        // SAFETY: Groups are correct.
        let idxs_in_groups = if self.is_max_by {
            unsafe { by_col.agg_arg_max(&by_groups) }
        } else {
            unsafe { by_col.agg_arg_min(&by_groups) }
        };
        let idxs_in_groups: &IdxCa = idxs_in_groups.as_materialized_series().as_ref().as_ref();

        // In the total order these rows come before the pick of the kernel, in this order.
        let mut before = Vec::new();
        if nulls_largest == Some(self.is_max_by) && by_col.has_nulls() {
            before.push(by_col.is_null());
        }
        before.extend(nan_mask);
        let before = before
            .into_iter()
            .map(|mask| {
                // SAFETY: Groups are correct.
                let pos = unsafe { mask.clone().into_series().agg_arg_max(&by_groups) };
                let pos = pos.idx()?.rechunk().downcast_as_array().clone();
                Ok((pos, mask.rechunk().downcast_as_array().clone()))
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        // SAFETY: Callers pass a group and a position within that group.
        let to_global = |groups: &GroupsType, group_idx: usize, i: IdxSize| unsafe {
            match groups {
                GroupsType::Idx(g) => *g.all().get_unchecked(group_idx).get_unchecked(i as usize),
                GroupsType::Slice { groups, .. } => groups.get_unchecked(group_idx)[0] + i,
            }
        };
        let group_len = |group_idx: usize| match by_groups.as_ref().as_ref() {
            GroupsType::Idx(g) => g.all()[group_idx].len(),
            GroupsType::Slice { groups, .. } => groups[group_idx][1] as usize,
        };
        let first_before = |group_idx: usize| {
            before.iter().find_map(|(pos, mask)| {
                // SAFETY: There is one position per group, and it is within the group.
                unsafe {
                    let i = pos.get_unchecked(group_idx)?;
                    let by_idx = to_global(by_groups.as_ref().as_ref(), group_idx, i);
                    mask.get_unchecked(by_idx as usize)?.then_some(i)
                }
            })
        };
        let gather_idxs: IdxCa = idxs_in_groups
            .iter()
            .enumerate()
            .map(|(group_idx, idx_in_group)| {
                // A group without a pick that is not empty only has null keys.
                let fallback =
                    || (nulls_largest.is_some() && group_len(group_idx) > 0).then_some(0);
                let idx_in_group = first_before(group_idx).or(idx_in_group).or_else(fallback)?;
                Some(to_global(
                    input_groups.as_ref().as_ref(),
                    group_idx,
                    idx_in_group,
                ))
            })
            .collect();

        // SAFETY: All non-null indices are within input_col's groups.
        let gathered = unsafe { input_col.take_unchecked(&gather_idxs) };
        let agg_state = AggregatedScalar(gathered.with_name(keep_name));
        Ok(AggregationContext::from_agg_state(
            agg_state,
            Cow::Borrowed(groups),
        ))
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.input.to_field(input_schema)
    }

    fn is_scalar(&self) -> bool {
        true
    }
}

pub(crate) struct AnonymousAggregationExpr {
    pub(crate) inputs: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) grouped_reduction: Box<dyn GroupedReduction>,
    pub(crate) output_field: Field,
}

impl AnonymousAggregationExpr {
    pub fn new(
        inputs: Vec<Arc<dyn PhysicalExpr>>,
        grouped_reduction: Box<dyn GroupedReduction>,
        output_field: Field,
    ) -> Self {
        Self {
            inputs,
            grouped_reduction,
            output_field,
        }
    }
}

impl PhysicalExpr for AnonymousAggregationExpr {
    fn as_expression(&self) -> Option<&Expr> {
        None
    }

    fn evaluate_impl(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let mut cols = self
            .inputs
            .iter()
            .map(|e| e.evaluate(df, state))
            .collect::<PolarsResult<Vec<_>>>()?;
        let len = cols.iter().map(Column::len).max().unwrap();
        for (col, e) in cols.iter_mut().zip(&self.inputs) {
            if col.len() == 1 && e.is_scalar() {
                *col = col.new_from_index(0, len);
            }
        }
        polars_ensure!(
            cols.iter().all(|c| c.len() == len),
            ShapeMismatch: "inputs of an anonymous aggregation have different lengths"
        );

        let mut gr = self.grouped_reduction.new_empty();
        gr.resize(1);
        gr.update_group(&cols.iter().collect::<Vec<_>>(), 0, 0)?;
        let out_series = gr.finalize()?;
        Ok(Column::new(cols[0].name().clone(), out_series))
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups_impl<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut acs = self
            .inputs
            .iter()
            .map(|e| e.evaluate_on_groups(df, groups, state))
            .collect::<PolarsResult<Vec<_>>>()?;

        // don't change names by aggregations as is done in polars-core
        let input_column_name = acs[0].get_values().name().clone();

        if acs
            .iter()
            .all(|ac| matches!(ac.state, AggState::LiteralScalar(_)))
        {
            let mut ac = acs.swap_remove(0);
            ac.state = AggState::LiteralScalar(self.evaluate(df, state)?);
            return Ok(ac);
        }

        for ac in &mut acs {
            ac.set_groups_for_undefined_agg_states();
        }
        if let Some(i) = self.inputs.iter().position(|e| !e.is_scalar()) {
            for j in (0..acs.len()).filter(|&j| j != i && self.inputs[j].is_scalar()) {
                let [ac, target] = acs.get_disjoint_mut([j, i]).unwrap();
                if ac.broadcast_unit_groups_to(target) {
                    ac.normalize_values();
                }
            }
        }

        let (columns, mut input_groups): (Vec<_>, Vec<_>) =
            acs.into_iter().map(|ac| ac.get_final_aggregation()).unzip();
        let resolved_groups = &input_groups[0];
        for g in &input_groups[1..] {
            resolved_groups.check_lengths(g)?;
        }

        let mut gr = self.grouped_reduction.new_empty();
        gr.resize(resolved_groups.len() as IdxSize);

        let total = resolved_groups.iter().map(|g| g.len()).sum::<usize>();
        // The positions of each input's values, group after group.
        let positions = |groups: &GroupsType| -> Vec<IdxSize> {
            let mut out = Vec::with_capacity(total);
            match groups {
                GroupsType::Idx(idx) => {
                    out.extend(idx.all().iter().flat_map(|g| g.iter().copied()))
                },
                GroupsType::Slice { groups, .. } => {
                    out.extend(groups.iter().flat_map(|&[start, len]| start..start + len))
                },
            }
            out
        };
        // `update_groups_subset` needs single chunks. Inputs with their own groups are gathered
        // into group order, so that their values line up.
        let (columns, subset): (Vec<_>, _) =
            if input_groups.iter().all(|g| g.is_same(resolved_groups)) {
                let columns = columns.iter().map(Column::rechunk).collect();
                (columns, positions(resolved_groups))
            } else {
                let columns = columns
                    .iter()
                    .zip(&input_groups)
                    .map(|(c, g)| c.take_slice(&positions(g)))
                    .collect::<PolarsResult<_>>()?;
                (columns, (0..total as IdxSize).collect())
            };
        let mut group_idxs = Vec::with_capacity(total);
        for (group_idx, g) in resolved_groups.iter().enumerate() {
            group_idxs.extend(std::iter::repeat_n(group_idx as IdxSize, g.len()));
        }

        // Single call so no need to resolve ordering.
        let seq_id = 0;

        // SAFETY:
        // - `subset` holds the group positions into `columns`, or `0..total` into the gathered
        //   columns, so it is in-bounds.
        // - Each `group_idxs` entry is `< resolved_groups.len()`, the size passed to `gr.resize`.
        unsafe {
            gr.update_groups_subset(
                &columns.iter().collect::<Vec<_>>(),
                &subset,
                &group_idxs,
                seq_id,
            )?;
        }

        let out_series = gr.finalize()?;
        let out = AggregatedScalar(Column::new(input_column_name, out_series));

        Ok(AggregationContext::from_agg_state(
            out,
            input_groups.swap_remove(0),
        ))
    }

    fn to_field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        Ok(self.output_field.clone())
    }

    fn is_scalar(&self) -> bool {
        true
    }
}

/// Simple wrapper to parallelize functions that can be divided over threads aggregated and
/// finally aggregated in the main thread. This can be done for sum, min, max, etc.
fn parallel_op_columns<F>(f: F, s: Column, allow_threading: bool) -> PolarsResult<Column>
where
    F: Fn(Column) -> PolarsResult<Column> + Send + Sync,
{
    // set during debug low so
    // we mimic production size data behavior
    #[cfg(debug_assertions)]
    let thread_boundary = 0;

    #[cfg(not(debug_assertions))]
    let thread_boundary = 100_000;

    // threading overhead/ splitting work stealing is costly..

    if !allow_threading
        || s.len() < thread_boundary
        || RAYON.current_thread_has_pending_tasks().unwrap_or(false)
    {
        return f(s);
    }
    let n_threads = RAYON.current_num_threads();
    let splits = _split_offsets(s.len(), n_threads);

    let chunks = RAYON.install(|| {
        splits
            .into_par_iter()
            .map(|(offset, len)| {
                let s = s.slice(offset as i64, len);
                f(s)
            })
            .collect::<PolarsResult<Vec<_>>>()
    })?;

    let mut iter = chunks.into_iter();
    let first = iter.next().unwrap();
    let dtype = first.dtype();
    let out = iter.fold(first.to_physical_repr(), |mut acc, s| {
        acc.append(&s.to_physical_repr()).unwrap();
        acc
    });

    unsafe { f(out.from_physical_unchecked(dtype).unwrap()) }
}
