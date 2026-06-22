use std::borrow::Cow;

use polars_core::prelude::*;
use polars_core::runtime::RAYON;
use polars_core::series::IsSorted;
use polars_core::utils::_split_offsets;
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
}

pub(crate) struct AggregationExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) agg_type: AggregationType,
    pub(crate) output_field: Field,
}

impl AggregationExpr {
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
        } = self.agg_type;

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
            GroupByMethod::Sum => parallel_op_columns(
                |s| s.sum_reduce().map(|sc| sc.into_column(s.name().clone())),
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
                    AggregatedScalar(agg_c.with_name(keep_name))
                },
                GroupByMethod::Max => {
                    let (c, groups) = ac.get_final_aggregation();
                    let agg_c = c.agg_max(&groups);
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
                GroupByMethod::Sum => {
                    let (c, groups) = ac.get_final_aggregation();
                    let agg_c = c.agg_sum(&groups);
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
                        AggState::AggregatedScalar(c) => c.as_list().into_column(),
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

pub struct AggMinMaxByExpr {
    input: Arc<dyn PhysicalExpr>,
    by: Arc<dyn PhysicalExpr>,
    is_max_by: bool,
}

impl AggMinMaxByExpr {
    pub fn new_min_by(input: Arc<dyn PhysicalExpr>, by: Arc<dyn PhysicalExpr>) -> Self {
        Self {
            input,
            by,
            is_max_by: false,
        }
    }

    pub fn new_max_by(input: Arc<dyn PhysicalExpr>, by: Arc<dyn PhysicalExpr>) -> Self {
        Self {
            input,
            by,
            is_max_by: true,
        }
    }
}

impl PhysicalExpr for AggMinMaxByExpr {
    fn as_expression(&self) -> Option<&Expr> {
        None
    }

    fn evaluate_impl(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let input = self.input.evaluate(df, state)?;
        let by = self.by.evaluate(df, state)?;
        let name = if self.is_max_by { "max_by" } else { "min_by" };
        polars_ensure!(
            input.len() == by.len(),
            ShapeMismatch: "'by' column in {} expression has incorrect length: expected {}, got {}",
            name, input.len(), by.len()
        );
        let arg_extremum = if self.is_max_by {
            by.as_materialized_series_maintain_scalar().arg_max()
        } else {
            by.as_materialized_series_maintain_scalar().arg_min()
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
        let ac = self.input.evaluate_on_groups(df, groups, state)?;
        let ac_by = self.by.evaluate_on_groups(df, groups, state)?;
        assert!(ac.groups.len() == ac_by.groups.len());

        // Don't change names by aggregations as is done in polars-core
        let keep_name = ac.get_values().name().clone();

        let (input_col, input_groups) = ac.get_final_aggregation();
        let (by_col, by_groups) = ac_by.get_final_aggregation();
        GroupsType::check_lengths(&input_groups, &by_groups)?;

        // Dispatch to arg_min/arg_max and then gather
        // SAFETY: Groups are correct.
        let idxs_in_groups = if self.is_max_by {
            unsafe { by_col.agg_arg_max(&by_groups) }
        } else {
            unsafe { by_col.agg_arg_min(&by_groups) }
        };
        let idxs_in_groups: &IdxCa = idxs_in_groups.as_materialized_series().as_ref().as_ref();
        let gather_idxs: IdxCa = match input_groups.as_ref().as_ref() {
            GroupsType::Idx(g) => idxs_in_groups
                .iter()
                .enumerate()
                .map(|(group_idx, idx_in_group)| {
                    idx_in_group.map(|i| g.all()[group_idx][i as usize])
                })
                .collect(),
            GroupsType::Slice { groups, .. } => idxs_in_groups
                .iter()
                .enumerate()
                .map(|(group_idx, idx_in_group)| idx_in_group.map(|i| groups[group_idx][0] + i))
                .collect(),
        };

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
        polars_ensure!(
            self.inputs.len() == 1,
            ComputeError: "AnonymousAggregationExpr with more than one input is not supported"
        );

        let col = self.inputs[0].evaluate(df, state)?;
        let mut gr = self.grouped_reduction.new_empty();
        gr.resize(1);
        gr.update_group(&[&col], 0, 0)?;
        let out_series = gr.finalize()?;
        Ok(Column::new(col.name().clone(), out_series))
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups_impl<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        polars_ensure!(
            self.inputs.len() == 1,
            ComputeError: "AnonymousAggregationExpr with more than one input is not supported"
        );

        let input = &self.inputs[0];
        let mut ac = input.evaluate_on_groups(df, groups, state)?;

        // don't change names by aggregations as is done in polars-core
        let input_column_name = ac.get_values().name().clone();

        if let AggState::LiteralScalar(input_column) = &mut ac.state {
            *input_column = self.evaluate(df, state)?;
            return Ok(ac);
        }

        let (input_column, resolved_groups) = ac.get_final_aggregation();

        let mut gr = self.grouped_reduction.new_empty();
        gr.resize(groups.len() as IdxSize);

        assert!(
            !resolved_groups.is_overlapping(),
            "Aggregating with overlapping groups is a logic error"
        );

        let subset = (0..input_column.len() as IdxSize).collect::<Vec<IdxSize>>();

        let mut group_idxs = Vec::with_capacity(input_column.len());
        match &**resolved_groups {
            GroupsType::Idx(group_indices) => {
                group_idxs.resize(input_column.len(), 0);
                for (group_idx, indices_in_group) in group_indices.all().iter().enumerate() {
                    for pos in indices_in_group.iter() {
                        group_idxs[*pos as usize] = group_idx as IdxSize;
                    }
                }
            },
            GroupsType::Slice { groups, .. } => {
                for (group_idx, [_start, len]) in groups.iter().enumerate() {
                    group_idxs.extend(std::iter::repeat_n(group_idx as IdxSize, *len as usize));
                }
            },
        };
        assert_eq!(group_idxs.len(), input_column.len());

        // `update_groups_subset` needs a single chunk.
        let input_column_rechunked = input_column.rechunk();

        // Single call so no need to resolve ordering.
        let seq_id = 0;

        // SAFETY:
        // - `subset` is in-bounds because it is 0..N
        // - `group_idxs` is in-bounds because we checked that it matches `input_column.len()` *and*
        //   is filled with values <= `input_column.len()` since they are derived from it via
        //   `enumerate`.
        unsafe {
            gr.update_groups_subset(&[&input_column_rechunked], &subset, &group_idxs, seq_id)?;
        }

        let out_series = gr.finalize()?;
        let out = AggregatedScalar(Column::new(input_column_name, out_series));

        Ok(AggregationContext::from_agg_state(out, resolved_groups))
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
