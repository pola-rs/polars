use std::sync::Arc;

use polars_arrow::utils::CustomIterTools;
use polars_core::frame::groupby::GroupsProxy;
use polars_core::prelude::*;

use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;

pub struct SortExpr {
    pub(crate) physical_expr: Arc<dyn PhysicalExpr>,
    pub(crate) options: SortOptions,
    expr: Expr,
}

impl SortExpr {
    pub fn new(physical_expr: Arc<dyn PhysicalExpr>, options: SortOptions, expr: Expr) -> Self {
        Self {
            physical_expr,
            options,
            expr,
        }
    }
}

/// Map arg_sort result back to the indices on the `GroupIdx`
pub(crate) fn map_sorted_indices_to_group_idx(sorted_idx: &IdxCa, idx: &[IdxSize]) -> Vec<IdxSize> {
    sorted_idx
        .cont_slice()
        .unwrap()
        .iter()
        .map(|&i| {
            debug_assert!(idx.get(i as usize).is_some());
            unsafe { *idx.get_unchecked(i as usize) }
        })
        .collect_trusted()
}

pub(crate) fn map_sorted_indices_to_group_slice(
    sorted_idx: &IdxCa,
    first: IdxSize,
) -> Vec<IdxSize> {
    sorted_idx
        .cont_slice()
        .unwrap()
        .iter()
        .map(|&i| i + first)
        .collect_trusted()
}

impl PhysicalExpr for SortExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Series> {
        let series = self.physical_expr.evaluate(df, state)?;
        Ok(series.sort_with(self.options))
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut ac = self.physical_expr.evaluate_on_groups(df, groups, state)?;
        match ac.agg_state() {
            AggState::AggregatedList(s) => {
                let ca = s.list().unwrap();
                let out = ca.lst_sort(self.options);
                ac.with_series(out.into_series(), true);
            }
            _ => {
                let series = ac.flat_naive().into_owned();

                let groups = match ac.groups().as_ref() {
                    GroupsProxy::Idx(groups) => {
                        groups
                            .iter()
                            .map(|(first, idx)| {
                                // Safety:
                                // Group tuples are always in bounds
                                let group = unsafe {
                                    series.take_iter_unchecked(&mut idx.iter().map(|i| *i as usize))
                                };

                                let sorted_idx = group.arg_sort(self.options);
                                let new_idx = map_sorted_indices_to_group_idx(&sorted_idx, idx);
                                (new_idx.first().copied().unwrap_or(first), new_idx)
                            })
                            .collect()
                    }
                    GroupsProxy::Slice { groups, .. } => groups
                        .iter()
                        .map(|&[first, len]| {
                            let group = series.slice(first as i64, len as usize);
                            let sorted_idx = group.arg_sort(self.options);
                            let new_idx = map_sorted_indices_to_group_slice(&sorted_idx, first);
                            (new_idx.first().copied().unwrap_or(first), new_idx)
                        })
                        .collect(),
                };
                let groups = GroupsProxy::Idx(groups);
                ac.with_groups(groups);
            }
        }

        Ok(ac)
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.physical_expr.to_field(input_schema)
    }

    fn is_valid_aggregation(&self) -> bool {
        true
    }
}
