use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_arrow::utils::CustomIterTools;
use polars_core::frame::groupby::GroupsProxy;
use polars_core::prelude::*;
use std::sync::Arc;

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

/// Map argsort result back to the indices on the `GroupIdx`
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
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let series = self.physical_expr.evaluate(df, state)?;
        Ok(series.sort_with(self.options))
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        let mut ac = self.physical_expr.evaluate_on_groups(df, groups, state)?;
        let series = ac.flat_naive().into_owned();

        let groups = match ac.groups().as_ref() {
            GroupsProxy::Idx(groups) => {
                groups
                    .iter()
                    .map(|(_first, idx)| {
                        // Safety:
                        // Group tuples are always in bounds
                        let group = unsafe {
                            series.take_iter_unchecked(&mut idx.iter().map(|i| *i as usize))
                        };

                        let sorted_idx = group.argsort(self.options);
                        let new_idx = map_sorted_indices_to_group_idx(&sorted_idx, idx);
                        (new_idx[0], new_idx)
                    })
                    .collect()
            }
            GroupsProxy::Slice(groups) => groups
                .iter()
                .map(|&[first, len]| {
                    let group = series.slice(first as i64, len as usize);
                    let sorted_idx = group.argsort(self.options);
                    let new_idx = map_sorted_indices_to_group_slice(&sorted_idx, first);
                    (new_idx[0], new_idx)
                })
                .collect(),
        };
        let groups = GroupsProxy::Idx(groups);

        ac.with_groups(groups);

        Ok(ac)
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.physical_expr.to_field(input_schema)
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}
impl PhysicalAggregation for SortExpr {
    // As a final aggregation a Sort returns a list array.
    fn aggregate(
        &self,
        df: &DataFrame,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        let mut ac = self.physical_expr.evaluate_on_groups(df, groups, state)?;
        let agg_s = ac.aggregated();
        let agg_s = agg_s
            .list()
            .unwrap()
            .apply_amortized(|s| s.as_ref().sort_with(self.options))
            .into_series();
        Ok(Some(agg_s))
    }
}
