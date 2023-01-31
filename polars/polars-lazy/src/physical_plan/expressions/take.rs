use std::sync::Arc;

use polars_arrow::utils::CustomIterTools;
use polars_core::frame::groupby::GroupsProxy;
use polars_core::prelude::*;
use polars_core::utils::NoNull;

use crate::physical_plan::expression_err;
use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;

pub struct TakeExpr {
    pub(crate) phys_expr: Arc<dyn PhysicalExpr>,
    pub(crate) idx: Arc<dyn PhysicalExpr>,
    pub(crate) expr: Expr,
}

impl TakeExpr {
    fn finish(
        &self,
        df: &DataFrame,
        state: &ExecutionState,
        series: Series,
    ) -> PolarsResult<Series> {
        let idx = self.idx.evaluate(df, state)?;

        let nulls_before_cast = idx.null_count();

        let idx = idx.cast(&IDX_DTYPE)?;
        if idx.null_count() != nulls_before_cast {
            self.oob_err()?;
        }
        let idx_ca = idx.idx()?;

        series.take(idx_ca)
    }

    fn oob_err(&self) -> PolarsResult<()> {
        let msg = "Out of bounds.";
        Err(expression_err!(msg, self.expr, ComputeError))
    }
}

impl PhysicalExpr for TakeExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Series> {
        let series = self.phys_expr.evaluate(df, state)?;
        self.finish(df, state, series)
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut ac = self.phys_expr.evaluate_on_groups(df, groups, state)?;
        let mut idx = self.idx.evaluate_on_groups(df, groups, state)?;

        let idx =
            match idx.state {
                AggState::AggregatedFlat(s) => {
                    let idx = s.cast(&IDX_DTYPE)?;
                    let idx = idx.idx().unwrap();

                    // The indexes are AggregatedFlat, meaning they are a single values pointing into
                    // a group.
                    // If we zip this with the first of each group -> `idx + firs` then we can
                    // simply use a take operation on the whole array instead of per group.

                    // The groups maybe scattered all over the place, so we sort by group
                    ac.sort_by_groups();

                    // A previous aggregation may have updated the groups
                    let groups = ac.groups();

                    // Determine the take indices
                    let idx: IdxCa =
                        match groups.as_ref() {
                            GroupsProxy::Idx(groups) => {
                                if groups.all().iter().zip(idx.into_iter()).any(
                                    |(g, idx)| match idx {
                                        None => true,
                                        Some(idx) => idx >= g.len() as IdxSize,
                                    },
                                ) {
                                    self.oob_err()?;
                                }

                                idx.into_iter()
                                    .zip(groups.first().iter())
                                    .map(|(idx, first)| idx.map(|idx| idx + first))
                                    .collect_trusted()
                            }
                            GroupsProxy::Slice { groups, .. } => {
                                if groups
                                    .iter()
                                    .zip(idx.into_iter())
                                    .any(|(g, idx)| match idx {
                                        None => true,
                                        Some(idx) => idx >= g[1],
                                    })
                                {
                                    self.oob_err()?;
                                }

                                idx.into_iter()
                                    .zip(groups.iter())
                                    .map(|(idx, g)| idx.map(|idx| idx + g[0]))
                                    .collect_trusted()
                            }
                        };
                    let taken = ac.flat_naive().take(&idx)?;
                    ac.with_series(taken, true);
                    return Ok(ac);
                }
                AggState::AggregatedList(s) => s.list().unwrap().clone(),
                // Maybe a literal as well, this needs a different path
                AggState::NotAggregated(_) => {
                    let s = idx.aggregated();
                    s.list().unwrap().clone()
                }
                AggState::Literal(s) => {
                    let idx = s.cast(&IDX_DTYPE)?;
                    let idx = idx.idx().unwrap();

                    return if idx.len() == 1 {
                        match idx.get(0) {
                            None => Err(PolarsError::ComputeError("cannot take by a null".into())),
                            Some(idx) => {
                                if idx != 0 {
                                    // We must make sure that the column we take from is sorted by
                                    // groups otherwise we might point into the wrong group
                                    ac.sort_by_groups()
                                }
                                // Make sure that we look at the updated groups.
                                let groups = ac.groups();

                                // we offset the groups first by idx;
                                let idx: NoNull<IdxCa> = match groups.as_ref() {
                                    GroupsProxy::Idx(groups) => {
                                        if groups.all().iter().any(|g| idx >= g.len() as IdxSize) {
                                            self.oob_err()?;
                                        }

                                        groups.first().iter().map(|f| *f + idx).collect_trusted()
                                    }
                                    GroupsProxy::Slice { groups, .. } => {
                                        if groups.iter().any(|g| idx >= g[1]) {
                                            self.oob_err()?;
                                        }

                                        groups.iter().map(|g| g[0] + idx).collect_trusted()
                                    }
                                };
                                let taken = ac.flat_naive().take(&idx.into_inner())?;
                                ac.with_series(taken, true);
                                ac.with_update_groups(UpdateGroups::WithGroupsLen);
                                Ok(ac)
                            }
                        }
                    } else {
                        let out = ac
                            .aggregated()
                            .list()
                            .unwrap()
                            .try_apply_amortized(|s| s.as_ref().take(idx))?;

                        ac.with_series(out.into_series(), true);
                        ac.with_update_groups(UpdateGroups::WithGroupsLen);
                        Ok(ac)
                    };
                }
            };

        let s = idx.cast(&DataType::List(Box::new(IDX_DTYPE)))?;
        let idx = s.list().unwrap();

        let mut taken = ac
            .aggregated()
            .list()
            .unwrap()
            .amortized_iter()
            .zip(idx.amortized_iter())
            .map(|(s, idx)| {
                s.and_then(|s| {
                    idx.map(|idx| {
                        let idx = idx.as_ref().idx().unwrap();
                        s.as_ref().take(idx)
                    })
                })
                .transpose()
            })
            .collect::<PolarsResult<ListChunked>>()?;

        taken.rename(ac.series().name());

        ac.with_series(taken.into_series(), true);
        ac.with_update_groups(UpdateGroups::WithGroupsLen);
        Ok(ac)
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.phys_expr.to_field(input_schema)
    }

    fn is_valid_aggregation(&self) -> bool {
        true
    }
}
