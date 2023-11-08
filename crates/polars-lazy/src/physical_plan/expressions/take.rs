use std::sync::Arc;

use arrow::legacy::utils::CustomIterTools;
use polars_core::frame::group_by::GroupsProxy;
use polars_core::prelude::*;
use polars_core::utils::NoNull;

use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;

pub struct TakeExpr {
    pub(crate) phys_expr: Arc<dyn PhysicalExpr>,
    pub(crate) idx: Arc<dyn PhysicalExpr>,
    pub(crate) expr: Expr,
    pub(crate) returns_scalar: bool,
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
        polars_bail!(expr = self.expr, ComputeError: "index out of bounds");
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

        let idx = match idx.state {
            AggState::AggregatedScalar(s) => {
                let idx = s.cast(&IDX_DTYPE)?;
                let idx = idx.idx().unwrap();

                // The indexes are AggregatedScalar, meaning they are a single values pointing into
                // a group. If we zip this with the first of each group -> `idx + firs` then we can
                // simply use a take operation on the whole array instead of per group.

                // The groups maybe scattered all over the place, so we sort by group.
                ac.sort_by_groups();

                // A previous aggregation may have updated the groups.
                let groups = ac.groups();

                // Determine the take indices.
                let idx: IdxCa = match groups.as_ref() {
                    GroupsProxy::Idx(groups) => {
                        if groups.all().iter().zip(idx).any(|(g, idx)| match idx {
                            None => true,
                            Some(idx) => idx >= g.len() as IdxSize,
                        }) {
                            self.oob_err()?;
                        }

                        idx.into_iter()
                            .zip(groups.first().iter())
                            .map(|(idx, first)| idx.map(|idx| idx + first))
                            .collect_trusted()
                    },
                    GroupsProxy::Slice { groups, .. } => {
                        if groups.iter().zip(idx).any(|(g, idx)| match idx {
                            None => true,
                            Some(idx) => idx >= g[1],
                        }) {
                            self.oob_err()?;
                        }

                        idx.into_iter()
                            .zip(groups.iter())
                            .map(|(idx, g)| idx.map(|idx| idx + g[0]))
                            .collect_trusted()
                    },
                };
                let taken = ac.flat_naive().take(&idx)?;

                let taken = if self.returns_scalar {
                    taken
                } else {
                    taken.as_list().into_series()
                };

                ac.with_series(taken, true, Some(&self.expr))?;
                return Ok(ac);
            },
            AggState::AggregatedList(s) => {
                polars_ensure!(!self.returns_scalar, ComputeError: "expected single index");
                s.list().unwrap().clone()
            },
            // Maybe a literal as well, this needs a different path.
            AggState::NotAggregated(_) => {
                polars_ensure!(!self.returns_scalar, ComputeError: "expected single index");
                let s = idx.aggregated();
                s.list().unwrap().clone()
            },
            AggState::Literal(s) => {
                let idx = s.cast(&IDX_DTYPE)?;
                let idx = idx.idx().unwrap();

                return if idx.len() == 1 {
                    match idx.get(0) {
                        None => polars_bail!(ComputeError: "cannot take by a null"),
                        Some(idx) => {
                            if idx != 0 {
                                // We must make sure that the column we take from is sorted by
                                // groups otherwise we might point into the wrong group.
                                ac.sort_by_groups()
                            }
                            // Make sure that we look at the updated groups.
                            let groups = ac.groups();

                            // We offset the groups first by idx.
                            let idx: NoNull<IdxCa> = match groups.as_ref() {
                                GroupsProxy::Idx(groups) => {
                                    if groups.all().iter().any(|g| idx >= g.len() as IdxSize) {
                                        self.oob_err()?;
                                    }

                                    groups.first().iter().map(|f| *f + idx).collect_trusted()
                                },
                                GroupsProxy::Slice { groups, .. } => {
                                    if groups.iter().any(|g| idx >= g[1]) {
                                        self.oob_err()?;
                                    }

                                    groups.iter().map(|g| g[0] + idx).collect_trusted()
                                },
                            };
                            let taken = ac.flat_naive().take(&idx.into_inner())?;

                            let taken = if self.returns_scalar {
                                taken
                            } else {
                                taken.as_list().into_series()
                            };

                            ac.with_series(taken, true, Some(&self.expr))?;
                            ac.with_update_groups(UpdateGroups::WithGroupsLen);
                            Ok(ac)
                        },
                    }
                } else {
                    let out = ac
                        .aggregated()
                        .list()
                        .unwrap()
                        .try_apply_amortized(|s| s.as_ref().take(idx))?;

                    ac.with_series(out.into_series(), true, Some(&self.expr))?;
                    ac.with_update_groups(UpdateGroups::WithGroupsLen);
                    Ok(ac)
                };
            },
        };

        let s = idx.cast(&DataType::List(Box::new(IDX_DTYPE)))?;
        let idx = s.list().unwrap();

        let taken = unsafe {
            ac.aggregated()
                .list()
                .unwrap()
                .amortized_iter()
                .zip(idx.amortized_iter())
                .map(|(s, idx)| Some(s?.as_ref().take(idx?.as_ref().idx().unwrap())))
                .map(|opt_res| opt_res.transpose())
                .collect::<PolarsResult<ListChunked>>()?
                .with_name(ac.series().name())
        };

        ac.with_series(taken.into_series(), true, Some(&self.expr))?;
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
