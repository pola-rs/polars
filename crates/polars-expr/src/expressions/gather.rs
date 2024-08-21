use arrow::legacy::utils::CustomIterTools;
use polars_core::chunked_array::builder::get_list_builder;
use polars_core::prelude::*;
use polars_core::utils::NoNull;
use polars_ops::prelude::{convert_to_unsigned_index, is_positive_idx_uncertain};
use polars_utils::slice::GetSaferUnchecked;

use super::*;
use crate::expressions::{AggState, AggregationContext, PhysicalExpr, UpdateGroups};

pub struct GatherExpr {
    pub(crate) phys_expr: Arc<dyn PhysicalExpr>,
    pub(crate) idx: Arc<dyn PhysicalExpr>,
    pub(crate) expr: Expr,
    pub(crate) returns_scalar: bool,
}

impl PhysicalExpr for GatherExpr {
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

        let s_idx = idx.series();
        match s_idx.dtype() {
            DataType::List(inner) => {
                polars_ensure!(inner.is_integer(), InvalidOperation: "expected numeric dtype as index, got {:?}", inner)
            },
            dt if dt.is_integer() => {
                // Unsigned integers will fall through and will use faster paths.
                if !is_positive_idx_uncertain(s_idx) {
                    return self.process_negative_indices_agg(ac, idx, groups);
                }
            },
            dt => polars_bail!(InvalidOperation: "expected numeric dtype as index, got {:?}", dt),
        }

        let idx = match idx.state {
            AggState::AggregatedScalar(s) => {
                let idx = s.cast(&IDX_DTYPE)?;
                return self.process_positive_indices_agg_scalar(ac, idx.idx().unwrap());
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
                return self.process_positive_indices_agg_literal(ac, idx.idx().unwrap());
            },
        };

        let s = idx.cast(&DataType::List(Box::new(IDX_DTYPE)))?;
        let idx = s.list().unwrap();

        let taken = {
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
}

impl GatherExpr {
    fn finish(
        &self,
        df: &DataFrame,
        state: &ExecutionState,
        series: Series,
    ) -> PolarsResult<Series> {
        let idx = self.idx.evaluate(df, state)?;
        let idx = convert_to_unsigned_index(&idx, series.len())?;
        series.take(&idx)
    }

    fn oob_err(&self) -> PolarsResult<()> {
        polars_bail!(expr = self.expr, OutOfBounds: "index out of bounds");
    }

    fn process_positive_indices_agg_scalar<'b>(
        &self,
        mut ac: AggregationContext<'b>,
        idx: &IdxCa,
    ) -> PolarsResult<AggregationContext<'b>> {
        if ac.is_not_aggregated() {
            // A previous aggregation may have updated the groups.
            let groups = ac.groups();

            // Determine the gather indices.
            let idx: IdxCa = match groups.as_ref() {
                GroupsProxy::Idx(groups) => {
                    if groups.all().iter().zip(idx).any(|(g, idx)| match idx {
                        None => true,
                        Some(idx) => idx >= g.len() as IdxSize,
                    }) {
                        self.oob_err()?;
                    }

                    idx.into_iter()
                        .zip(groups.iter())
                        .map(|(idx, (_first, groups))| {
                            idx.map(|idx| {
                                // SAFETY:
                                // we checked bounds
                                unsafe {
                                    *groups.get_unchecked_release(usize::try_from(idx).unwrap())
                                }
                            })
                        })
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
            Ok(ac)
        } else {
            self.gather_aggregated_expensive(ac, idx)
        }
    }

    fn gather_aggregated_expensive<'b>(
        &self,
        mut ac: AggregationContext<'b>,
        idx: &IdxCa,
    ) -> PolarsResult<AggregationContext<'b>> {
        let out = ac
            .aggregated()
            .list()
            .unwrap()
            .try_apply_amortized(|s| s.as_ref().take(idx))?;

        ac.with_series(out.into_series(), true, Some(&self.expr))?;
        ac.with_update_groups(UpdateGroups::WithGroupsLen);
        Ok(ac)
    }

    fn process_positive_indices_agg_literal<'b>(
        &self,
        mut ac: AggregationContext<'b>,
        idx: &IdxCa,
    ) -> PolarsResult<AggregationContext<'b>> {
        if idx.len() == 1 {
            match idx.get(0) {
                None => polars_bail!(ComputeError: "cannot take by a null"),
                Some(idx) => {
                    // Make sure that we look at the updated groups.
                    let groups = ac.groups();

                    // We offset the groups first by idx.
                    let idx: NoNull<IdxCa> = match groups.as_ref() {
                        GroupsProxy::Idx(groups) => {
                            if groups.all().iter().any(|g| idx >= g.len() as IdxSize) {
                                self.oob_err()?;
                            }

                            groups
                                .iter()
                                .map(|(_, group)| {
                                    // SAFETY: we just bound checked.
                                    unsafe { *group.get_unchecked_release(idx as usize) }
                                })
                                .collect_trusted()
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
            self.gather_aggregated_expensive(ac, idx)
        }
    }

    fn process_negative_indices_agg<'b>(
        &self,
        mut ac: AggregationContext<'b>,
        mut idx: AggregationContext<'b>,
        groups: &'b GroupsProxy,
    ) -> PolarsResult<AggregationContext<'b>> {
        let mut builder = get_list_builder(
            &ac.dtype(),
            idx.series().len(),
            groups.len(),
            ac.series().name(),
        )?;

        let iter = ac.iter_groups(false).zip(idx.iter_groups(false));
        for (s, idx) in iter {
            match (s, idx) {
                (Some(s), Some(idx)) => {
                    let idx = convert_to_unsigned_index(idx.as_ref(), s.as_ref().len())?;
                    let out = s.as_ref().take(&idx)?;
                    builder.append_series(&out)?;
                },
                _ => builder.append_null(),
            };
        }
        let out = builder.finish().into_series();
        ac.with_agg_state(AggState::AggregatedList(out));
        Ok(ac)
    }
}
