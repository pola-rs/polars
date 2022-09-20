use std::sync::Arc;

use polars_arrow::is_valid::IsValid;
use polars_core::frame::groupby::GroupsProxy;
use polars_core::prelude::*;
use polars_core::POOL;
use rayon::prelude::*;

use crate::physical_plan::state::ExecutionState;
use crate::prelude::UpdateGroups::WithSeriesLen;
use crate::prelude::*;

pub struct FilterExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) by: Arc<dyn PhysicalExpr>,
    expr: Expr,
}

impl FilterExpr {
    pub fn new(input: Arc<dyn PhysicalExpr>, by: Arc<dyn PhysicalExpr>, expr: Expr) -> Self {
        Self { input, by, expr }
    }
}

impl PhysicalExpr for FilterExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Series> {
        let s_f = || self.input.evaluate(df, state);
        let predicate_f = || self.by.evaluate(df, state);

        let (series, predicate) = POOL.install(|| rayon::join(s_f, predicate_f));
        let (series, predicate) = (series?, predicate?);

        series.filter(predicate.bool()?)
    }

    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let ac_s_f = || self.input.evaluate_on_groups(df, groups, state);
        let ac_predicate_f = || self.by.evaluate_on_groups(df, groups, state);

        let (ac_s, ac_predicate) = POOL.install(|| rayon::join(ac_s_f, ac_predicate_f));
        let (mut ac_s, mut ac_predicate) = (ac_s?, ac_predicate?);

        match ac_predicate.is_aggregated() {
            true => {
                let preds = ac_predicate.iter_groups();
                let s = ac_s.aggregated();
                let ca = s.list()?;
                let mut out = ca
                    .amortized_iter()
                    .zip(preds)
                    .map(|(opt_s, opt_pred)| match (opt_s, opt_pred) {
                        (Some(s), Some(pred)) => s.as_ref().filter(pred.as_ref().bool()?).map(Some),
                        _ => Ok(None),
                    })
                    .collect::<PolarsResult<ListChunked>>()?;
                out.rename(s.name());
                ac_s.with_series(out.into_series(), true);
                ac_s.update_groups = WithSeriesLen;
                Ok(ac_s)
            }
            false => {
                let groups = ac_s.groups();
                let predicate_s = ac_predicate.flat_naive();
                let predicate = predicate_s.bool()?.rechunk();

                // all values true don't do anything
                if predicate.all() {
                    return Ok(ac_s);
                }
                // all values false
                // create empty groups
                let groups = if !predicate.any() {
                    let groups = groups.iter().map(|gi| [gi.first(), 0]).collect::<Vec<_>>();
                    GroupsProxy::Slice {
                        groups,
                        rolling: false,
                    }
                }
                // filter the indexes that are true
                else {
                    let predicate = predicate.downcast_iter().next().unwrap();
                    POOL.install(|| {
                        match groups.as_ref() {
                            GroupsProxy::Idx(groups) => {
                                let groups = groups
                                    .par_iter()
                                    .map(|(first, idx)| unsafe {
                                        let idx: Vec<IdxSize> = idx
                                            .iter()
                                            // Safety:
                                            // just checked bounds in short circuited lhs
                                            .filter_map(|i| {
                                                match predicate.value(*i as usize)
                                                    && predicate.is_valid_unchecked(*i as usize)
                                                {
                                                    true => Some(*i),
                                                    _ => None,
                                                }
                                            })
                                            .collect();

                                        (*idx.first().unwrap_or(&first), idx)
                                    })
                                    .collect();

                                GroupsProxy::Idx(groups)
                            }
                            GroupsProxy::Slice { groups, .. } => {
                                let groups = groups
                                    .par_iter()
                                    .map(|&[first, len]| unsafe {
                                        let idx: Vec<IdxSize> = (first..first + len)
                                            // Safety:
                                            // just checked bounds in short circuited lhs
                                            .filter(|&i| {
                                                predicate.value(i as usize)
                                                    && predicate.is_valid_unchecked(i as usize)
                                            })
                                            .collect();

                                        (*idx.first().unwrap_or(&first), idx)
                                    })
                                    .collect();
                                GroupsProxy::Idx(groups)
                            }
                        }
                    })
                };

                ac_s.with_groups(groups).set_original_len(false);
                Ok(ac_s)
            }
        }
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.input.to_field(input_schema)
    }

    fn is_valid_aggregation(&self) -> bool {
        self.input.is_valid_aggregation()
    }
}
