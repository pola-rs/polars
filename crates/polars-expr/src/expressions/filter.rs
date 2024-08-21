use arrow::legacy::is_valid::IsValid;
use polars_core::prelude::*;
use polars_core::POOL;
use polars_utils::idx_vec::IdxVec;
use rayon::prelude::*;

use super::*;
use crate::expressions::UpdateGroups::WithSeriesLen;
use crate::expressions::{AggregationContext, PhysicalExpr};

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
        // Check if the groups are still equal, otherwise aggregate.
        // TODO! create a special group iters that don't materialize
        if ac_s.groups.as_ref() as *const _ != ac_predicate.groups.as_ref() as *const _ {
            let _ = ac_s.aggregated();
            let _ = ac_predicate.aggregated();
        }

        if ac_predicate.is_aggregated() || ac_s.is_aggregated() {
            let preds = ac_predicate.iter_groups(false);
            let s = ac_s.aggregated();
            let ca = s.list()?;
            let out = if ca.is_empty() {
                // return an empty list if ca is empty.
                ListChunked::full_null_with_dtype(ca.name(), 0, ca.inner_dtype())
            } else {
                {
                    ca.amortized_iter()
                        .zip(preds)
                        .map(|(opt_s, opt_pred)| match (opt_s, opt_pred) {
                            (Some(s), Some(pred)) => {
                                s.as_ref().filter(pred.as_ref().bool()?).map(Some)
                            },
                            _ => Ok(None),
                        })
                        .collect::<PolarsResult<ListChunked>>()?
                        .with_name(s.name())
                }
            };
            ac_s.with_series(out.into_series(), true, Some(&self.expr))?;
            ac_s.update_groups = WithSeriesLen;
            Ok(ac_s)
        } else {
            let groups = ac_s.groups();
            let predicate_s = ac_predicate.flat_naive();
            let predicate = predicate_s.bool()?;

            // All values true - don't do anything.
            if let Some(true) = predicate.all_kleene() {
                return Ok(ac_s);
            }
            // All values false - create empty groups.
            let groups = if !predicate.any() {
                let groups = groups.iter().map(|gi| [gi.first(), 0]).collect::<Vec<_>>();
                GroupsProxy::Slice {
                    groups,
                    rolling: false,
                }
            }
            // Filter the indexes that are true.
            else {
                let predicate = predicate.rechunk();
                let predicate = predicate.downcast_iter().next().unwrap();
                POOL.install(|| {
                    match groups.as_ref() {
                        GroupsProxy::Idx(groups) => {
                            let groups = groups
                                .par_iter()
                                .map(|(first, idx)| unsafe {
                                    let idx: IdxVec = idx
                                        .iter()
                                        .copied()
                                        .filter(|i| {
                                            // SAFETY: just checked bounds in short circuited lhs.
                                            predicate.value(*i as usize)
                                                && predicate.is_valid_unchecked(*i as usize)
                                        })
                                        .collect();

                                    (*idx.first().unwrap_or(&first), idx)
                                })
                                .collect();

                            GroupsProxy::Idx(groups)
                        },
                        GroupsProxy::Slice { groups, .. } => {
                            let groups = groups
                                .par_iter()
                                .map(|&[first, len]| unsafe {
                                    let idx: IdxVec = (first..first + len)
                                        .filter(|&i| {
                                            // SAFETY: just checked bounds in short circuited lhs
                                            predicate.value(i as usize)
                                                && predicate.is_valid_unchecked(i as usize)
                                        })
                                        .collect();

                                    (*idx.first().unwrap_or(&first), idx)
                                })
                                .collect();
                            GroupsProxy::Idx(groups)
                        },
                    }
                })
            };

            ac_s.with_groups(groups).set_original_len(false);
            Ok(ac_s)
        }
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.input.to_field(input_schema)
    }
}
