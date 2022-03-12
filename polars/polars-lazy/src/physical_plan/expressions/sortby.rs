use crate::physical_plan::state::ExecutionState;
use crate::prelude::sort::{map_sorted_indices_to_group_idx, map_sorted_indices_to_group_slice};
use crate::prelude::*;
use polars_core::frame::groupby::{GroupsIndicator, GroupsProxy};
use polars_core::prelude::*;
use polars_core::POOL;
use rayon::prelude::*;
use std::sync::Arc;

pub struct SortByExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) by: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) reverse: Vec<bool>,
    pub(crate) expr: Expr,
}

impl SortByExpr {
    pub fn new(
        input: Arc<dyn PhysicalExpr>,
        by: Vec<Arc<dyn PhysicalExpr>>,
        reverse: Vec<bool>,
        expr: Expr,
    ) -> Self {
        Self {
            input,
            by,
            reverse,
            expr,
        }
    }
}

fn prepare_reverse(reverse: &[bool], by_len: usize) -> Vec<bool> {
    match (reverse.len(), by_len) {
        // equal length
        (n_reverse, n) if n_reverse == n => reverse.to_vec(),
        // none given all false
        (0, n) => vec![false; n],
        // broadcast first
        (_, n) => vec![reverse[0]; n],
    }
}

impl PhysicalExpr for SortByExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let series_f = || self.input.evaluate(df, state);
        let reverse = prepare_reverse(&self.reverse, self.by.len());

        let (series, sorted_idx) = if self.by.len() == 1 {
            let sorted_idx_f = || {
                let s_sort_by = self.by[0].evaluate(df, state)?;
                Ok(s_sort_by.argsort(SortOptions {
                    descending: reverse[0],
                    ..Default::default()
                }))
            };
            POOL.install(|| rayon::join(series_f, sorted_idx_f))
        } else {
            let sorted_idx_f = || {
                let s_sort_by = self
                    .by
                    .iter()
                    .map(|e| e.evaluate(df, state))
                    .collect::<Result<Vec<_>>>()?;

                s_sort_by[0].argsort_multiple(&s_sort_by[1..], &reverse)
            };
            POOL.install(|| rayon::join(series_f, sorted_idx_f))
        };
        let (sorted_idx, series) = (sorted_idx?, series?);

        // Safety:
        // sorted index are within bounds
        unsafe { series.take_unchecked(&sorted_idx) }
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        let mut ac_in = self.input.evaluate_on_groups(df, groups, state)?;
        let reverse = prepare_reverse(&self.reverse, self.by.len());

        let groups = if self.by.len() == 1 {
            let mut ac_sort_by = self.by[0].evaluate_on_groups(df, groups, state)?;
            let sort_by_s = ac_sort_by.flat_naive().into_owned();
            let groups = ac_sort_by.groups();

            let groups = groups
                .par_iter()
                .map(|indicator| {
                    let new_idx = match indicator {
                        GroupsIndicator::Idx((_, idx)) => {
                            // Safety:
                            // Group tuples are always in bounds
                            let group = unsafe {
                                sort_by_s.take_iter_unchecked(&mut idx.iter().map(|i| *i as usize))
                            };

                            let sorted_idx = group.argsort(SortOptions {
                                descending: reverse[0],
                                ..Default::default()
                            });
                            map_sorted_indices_to_group_idx(&sorted_idx, idx)
                        }
                        GroupsIndicator::Slice([first, len]) => {
                            let group = sort_by_s.slice(first as i64, len as usize);
                            let sorted_idx = group.argsort(SortOptions {
                                descending: reverse[0],
                                ..Default::default()
                            });
                            map_sorted_indices_to_group_slice(&sorted_idx, first)
                        }
                    };

                    (new_idx[0], new_idx)
                })
                .collect();

            GroupsProxy::Idx(groups)
        } else {
            let mut ac_sort_by = self
                .by
                .iter()
                .map(|e| e.evaluate_on_groups(df, groups, state))
                .collect::<Result<Vec<_>>>()?;
            let sort_by_s = ac_sort_by
                .iter()
                .map(|s| s.flat_naive().into_owned())
                .collect::<Vec<_>>();
            let groups = ac_sort_by[0].groups();

            let groups = groups
                .par_iter()
                .map(|indicator| {
                    let new_idx = match indicator {
                        GroupsIndicator::Idx((_first, idx)) => {
                            // Safety:
                            // Group tuples are always in bounds
                            let groups = sort_by_s
                                .iter()
                                .map(|s| unsafe {
                                    s.take_iter_unchecked(&mut idx.iter().map(|i| *i as usize))
                                })
                                .collect::<Vec<_>>();

                            let sorted_idx =
                                groups[0].argsort_multiple(&groups[1..], &reverse).unwrap();
                            map_sorted_indices_to_group_idx(&sorted_idx, idx)
                        }
                        GroupsIndicator::Slice([first, len]) => {
                            let groups = sort_by_s
                                .iter()
                                .map(|s| s.slice(first as i64, len as usize))
                                .collect::<Vec<_>>();
                            let sorted_idx =
                                groups[0].argsort_multiple(&groups[1..], &reverse).unwrap();
                            map_sorted_indices_to_group_slice(&sorted_idx, first)
                        }
                    };

                    (new_idx[0], new_idx)
                })
                .collect();

            GroupsProxy::Idx(groups)
        };

        ac_in.with_groups(groups);
        Ok(ac_in)
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.input.to_field(input_schema)
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}
impl PhysicalAggregation for SortByExpr {
    // As a final aggregation a Sort returns a list array.
    fn aggregate(
        &self,
        df: &DataFrame,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        let mut ac_in = self.input.evaluate_on_groups(df, groups, state)?;
        let err = || {
            PolarsError::ComputeError(
                format!("cannot aggregate {:?} as list array", self.expr).into(),
            )
        };
        let reverse = prepare_reverse(&self.reverse, self.by.len());

        let mut agg_s = if self.by.len() == 1 {
            let reverse = reverse[0];
            let s_sort_by = self.by[0].evaluate(df, state)?;

            let s_sort_by = s_sort_by.agg_list(groups).ok_or_else(err)?;

            let agg_s = ac_in.aggregated();
            agg_s
                .list()
                .unwrap()
                .amortized_iter()
                .zip(s_sort_by.list().unwrap().amortized_iter())
                .map(|(opt_s, opt_sort_by)| {
                    match (opt_s, opt_sort_by) {
                        (Some(s), Some(sort_by)) => {
                            let sorted_idx = sort_by.as_ref().argsort(SortOptions {
                                descending: reverse,
                                ..Default::default()
                            });
                            // Safety:
                            // sorted index are within bounds
                            unsafe { s.as_ref().take_unchecked(&sorted_idx) }.ok()
                        }
                        _ => None,
                    }
                })
                .collect::<ListChunked>()
                .into_series()
        } else {
            let s_sort_by = self
                .by
                .iter()
                .map(|e| e.evaluate(df, state)?.agg_list(groups).ok_or_else(err))
                .collect::<Result<Vec<_>>>()?;

            let mut sort_by_iters = s_sort_by
                .iter()
                .map(|s| s.list().unwrap().amortized_iter())
                .collect::<Vec<_>>();
            let mut items = vec![Default::default(); sort_by_iters.len()];

            let agg_s = ac_in.aggregated();
            agg_s
                .list()
                .unwrap()
                .amortized_iter()
                .map(|opt_s| {
                    match opt_s {
                        None => Ok(None),
                        Some(s) => {
                            let mut all_some = true;
                            sort_by_iters.iter_mut().zip(items.iter_mut()).for_each(
                                |(iter, item)| {
                                    // we unwrap the iterator, because it should be same length as agg_s
                                    match iter.next().unwrap() {
                                        Some(s_sort) => {
                                            *item = s_sort.as_ref().clone();
                                        }
                                        None => {
                                            all_some = false;
                                        }
                                    }
                                },
                            );
                            if all_some {
                                let sorted_idx =
                                    items[0].argsort_multiple(&items[1..], &reverse)?;
                                unsafe { s.as_ref().take_unchecked(&sorted_idx) }.map(Some)
                            } else {
                                Ok(None)
                            }
                        }
                    }
                })
                .collect::<Result<ListChunked>>()?
                .into_series()
        };
        agg_s.rename(ac_in.series().name());
        Ok(Some(agg_s))
    }
}
