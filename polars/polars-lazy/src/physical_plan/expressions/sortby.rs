use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::frame::groupby::GroupTuples;
use polars_core::prelude::*;
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
        let series = self.input.evaluate(df, state)?;
        let reverse = prepare_reverse(&self.reverse, self.by.len());

        let sorted_idx = if self.by.len() == 1 {
            let s_sort_by = self.by[0].evaluate(df, state)?;
            s_sort_by.argsort(reverse[0])
        } else {
            let s_sort_by = self
                .by
                .iter()
                .map(|e| e.evaluate(df, state))
                .collect::<Result<Vec<_>>>()?;

            s_sort_by[0].argsort_multiple(&s_sort_by[1..], &reverse)?
        };

        // Safety:
        // sorted index are within bounds
        unsafe { series.take_unchecked(&sorted_idx) }
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupTuples,
        state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        let mut ac_in = self.input.evaluate_on_groups(df, groups, state)?;
        let reverse = prepare_reverse(&self.reverse, self.by.len());

        let groups = if self.by.len() == 1 {
            let mut ac_sort_by = self.by[0].evaluate_on_groups(df, groups, state)?;
            let sort_by_s = ac_sort_by.flat().into_owned();
            let groups = ac_sort_by.groups();

            groups
                .par_iter()
                .map(|(_first, idx)| {
                    // Safety:
                    // Group tuples are always in bounds
                    let group = unsafe {
                        sort_by_s.take_iter_unchecked(&mut idx.iter().map(|i| *i as usize))
                    };

                    let sorted_idx = group.argsort(reverse[0]);

                    let new_idx: Vec<_> = sorted_idx
                        .cont_slice()
                        .unwrap()
                        .iter()
                        .map(|&i| {
                            debug_assert!(idx.get(i as usize).is_some());
                            unsafe { *idx.get_unchecked(i as usize) }
                        })
                        .collect();
                    (new_idx[0], new_idx)
                })
                .collect()
        } else {
            let mut ac_sort_by = self
                .by
                .iter()
                .map(|e| e.evaluate_on_groups(df, groups, state))
                .collect::<Result<Vec<_>>>()?;
            let sort_by_s = ac_sort_by
                .iter()
                .map(|s| s.flat().into_owned())
                .collect::<Vec<_>>();
            let groups = ac_sort_by[0].groups();

            groups
                .par_iter()
                .map(|(_first, idx)| {
                    // Safety:
                    // Group tuples are always in bounds
                    let groups = sort_by_s
                        .iter()
                        .map(|s| unsafe {
                            s.take_iter_unchecked(&mut idx.iter().map(|i| *i as usize))
                        })
                        .collect::<Vec<_>>();

                    let sorted_idx = groups[0].argsort_multiple(&groups[1..], &reverse).unwrap();

                    let new_idx: Vec<_> = sorted_idx
                        .cont_slice()
                        .unwrap()
                        .iter()
                        .map(|&i| {
                            debug_assert!(idx.get(i as usize).is_some());
                            unsafe { *idx.get_unchecked(i as usize) }
                        })
                        .collect();
                    (new_idx[0], new_idx)
                })
                .collect()
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
        groups: &GroupTuples,
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
                            let sorted_idx = sort_by.as_ref().argsort(reverse);
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
