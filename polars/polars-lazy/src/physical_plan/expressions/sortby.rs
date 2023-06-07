use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use polars_core::frame::groupby::{GroupsIndicator, GroupsProxy};
use polars_core::prelude::*;
use polars_core::POOL;
use rayon::prelude::*;

use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;

pub struct SortByExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) by: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) descending: Vec<bool>,
    pub(crate) expr: Expr,
}

impl SortByExpr {
    pub fn new(
        input: Arc<dyn PhysicalExpr>,
        by: Vec<Arc<dyn PhysicalExpr>>,
        descending: Vec<bool>,
        expr: Expr,
    ) -> Self {
        Self {
            input,
            by,
            descending,
            expr,
        }
    }
}

fn prepare_descending(descending: &[bool], by_len: usize) -> Vec<bool> {
    match (descending.len(), by_len) {
        // equal length
        (n_rdescending, n) if n_rdescending == n => descending.to_vec(),
        // none given all false
        (0, n) => vec![false; n],
        // broadcast first
        (_, n) => vec![descending[0]; n],
    }
}

impl PhysicalExpr for SortByExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Series> {
        let series_f = || self.input.evaluate(df, state);
        let descending = prepare_descending(&self.descending, self.by.len());

        let (series, sorted_idx) = if self.by.len() == 1 {
            let sorted_idx_f = || {
                let s_sort_by = self.by[0].evaluate(df, state)?;
                Ok(s_sort_by.arg_sort(SortOptions {
                    descending: descending[0],
                    ..Default::default()
                }))
            };
            POOL.install(|| rayon::join(series_f, sorted_idx_f))
        } else {
            let sorted_idx_f = || {
                let s_sort_by = self
                    .by
                    .iter()
                    .map(|e| {
                        e.evaluate(df, state).map(|s| match s.dtype() {
                            #[cfg(feature = "dtype-categorical")]
                            DataType::Categorical(_) => s,
                            _ => s.to_physical_repr().into_owned(),
                        })
                    })
                    .collect::<PolarsResult<Vec<_>>>()?;

                let options = SortMultipleOptions {
                    other: s_sort_by[1..].to_vec(),
                    descending,
                    multithreaded: true,
                };
                s_sort_by[0].arg_sort_multiple(&options)
            };
            POOL.install(|| rayon::join(series_f, sorted_idx_f))
        };
        let (sorted_idx, series) = (sorted_idx?, series?);
        polars_ensure!(
            sorted_idx.len() == series.len(),
            expr = self.expr, ComputeError:
            "`sort_by` produced different length: {} than the series that has to be sorted: {}",
            sorted_idx.len(), series.len()
        );

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
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut ac_in = self.input.evaluate_on_groups(df, groups, state)?;
        // if the length of the sort_by argument differs
        // we raise an error
        let invalid = AtomicBool::new(false);

        // the groups of the lhs of the expressions do not match the series values
        // we must take the slower path.
        if !matches!(ac_in.update_groups, UpdateGroups::No) {
            polars_ensure!(
                self.by.len() <= 1, expr = self.expr, ComputeError:
                "this expression is not supported for more than two sort columns"
            );
            let mut ac_sort_by = self.by[0].evaluate_on_groups(df, groups, state)?;
            let sort_by = ac_sort_by.aggregated();
            let mut sort_by = sort_by.list().unwrap().clone();
            let s = ac_in.aggregated();
            let mut s = s.list().unwrap().clone();

            let descending = self.descending[0];
            let mut ca: ListChunked = POOL.install(|| {
                s.par_iter_indexed()
                    .zip(sort_by.par_iter_indexed())
                    .map(|(opt_s, s_sort_by)| match (opt_s, s_sort_by) {
                        (Some(s), Some(s_sort_by)) => {
                            if s.len() != s_sort_by.len() {
                                invalid.store(true, Ordering::Relaxed);
                                None
                            } else {
                                let idx = s_sort_by.arg_sort(SortOptions {
                                    descending,
                                    // we are already in par iter.
                                    multithreaded: false,
                                    ..Default::default()
                                });
                                Some(unsafe { s.take_unchecked(&idx).unwrap() })
                            }
                        }
                        _ => None,
                    })
                    .collect()
            });
            ca.rename(s.name());
            let s = ca.into_series();
            ac_in.with_series(s, true, Some(&self.expr))?;
            Ok(ac_in)
        } else {
            let descending = prepare_descending(&self.descending, self.by.len());

            let (groups, ordered_by_group_operation) = if self.by.len() == 1 {
                let mut ac_sort_by = self.by[0].evaluate_on_groups(df, groups, state)?;
                let sort_by_s = ac_sort_by.flat_naive().into_owned();
                polars_ensure!(
                    sort_by_s.len() == ac_in.flat_naive().len(), expr = self.expr, ComputeError:
                    "the expression in `sort_by` argument must result in the same length"
                );
                let ordered_by_group_operation = matches!(
                    ac_sort_by.update_groups,
                    UpdateGroups::WithSeriesLen | UpdateGroups::WithGroupsLen
                );
                let groups = ac_sort_by.groups();

                let groups = POOL.install(|| {
                    groups
                        .par_iter()
                        .map(|indicator| {
                            let new_idx = match indicator {
                                GroupsIndicator::Idx((_, idx)) => {
                                    // Safety:
                                    // Group tuples are always in bounds
                                    let group = unsafe {
                                        sort_by_s.take_iter_unchecked(
                                            &mut idx.iter().map(|i| *i as usize),
                                        )
                                    };

                                    let sorted_idx = group.arg_sort(SortOptions {
                                        descending: descending[0],
                                        // we are already in par iter.
                                        multithreaded: false,
                                        ..Default::default()
                                    });
                                    map_sorted_indices_to_group_idx(&sorted_idx, idx)
                                }
                                GroupsIndicator::Slice([first, len]) => {
                                    let group = sort_by_s.slice(first as i64, len as usize);
                                    let sorted_idx = group.arg_sort(SortOptions {
                                        descending: descending[0],
                                        // we are already in par iter.
                                        multithreaded: false,
                                        ..Default::default()
                                    });
                                    map_sorted_indices_to_group_slice(&sorted_idx, first)
                                }
                            };
                            let first = new_idx.first().unwrap_or_else(|| {
                                invalid.store(true, Ordering::Relaxed);
                                &0
                            });

                            (*first, new_idx)
                        })
                        .collect()
                });

                (GroupsProxy::Idx(groups), ordered_by_group_operation)
            } else {
                let mut ac_sort_by = self
                    .by
                    .iter()
                    .map(|e| e.evaluate_on_groups(df, groups, state))
                    .collect::<PolarsResult<Vec<_>>>()?;
                let sort_by_s = ac_sort_by
                    .iter()
                    .map(|s| {
                        let s = s.flat_naive();
                        match s.dtype() {
                            #[cfg(feature = "dtype-categorical")]
                            DataType::Categorical(_) => s.into_owned(),
                            _ => s.to_physical_repr().into_owned(),
                        }
                    })
                    .collect::<Vec<_>>();

                for sort_by_s in &sort_by_s {
                    polars_ensure!(
                        sort_by_s.len() == ac_in.flat_naive().len(), expr = self.expr, ComputeError:
                        "the expression in `sort_by` argument must result in the same length"
                    );
                }

                let ordered_by_group_operation = matches!(
                    ac_sort_by[0].update_groups,
                    UpdateGroups::WithSeriesLen | UpdateGroups::WithGroupsLen
                );
                let groups = ac_sort_by[0].groups();

                let groups = POOL.install(|| {
                    groups
                        .par_iter()
                        .map(|indicator| {
                            let new_idx = match indicator {
                                GroupsIndicator::Idx((_first, idx)) => {
                                    // Safety:
                                    // Group tuples are always in bounds
                                    let groups = sort_by_s
                                        .iter()
                                        .map(|s| unsafe {
                                            s.take_iter_unchecked(
                                                &mut idx.iter().map(|i| *i as usize),
                                            )
                                        })
                                        .collect::<Vec<_>>();

                                    let options = SortMultipleOptions {
                                        other: groups[1..].to_vec(),
                                        descending: descending.clone(),
                                        multithreaded: false,
                                    };

                                    let sorted_idx = groups[0].arg_sort_multiple(&options).unwrap();
                                    map_sorted_indices_to_group_idx(&sorted_idx, idx)
                                }
                                GroupsIndicator::Slice([first, len]) => {
                                    let groups = sort_by_s
                                        .iter()
                                        .map(|s| s.slice(first as i64, len as usize))
                                        .collect::<Vec<_>>();

                                    let options = SortMultipleOptions {
                                        other: groups[1..].to_vec(),
                                        descending: descending.clone(),
                                        multithreaded: false,
                                    };
                                    let sorted_idx = groups[0].arg_sort_multiple(&options).unwrap();
                                    map_sorted_indices_to_group_slice(&sorted_idx, first)
                                }
                            };
                            let first = new_idx.first().unwrap_or_else(|| {
                                invalid.store(true, Ordering::Relaxed);
                                &0
                            });

                            (*first, new_idx)
                        })
                        .collect()
                });

                (GroupsProxy::Idx(groups), ordered_by_group_operation)
            };
            polars_ensure!(
                !invalid.load(Ordering::Relaxed), expr = self.expr, ComputeError:
                "the expression in `sort_by` argument must result in the same length"
            );

            // if the rhs is already aggregated once,
            // it is reordered by the groupby operation
            // we must ensure that we are as well.
            if ordered_by_group_operation {
                let s = ac_in.aggregated();
                ac_in.with_series(s.explode().unwrap(), false, None)?;
            }

            ac_in.with_groups(groups);
            Ok(ac_in)
        }
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.input.to_field(input_schema)
    }

    fn is_valid_aggregation(&self) -> bool {
        true
    }
}
