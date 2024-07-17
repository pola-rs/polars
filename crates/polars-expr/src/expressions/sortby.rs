use polars_core::chunked_array::from_iterator_par::ChunkedCollectParIterExt;
use polars_core::prelude::*;
use polars_core::POOL;
use polars_utils::idx_vec::IdxVec;
use rayon::prelude::*;

use super::*;
use crate::expressions::{
    map_sorted_indices_to_group_idx, map_sorted_indices_to_group_slice, AggregationContext,
    PhysicalExpr, UpdateGroups,
};

pub struct SortByExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) by: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) expr: Expr,
    pub(crate) sort_options: SortMultipleOptions,
}

impl SortByExpr {
    pub fn new(
        input: Arc<dyn PhysicalExpr>,
        by: Vec<Arc<dyn PhysicalExpr>>,
        expr: Expr,
        sort_options: SortMultipleOptions,
    ) -> Self {
        Self {
            input,
            by,
            expr,
            sort_options,
        }
    }
}

fn prepare_bool_vec(values: &[bool], by_len: usize) -> Vec<bool> {
    match (values.len(), by_len) {
        // Equal length.
        (n_rvalues, n) if n_rvalues == n => values.to_vec(),
        // None given all false.
        (0, n) => vec![false; n],
        // Broadcast first.
        (_, n) => vec![values[0]; n],
    }
}

static ERR_MSG: &str = "expressions in 'sort_by' produced a different number of groups";

fn check_groups(a: &GroupsProxy, b: &GroupsProxy) -> PolarsResult<()> {
    polars_ensure!(a.iter().zip(b.iter()).all(|(a, b)| {
        a.len() == b.len()
    }), ComputeError: ERR_MSG);
    Ok(())
}

pub(super) fn update_groups_sort_by(
    groups: &GroupsProxy,
    sort_by_s: &Series,
    options: &SortOptions,
) -> PolarsResult<GroupsProxy> {
    // Will trigger a gather for every group, so rechunk before.
    let sort_by_s = sort_by_s.rechunk();
    let groups = POOL.install(|| {
        groups
            .par_iter()
            .map(|indicator| sort_by_groups_single_by(indicator, &sort_by_s, options))
            .collect::<PolarsResult<_>>()
    })?;

    Ok(GroupsProxy::Idx(groups))
}

fn sort_by_groups_single_by(
    indicator: GroupsIndicator,
    sort_by_s: &Series,
    options: &SortOptions,
) -> PolarsResult<(IdxSize, IdxVec)> {
    let options = SortOptions {
        descending: options.descending,
        nulls_last: options.nulls_last,
        // We are already in par iter.
        multithreaded: false,
        ..Default::default()
    };
    let new_idx = match indicator {
        GroupsIndicator::Idx((_, idx)) => {
            // SAFETY: group tuples are always in bounds.
            let group = unsafe { sort_by_s.take_slice_unchecked(idx) };

            let sorted_idx = group.arg_sort(options);
            map_sorted_indices_to_group_idx(&sorted_idx, idx)
        },
        GroupsIndicator::Slice([first, len]) => {
            let group = sort_by_s.slice(first as i64, len as usize);
            let sorted_idx = group.arg_sort(options);
            map_sorted_indices_to_group_slice(&sorted_idx, first)
        },
    };
    let first = new_idx
        .first()
        .ok_or_else(|| polars_err!(ComputeError: "{}", ERR_MSG))?;

    Ok((*first, new_idx))
}

fn sort_by_groups_no_match_single<'a>(
    mut ac_in: AggregationContext<'a>,
    mut ac_by: AggregationContext<'a>,
    descending: bool,
    expr: &Expr,
) -> PolarsResult<AggregationContext<'a>> {
    let s_in = ac_in.aggregated();
    let s_by = ac_by.aggregated();
    let mut s_in = s_in.list().unwrap().clone();
    let mut s_by = s_by.list().unwrap().clone();

    let dtype = s_in.dtype().clone();
    let ca: PolarsResult<ListChunked> = POOL.install(|| {
        s_in.par_iter_indexed()
            .zip(s_by.par_iter_indexed())
            .map(|(opt_s, s_sort_by)| match (opt_s, s_sort_by) {
                (Some(s), Some(s_sort_by)) => {
                    polars_ensure!(s.len() == s_sort_by.len(), ComputeError: "series lengths don't match in 'sort_by' expression");
                    let idx = s_sort_by.arg_sort(SortOptions {
                        descending,
                        // We are already in par iter.
                        multithreaded: false,
                        ..Default::default()
                    });
                    Ok(Some(unsafe { s.take_unchecked(&idx) }))
                },
                _ => Ok(None),
            })
            .collect_ca_with_dtype("", dtype)
    });
    let s = ca?.with_name(s_in.name()).into_series();
    ac_in.with_series(s, true, Some(expr))?;
    Ok(ac_in)
}

fn sort_by_groups_multiple_by(
    indicator: GroupsIndicator,
    sort_by_s: &[Series],
    descending: &[bool],
    nulls_last: &[bool],
    multithreaded: bool,
    maintain_order: bool,
) -> PolarsResult<(IdxSize, IdxVec)> {
    let new_idx = match indicator {
        GroupsIndicator::Idx((_first, idx)) => {
            // SAFETY: group tuples are always in bounds.
            let groups = sort_by_s
                .iter()
                .map(|s| unsafe { s.take_slice_unchecked(idx) })
                .collect::<Vec<_>>();

            let options = SortMultipleOptions {
                descending: descending.to_owned(),
                nulls_last: nulls_last.to_owned(),
                multithreaded,
                maintain_order,
            };

            let sorted_idx = groups[0].arg_sort_multiple(&groups[1..], &options).unwrap();
            map_sorted_indices_to_group_idx(&sorted_idx, idx)
        },
        GroupsIndicator::Slice([first, len]) => {
            let groups = sort_by_s
                .iter()
                .map(|s| s.slice(first as i64, len as usize))
                .collect::<Vec<_>>();

            let options = SortMultipleOptions {
                descending: descending.to_owned(),
                nulls_last: nulls_last.to_owned(),
                multithreaded,
                maintain_order,
            };
            let sorted_idx = groups[0].arg_sort_multiple(&groups[1..], &options).unwrap();
            map_sorted_indices_to_group_slice(&sorted_idx, first)
        },
    };
    let first = new_idx
        .first()
        .ok_or_else(|| polars_err!(ComputeError: "{}", ERR_MSG))?;

    Ok((*first, new_idx))
}

impl PhysicalExpr for SortByExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Series> {
        let series_f = || self.input.evaluate(df, state);
        let (series, sorted_idx) = if self.by.len() == 1 {
            let sorted_idx_f = || {
                let s_sort_by = self.by[0].evaluate(df, state)?;
                Ok(s_sort_by.arg_sort(SortOptions::from(&self.sort_options)))
            };
            POOL.install(|| rayon::join(series_f, sorted_idx_f))
        } else {
            let descending = prepare_bool_vec(&self.sort_options.descending, self.by.len());
            let nulls_last = prepare_bool_vec(&self.sort_options.nulls_last, self.by.len());

            let sorted_idx_f = || {
                let s_sort_by = self
                    .by
                    .iter()
                    .map(|e| {
                        e.evaluate(df, state).map(|s| match s.dtype() {
                            #[cfg(feature = "dtype-categorical")]
                            DataType::Categorical(_, _) | DataType::Enum(_, _) => s,
                            _ => s.to_physical_repr().into_owned(),
                        })
                    })
                    .collect::<PolarsResult<Vec<_>>>()?;

                let options = self
                    .sort_options
                    .clone()
                    .with_order_descending_multi(descending)
                    .with_nulls_last_multi(nulls_last);

                for i in 1..s_sort_by.len() {
                    polars_ensure!(
                        s_sort_by[0].len() == s_sort_by[i].len(),
                        expr = self.expr, ComputeError:
                        "`sort_by` produced different length ({}) than earlier Series' length in `by` ({})",
                        s_sort_by[0].len(), s_sort_by[i].len()
                    );
                }

                s_sort_by[0].arg_sort_multiple(&s_sort_by[1..], &options)
            };
            POOL.install(|| rayon::join(series_f, sorted_idx_f))
        };
        let (sorted_idx, series) = (sorted_idx?, series?);
        polars_ensure!(
            sorted_idx.len() == series.len(),
            expr = self.expr, ComputeError:
            "`sort_by` produced different length ({}) than the Series that has to be sorted ({})",
            sorted_idx.len(), series.len()
        );

        // SAFETY: sorted index are within bounds.
        unsafe { Ok(series.take_unchecked(&sorted_idx)) }
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut ac_in = self.input.evaluate_on_groups(df, groups, state)?;
        let descending = prepare_bool_vec(&self.sort_options.descending, self.by.len());
        let nulls_last = prepare_bool_vec(&self.sort_options.nulls_last, self.by.len());

        let mut ac_sort_by = self
            .by
            .iter()
            .map(|e| e.evaluate_on_groups(df, groups, state))
            .collect::<PolarsResult<Vec<_>>>()?;
        let mut sort_by_s = ac_sort_by
            .iter()
            .map(|s| {
                let s = s.flat_naive();
                match s.dtype() {
                    #[cfg(feature = "dtype-categorical")]
                    DataType::Categorical(_, _) | DataType::Enum(_, _) => s.into_owned(),
                    _ => s.to_physical_repr().into_owned(),
                }
            })
            .collect::<Vec<_>>();

        // A check up front to ensure the input expressions have the same number of total elements.
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

        let groups = if self.by.len() == 1 {
            let mut ac_sort_by = ac_sort_by.pop().unwrap();

            // The groups of the lhs of the expressions do not match the series values,
            // we must take the slower path.
            if !matches!(ac_in.update_groups, UpdateGroups::No) {
                return sort_by_groups_no_match_single(
                    ac_in,
                    ac_sort_by,
                    self.sort_options.descending[0],
                    &self.expr,
                );
            };

            let sort_by_s = sort_by_s.pop().unwrap();
            let groups = ac_sort_by.groups();

            let (check, groups) = POOL.join(
                || check_groups(groups, ac_in.groups()),
                || {
                    update_groups_sort_by(
                        groups,
                        &sort_by_s,
                        &SortOptions {
                            descending: descending[0],
                            nulls_last: nulls_last[0],
                            ..Default::default()
                        },
                    )
                },
            );
            check?;

            groups?
        } else {
            let groups = ac_sort_by[0].groups();

            let groups = POOL.install(|| {
                groups
                    .par_iter()
                    .map(|indicator| {
                        sort_by_groups_multiple_by(
                            indicator,
                            &sort_by_s,
                            &descending,
                            &nulls_last,
                            self.sort_options.multithreaded,
                            self.sort_options.maintain_order,
                        )
                    })
                    .collect::<PolarsResult<_>>()
            });
            GroupsProxy::Idx(groups?)
        };

        // If the rhs is already aggregated once, it is reordered by the
        // group_by operation - we must ensure that we are as well.
        if ordered_by_group_operation {
            let s = ac_in.aggregated();
            ac_in.with_series(s.explode().unwrap(), false, None)?;
        }

        ac_in.with_groups(groups);
        Ok(ac_in)
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.input.to_field(input_schema)
    }
}
