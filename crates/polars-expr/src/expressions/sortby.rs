use polars_core::chunked_array::from_iterator_par::ChunkedCollectParIterExt;
use polars_core::prelude::sort::arg_sort;
use polars_core::prelude::*;
use polars_core::runtime::RAYON;
use polars_utils::broadcast::broadcast_len;
use polars_utils::idx_vec::IdxVec;
use rayon::prelude::*;

use super::*;
use crate::expressions::{
    AggregationContext, PhysicalExpr, UpdateGroups, map_sorted_indices_to_group_idx,
    map_sorted_indices_to_group_slice,
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

/// Preserve logical ordering information, including inside nested columns
fn to_sort_repr(c: &Column) -> Column {
    let dtype = c.dtype();
    if dtype.is_nested() || dtype.contains_categoricals() || dtype.contains_enums() {
        c.clone()
    } else {
        c.to_physical_repr()
    }
}

static ERR_MSG: &str = "expressions in 'sort_by' must have matching group lengths";

fn check_groups(groups_in: &GroupsType, groups_by: &GroupsType) -> PolarsResult<()> {
    if let Some((g_in, g_by)) = groups_in
        .iter()
        .zip(groups_by.iter())
        .find(|(g_in, g_by)| g_in.len() != g_by.len())
    {
        polars_bail!(
            ShapeMismatch: "{ERR_MSG} (got a group of length {} to sort but length {} in `by`)",
            g_in.len(), g_by.len()
        );
    }
    Ok(())
}

pub(super) fn update_groups_sort_by(
    groups: &GroupsType,
    sort_by_s: &Series,
    options: &SortOptions,
) -> PolarsResult<GroupsType> {
    // Will trigger a gather for every group, so rechunk before.
    let sort_by_s = sort_by_s.rechunk();
    let groups = RAYON.install(|| {
        groups
            .par_iter()
            .map(|indicator| sort_by_groups_single_by(indicator, &sort_by_s, options))
            .collect::<PolarsResult<_>>()
    })?;

    Ok(GroupsType::Idx(groups))
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

    let first = new_idx.first().unwrap_or(&0);
    Ok((*first, new_idx))
}

fn sort_by_groups_no_match<'a>(
    mut ac_in: AggregationContext<'a>,
    mut ac_sort_by: Vec<AggregationContext<'a>>,
    options: SortMultipleOptions,
    expr: &Expr,
) -> PolarsResult<AggregationContext<'a>> {
    // Sorting a single value, which the group length checks guarantee, leaves it unchanged.
    if matches!(ac_in.state, AggState::AggregatedScalar(_)) {
        return Ok(ac_in);
    }
    let s_in = ac_in.aggregated();
    let mut s_in = s_in.list().unwrap().clone();
    let s_sort_by = ac_sort_by
        .iter_mut()
        .map(|ac| ac.aggregated_as_list().into_owned())
        .collect::<Vec<_>>();

    let dtype = s_in.dtype().clone();
    let ca: PolarsResult<ListChunked> = RAYON.install(|| {
        s_in.par_iter_indexed()
            .enumerate()
            .map(|(idx, opt_s)| {
                let s_sort_by = s_sort_by
                    .iter()
                    .map(|s| s.get_as_series(idx))
                    .collect::<Option<Vec<_>>>();

                match (opt_s, s_sort_by) {
                    (Some(s), Some(s_sort_by)) => {
                        if let Some(mismatch) =
                            s_sort_by.iter().find(|s_sort_by| s_sort_by.len() != s.len())
                        {
                            polars_bail!(
                                ComputeError: "series lengths don't match in 'sort_by' expression: the series to sort has length {} but a `by` series has length {}",
                                s.len(), mismatch.len()
                            );
                        }
                        let columns = s_sort_by
                            .iter()
                            .cloned()
                            .map(Column::from)
                            .collect::<Vec<_>>();
                        let idx = arg_sort(
                            &columns,
                            SortMultipleOptions {
                                // We are already in par iter.
                                multithreaded: false,
                                ..options.clone()
                            },
                        )?;
                        Ok(Some(unsafe { s.take_unchecked(&idx) }))
                    },
                    _ => Ok(None),
                }
            })
            .collect_ca_with_dtype(PlSmallStr::EMPTY, dtype)
    });
    let c = ca?.with_name(s_in.name().clone()).into_column();
    ac_in.with_values(c, true, Some(expr))?;
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
                .map(Column::from)
                .collect::<Vec<_>>();

            let options = SortMultipleOptions {
                descending: descending.to_owned(),
                nulls_last: nulls_last.to_owned(),
                multithreaded,
                maintain_order,
                limit: None,
            };

            let sorted_idx = arg_sort(&groups, options)?;
            map_sorted_indices_to_group_idx(&sorted_idx, idx)
        },
        GroupsIndicator::Slice([first, len]) => {
            let groups = sort_by_s
                .iter()
                .map(|s| s.slice(first as i64, len as usize))
                .map(Column::from)
                .collect::<Vec<_>>();

            let options = SortMultipleOptions {
                descending: descending.to_owned(),
                nulls_last: nulls_last.to_owned(),
                multithreaded,
                maintain_order,
                limit: None,
            };
            let sorted_idx = arg_sort(&groups, options)?;
            map_sorted_indices_to_group_slice(&sorted_idx, first)
        },
    };
    let first = new_idx
        .first()
        .ok_or_else(|| polars_err!(ComputeError: "{ERR_MSG}"))?;

    Ok((*first, new_idx))
}

impl PhysicalExpr for SortByExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate_impl(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let series_f = || self.input.evaluate(df, state);
        if self.by.iter().all(|e| e.is_scalar()) {
            // Constant keys leave the input unchanged.
            for e in &self.by {
                e.evaluate(df, state)?;
            }
            return series_f();
        }
        let (series, sorted_idx) = if self.by.len() == 1 {
            let sorted_idx_f = || {
                let s_sort_by = self.by[0].evaluate(df, state)?;
                Ok(s_sort_by.arg_sort(SortOptions::from(&self.sort_options)))
            };
            RAYON.install(|| rayon::join(series_f, sorted_idx_f))
        } else {
            let descending = prepare_bool_vec(&self.sort_options.descending, self.by.len());
            let nulls_last = prepare_bool_vec(&self.sort_options.nulls_last, self.by.len());

            let sorted_idx_f = || {
                let mut s_sort_by = self
                    .by
                    .iter()
                    .map(|e| e.evaluate(df, state).map(|c| to_sort_repr(&c)))
                    .collect::<PolarsResult<Vec<_>>>()?;

                let broadcast_length = broadcast_len(s_sort_by.iter())
                    .context("`sort_by` produced Series of differing lengths in `by`")?;
                for (e, c) in self.by.iter().zip(s_sort_by.iter_mut()) {
                    if c.len() != broadcast_length {
                        polars_ensure!(
                            e.is_scalar(),
                            ShapeMismatch: "non-scalar expression produces broadcasting column",
                        );
                        c.broadcast_in_place_to(broadcast_length)?;
                    }
                }

                let options = self
                    .sort_options
                    .clone()
                    .with_order_descending_multi(descending)
                    .with_nulls_last_multi(nulls_last);

                arg_sort(&s_sort_by, options)
            };
            RAYON.install(|| rayon::join(series_f, sorted_idx_f))
        };
        let (sorted_idx, series) = (sorted_idx?, series?);
        polars_ensure!(
            sorted_idx.len() == series.len(),
            expr = self.expr, ShapeMismatch:
            "`sort_by` produced different length ({}) than the Series that has to be sorted ({})",
            sorted_idx.len(), series.len()
        );

        // SAFETY: sorted index are within bounds.
        unsafe { Ok(series.take_unchecked(&sorted_idx)) }
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups_impl<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
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

        assert!(
            ac_sort_by
                .iter()
                .all(|ac_sort_by| ac_sort_by.groups.len() == ac_in.groups.len())
        );

        // Constant keys leave the input unchanged, and a literal input stays a literal.
        if matches!(ac_in.state, AggState::LiteralScalar(_))
            || self.by.iter().all(|e| e.is_scalar())
        {
            return Ok(ac_in);
        }

        // Enable reliable length checks downstream
        ac_in.set_groups_for_undefined_agg_states();
        ac_sort_by
            .iter_mut()
            .for_each(|ac| ac.set_groups_for_undefined_agg_states());

        for (e, ac) in self.by.iter().zip(ac_sort_by.iter_mut()) {
            if e.is_scalar() && ac.broadcast_unit_groups_to(&mut ac_in) {
                ac.normalize_values();
            }
        }

        // The physical positions of independently evaluated expressions can
        // differ even when every group has the same length. In that case, sort
        // their materialized logical group values instead of applying a
        // permutation expressed in another expression's physical positions.
        let groups_match = matches!(ac_in.update_groups, UpdateGroups::No)
            && ac_sort_by.iter().all(|ac| {
                matches!(ac.update_groups, UpdateGroups::No)
                    && (ac_in.groups.is_same(&ac.groups)
                        || ac_in.groups.as_ref().as_ref() == ac.groups.as_ref().as_ref())
            });
        if !groups_match {
            let groups_in = ac_in.groups().clone();
            for ac in &mut ac_sort_by {
                let groups = ac.groups();
                check_groups(groups_in.as_ref().as_ref(), groups.as_ref().as_ref())?;
            }
            return sort_by_groups_no_match(
                ac_in,
                ac_sort_by,
                self.sort_options.clone(),
                &self.expr,
            );
        }

        let mut sort_by_s = ac_sort_by
            .iter()
            // @scalar-opt
            // @partition-opt
            .map(|c| to_sort_repr(&c.flat_naive()).take_materialized_series())
            .collect::<Vec<_>>();

        let ordered_by_group_operation = matches!(
            ac_sort_by[0].update_groups,
            UpdateGroups::WithSeriesLen | UpdateGroups::WithGroupsLen
        );

        let groups = if ac_sort_by.len() == 1 {
            let mut ac_sort_by = ac_sort_by.pop().unwrap();

            let sort_by_s = sort_by_s.pop().unwrap();
            let groups = ac_sort_by.groups();

            let (check, groups) = RAYON.join(
                || check_groups(ac_in.groups(), groups),
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
            let groups_in = ac_in.groups();
            for ac in ac_sort_by.iter() {
                check_groups(groups_in.as_ref().as_ref(), ac.groups.as_ref().as_ref())?;
            }

            let groups = ac_sort_by[0].groups();

            let groups = RAYON.install(|| {
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
            GroupsType::Idx(groups?)
        };

        // If the rhs is already aggregated once, it is reordered by the
        // group_by operation - we must ensure that we are as well.
        if ordered_by_group_operation {
            let s = ac_in.aggregated();
            ac_in.with_values(
                s.explode(ExplodeOptions {
                    empty_as_null: true,
                    keep_nulls: true,
                })
                .unwrap(),
                false,
                None,
            )?;
        }

        ac_in.with_groups(groups.into_sliceable());
        Ok(ac_in)
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.input.to_field(input_schema)
    }

    fn is_scalar(&self) -> bool {
        self.input.is_scalar()
    }
}
