use polars_core::prelude::*;
use polars_core::utils::{CustomIterTools, NoNull};

use crate::expressions::{AggState, AggregationContext, UpdateGroups};

pub fn evaluate_count_on_ac<'a>(
    mut ac: AggregationContext<'a>,
    include_nulls: bool,
) -> PolarsResult<Column> {
    let values_have_no_nulls = match ac.agg_state() {
        AggState::AggregatedList(s) => {
            let list = s.list()?;
            list.null_count() == 0
                && list
                    .downcast_iter()
                    .all(|arr| arr.values().null_count() == 0)
        },
        _ => ac.get_values().null_count() == 0,
    };

    let out = if include_nulls || values_have_no_nulls {
        // a few fast paths that prevent materializing new groups
        match ac.update_groups {
            UpdateGroups::WithSeriesLen => {
                let list = ac
                    .get_values()
                    .list()
                    .expect("impl error, should be a list at this point");

                let s = match list.chunks().len() {
                    1 => {
                        let arr = list.downcast_iter().next().unwrap();
                        let counts: NoNull<IdxCa> = (0..arr.len())
                            .map(|i| arr.value_length(i) as IdxSize)
                            .collect_trusted();
                        counts.into_inner()
                    },
                    _ => {
                        let counts: NoNull<IdxCa> = list
                            .amortized_iter()
                            .map(|s| {
                                if let Some(s) = s {
                                    s.as_ref().len() as IdxSize
                                } else {
                                    1
                                }
                            })
                            .collect_trusted();
                        counts.into_inner()
                    },
                };
                s.into_column()
            },
            UpdateGroups::WithGroupsLen => {
                // no need to update the groups
                // we can just get the attribute, because we only need the length,
                // not the correct order
                ac.groups.group_count().into_column()
            },
            // materialize groups
            _ => ac.groups().group_count().into_column(),
        }
    } else {
        // TODO: optimize this/and write somewhere else.
        match ac.agg_state() {
            AggState::LiteralScalar(_) => unreachable!(),
            AggState::AggregatedScalar(c) => {
                c.is_not_null().cast(&IDX_DTYPE).unwrap().into_column()
            },
            AggState::AggregatedList(s) => {
                let ca = s.list()?;
                ca.series_iter()
                    .map(|opt_s| opt_s.map(|s| s.len() as IdxSize - s.null_count() as IdxSize))
                    .collect::<IdxCa>()
                    .into_column()
            },
            AggState::NotAggregated(s) => {
                let s = s.clone();
                let groups = ac.groups();
                // Null-typed data has an all-invalid validity, so it counts as 0.
                // SAFETY: groups are always in bounds.
                unsafe { s.agg_valid_count(groups.as_ref().as_ref()) }
            },
        }
    };
    Ok(out)
}
