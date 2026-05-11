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
                        let offsets = arr.offsets().as_slice();

                        let mut previous = 0i64;
                        let counts: NoNull<IdxCa> = offsets[1..]
                            .iter()
                            .map(|&o| {
                                let len = (o - previous) as IdxSize;
                                previous = o;
                                len
                            })
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
                ca.into_iter()
                    .map(|opt_s| opt_s.map(|s| s.len() as IdxSize - s.null_count() as IdxSize))
                    .collect::<IdxCa>()
                    .into_column()
            },
            AggState::NotAggregated(s) => {
                let s = s.clone();
                let groups = ac.groups();
                let out: IdxCa = if matches!(s.dtype(), &DataType::Null) {
                    IdxCa::full(s.name().clone(), 0, groups.len())
                } else {
                    match groups.as_ref().as_ref() {
                        GroupsType::Idx(idx) => {
                            let s = s.rechunk();
                            // @scalar-opt
                            // @partition-opt
                            let array = &s.as_materialized_series().chunks()[0];
                            let validity = array.validity().unwrap();
                            idx.iter()
                                .map(|(_, g)| {
                                    let mut count = 0 as IdxSize;
                                    // Count valid values
                                    g.iter().for_each(|i| unsafe {
                                        count += validity.get_bit_unchecked(*i as usize) as IdxSize;
                                    });
                                    count
                                })
                                .collect_ca_trusted_with_dtype(PlSmallStr::EMPTY, IDX_DTYPE)
                        },
                        GroupsType::Slice { groups, .. } => {
                            // Slice and use computed null count
                            groups
                                .iter()
                                .map(|g| {
                                    let start = g[0];
                                    let len = g[1];
                                    len - s.slice(start as i64, len as usize).null_count()
                                        as IdxSize
                                })
                                .collect_ca_trusted_with_dtype(PlSmallStr::EMPTY, IDX_DTYPE)
                        },
                    }
                };
                out.into_column()
            },
        }
    };
    Ok(out)
}
