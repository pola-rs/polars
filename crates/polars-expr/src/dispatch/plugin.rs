use std::borrow::Cow;
use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::prelude::{Column, GroupPositions, GroupsType, IntoColumn};
use polars_core::schema::Schema;
use polars_core::series::Series;
use polars_ffi::version_1 as ffi;
use polars_plan::dsl::v1::{PluginV1, PluginV1Flags};
use polars_utils::IdxSize;

use crate::prelude::{AggState, AggregationContext, PhysicalExpr, UpdateGroups};
use crate::state::ExecutionState;

pub fn call(s: &mut [Column], plugin: Arc<PluginV1>) -> PolarsResult<Column> {
    let fields = Schema::from_iter(s.iter().map(|c| c.field().into_owned()));
    let series = s
        .iter_mut()
        .map(|c| std::mem::take(c).take_materialized_series())
        .collect::<Vec<_>>();

    let mut state = plugin.clone().initialize(&fields)?;

    let flags = plugin.flags();
    let insert = state.step(&series)?;

    assert!(insert.is_none() || flags.contains(PluginV1Flags::STEP_HAS_OUTPUT));

    if !flags.contains(PluginV1Flags::NEEDS_FINALIZE) || flags.is_elementwise() {
        let field = plugin.to_field(&fields)?;
        let out = insert.unwrap_or_else(|| Series::new_empty(field.name, &field.dtype));
        return Ok(out.into_column());
    }

    let finalize = state.finalize()?;
    Ok(match (insert, finalize) {
        (None, None) => {
            let field = plugin.to_field(&fields)?;
            Column::new_empty(field.name, &field.dtype)
        },
        (Some(s), None) | (None, Some(s)) => s.into_column(),
        (Some(mut s), Some(s2)) => {
            s.append_owned(s2)?;
            s.into_column()
        },
    })
}

pub fn call_on_groups<'a>(
    inputs: &[Arc<dyn PhysicalExpr>],
    df: &DataFrame,
    groups: &'a GroupPositions,
    state: &ExecutionState,
    plugin: Arc<PluginV1>,
) -> PolarsResult<AggregationContext<'a>> {
    let mut all_scalar = true;
    let input_data = inputs
        .iter()
        .map(|i| {
            let mut out = i.evaluate_on_groups(df, groups, state)?;
            all_scalar &= i.is_scalar();

            let groups = match &out.state {
                AggState::LiteralScalar(_) => ffi::GroupPositions::SharedAcrossGroups {
                    num_groups: groups.len(),
                },
                AggState::AggregatedScalar(_) => ffi::GroupPositions::ScalarPerGroup,
                AggState::NotAggregated(_) | AggState::AggregatedList(_) => {
                    out.groups();
                    match &**out.groups.as_ref() {
                        GroupsType::Idx(idxs) => {
                            let mut offset = 0;
                            let mut ends = Vec::with_capacity(idxs.all().len() + 1);
                            ends.extend(idxs.all().iter().map(|i| {
                                offset += i.len() as u64;
                                offset
                            }));
                            let mut index = Vec::with_capacity(*ends.last().unwrap() as usize);
                            for i in idxs.all() {
                                index.extend(i.iter().copied().map(|v| v as u64));
                            }

                            ffi::GroupPositions::Index(ffi::IndexGroups {
                                index: index.into(),
                                ends: ends.into(),
                            })
                        },
                        GroupsType::Slice {
                            groups,
                            overlapping: _,
                        } => {
                            let mut slices = Vec::with_capacity(groups.len());
                            for [offset, length] in groups {
                                slices.push(ffi::SliceGroup {
                                    offset: *offset as u64,
                                    length: *length as u64,
                                });
                            }
                            ffi::GroupPositions::Slice(ffi::SliceGroups(slices.into()))
                        },
                    }
                },
            };
            let data = out.flat_naive().as_materialized_series().clone();

            Ok((data, groups))
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    let input_data = input_data
        .iter()
        .map(|(s, g): &(Series, ffi::GroupPositions)| (s.clone(), g))
        .collect::<Vec<_>>();

    let returns_scalar =
        plugin.flags().returns_scalar() || (all_scalar && plugin.flags().is_length_preserving());
    let (series, out_groups) = plugin.evaluate_on_groups(&input_data)?;

    let num_groups = match out_groups.as_ref() {
        ffi::GroupPositions::SharedAcrossGroups { num_groups } => *num_groups,
        ffi::GroupPositions::ScalarPerGroup => series.len(),
        ffi::GroupPositions::Index(index) => index.ends.len(),
        ffi::GroupPositions::Slice(slices) => slices.len(),
    };
    assert_eq!(
        groups.len(),
        num_groups,
        "plugin implementation error: input groups is not equal to output groups",
    );

    match out_groups.as_ref() {
        ffi::GroupPositions::SharedAcrossGroups { num_groups: _ } => {
            if returns_scalar {
                assert_eq!(
                    series.len(),
                    1,
                    "plugin implementation error: set as returns scalar, but does not return scalar"
                );
                Ok(AggregationContext {
                    state: AggState::LiteralScalar(series.into_column()),
                    groups: Cow::Borrowed(groups),
                    update_groups: UpdateGroups::No,
                    original_len: false,
                })
            } else {
                Ok(AggregationContext {
                    state: AggState::AggregatedList(
                        series
                            .implode()?
                            .into_column()
                            .new_from_index(0, groups.len()),
                    ),
                    groups: Cow::Borrowed(groups),
                    update_groups: UpdateGroups::WithSeriesLen,
                    original_len: false,
                })
            }
        },
        ffi::GroupPositions::ScalarPerGroup => {
            assert_eq!(
                series.len(),
                groups.len(),
                "plugin implementation error: invalid number of return values."
            );
            if returns_scalar {
                Ok(AggregationContext {
                    state: AggState::AggregatedScalar(series.into_column()),
                    groups: Cow::Borrowed(groups),
                    update_groups: UpdateGroups::No,
                    original_len: false,
                })
            } else {
                Ok(AggregationContext {
                    state: AggState::AggregatedList(series.as_list().into_column()),
                    groups: Cow::Borrowed(groups),
                    update_groups: UpdateGroups::WithSeriesLen,
                    original_len: false,
                })
            }
        },
        ffi::GroupPositions::Index(index) => {
            if returns_scalar {
                assert!(
                    index.lengths().all(|l| l == 1),
                    "plugin implementation error: flagged as returns scalar, but num values != 1."
                );

                let indices = index
                    .index
                    .iter()
                    .map(|v| *v as IdxSize)
                    .collect::<Vec<_>>();
                let series = unsafe { series.take_slice_unchecked(&indices) };

                Ok(AggregationContext {
                    state: AggState::AggregatedScalar(series.into_column()),
                    groups: Cow::Borrowed(groups),
                    update_groups: UpdateGroups::No,
                    original_len: false,
                })
            } else {
                let groups = GroupsType::Idx(index.to_core()).into_sliceable();
                Ok(AggregationContext {
                    state: AggState::NotAggregated(series.into_column()),
                    groups: Cow::Owned(groups),
                    update_groups: UpdateGroups::No,
                    original_len: false,
                })
            }
        },
        ffi::GroupPositions::Slice(slices) => {
            if returns_scalar {
                assert!(
                    slices.lengths().all(|l| l == 1),
                    "plugin implementation error: flagged as returns scalar, but num values != 1."
                );

                let indices = slices
                    .0
                    .iter()
                    .map(|s| s.offset as IdxSize)
                    .collect::<Vec<_>>();
                let series = unsafe { series.take_slice_unchecked(&indices) };

                Ok(AggregationContext {
                    state: AggState::AggregatedScalar(series.into_column()),
                    groups: Cow::Borrowed(groups),
                    update_groups: UpdateGroups::No,
                    original_len: false,
                })
            } else {
                let groups = GroupsType::Slice {
                    groups: slices.to_core(),
                    overlapping: true,
                }
                .into_sliceable();

                Ok(AggregationContext {
                    state: AggState::NotAggregated(series.into_column()),
                    groups: Cow::Owned(groups),
                    update_groups: UpdateGroups::No,
                    original_len: false,
                })
            }
        },
    }
}
