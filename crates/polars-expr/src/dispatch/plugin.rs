use std::borrow::Cow;
use std::sync::Arc;

use arrow::array::{FixedSizeListArray, ListArray, PrimitiveArray};
use arrow::datatypes::ArrowDataType;
use arrow::offset::OffsetsBuffer;
use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::prelude::{
    Column, CompatLevel, GroupPositions, GroupsIdx, GroupsType, IDX_DTYPE, IdxItem, IntoColumn,
};
use polars_core::schema::Schema;
use polars_core::series::Series;
use polars_plan::dsl::v1::{PluginV1, PluginV1Flags};
use polars_utils::{IdxSize, UnitVec};

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

            let idx_dtype = IDX_DTYPE.to_arrow(CompatLevel::newest());
            out.groups();
            let groups = match &**out.groups.as_ref() {
                _ if matches!(out.agg_state(), AggState::AggregatedScalar(_)) => {
                    // @TODO: Make a shared kind
                    let values = (0..groups.len() * 2)
                        .map(|i| (i as IdxSize) % 2)
                        .collect::<Vec<_>>();
                    let values =
                        PrimitiveArray::<IdxSize>::new(idx_dtype.clone(), values.into(), None);
                    let dtype = idx_dtype.to_fixed_size_list(2, false);
                    Some(
                        FixedSizeListArray::try_new(dtype, groups.len(), values.boxed(), None)
                            .unwrap()
                            .boxed(),
                    )
                },
                _ if matches!(out.agg_state(), AggState::AggregatedScalar(_)) => None,
                GroupsType::Idx(idxs) => {
                    let mut offset = 0;
                    let mut offsets = Vec::with_capacity(idxs.all().len() + 1);
                    offsets.push(0);
                    offsets.extend(idxs.all().iter().map(|i| {
                        offset += i.len() as i64;
                        offset
                    }));
                    let mut values = Vec::with_capacity(*offsets.last().unwrap() as usize);
                    for i in idxs.all() {
                        values.extend_from_slice(i);
                    }

                    let values =
                        PrimitiveArray::<IdxSize>::new(idx_dtype.clone(), values.into(), None);
                    let offsets = unsafe { OffsetsBuffer::new_unchecked(offsets.into()) };

                    let dtype = idx_dtype.to_large_list(false);
                    Some(
                        ListArray::try_new(dtype, offsets, values.boxed(), None)
                            .unwrap()
                            .boxed(),
                    )
                },
                GroupsType::Slice {
                    groups,
                    overlapping: _,
                } => {
                    let mut values = Vec::with_capacity(groups.len() * 2);
                    for i in groups {
                        values.extend_from_slice(i);
                    }

                    let values =
                        PrimitiveArray::<IdxSize>::new(idx_dtype.clone(), values.into(), None);

                    let dtype = idx_dtype.to_fixed_size_list(2, false);
                    Some(
                        FixedSizeListArray::try_new(dtype, groups.len(), values.boxed(), None)
                            .unwrap()
                            .boxed(),
                    )
                },
            };
            let data = out.flat_naive().as_materialized_series().clone();

            Ok((data, groups))
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    let returns_scalar =
        plugin.flags().returns_scalar() || (all_scalar && plugin.flags().is_length_preserving());
    let (series, out_groups) = plugin.evaluate_on_groups(&input_data)?;

    match out_groups {
        None if !returns_scalar => Ok(AggregationContext {
            state: AggState::AggregatedList(series.as_list().into_column()),
            groups: Cow::Borrowed(groups),
            update_groups: UpdateGroups::WithSeriesLen,
            original_len: false,
        }),
        None => Ok(AggregationContext {
            state: AggState::AggregatedScalar(series.into_column()),
            groups: Cow::Borrowed(groups),
            update_groups: UpdateGroups::No,
            original_len: false,
        }),

        Some(groups) => {
            fn from_indices<E: std::fmt::Debug, T: Default + Copy + TryInto<IdxSize, Error = E>>(
                offsets: &OffsetsBuffer<i64>,
                values: &[T],
            ) -> GroupsIdx {
                assert_eq!(*offsets.first(), 0);
                assert_eq!(*offsets.last(), values.len() as i64);
                assert!(
                    values
                        .iter()
                        .all(|v| (*v).try_into().is_ok_and(|v| v < IdxSize::MAX))
                );

                offsets
                    .offset_and_length_iter()
                    .map(|(offset, length)| {
                        (
                            values
                                .get(offset)
                                .copied()
                                .unwrap_or_default()
                                .try_into()
                                .unwrap(),
                            UnitVec::from_iter(
                                values[offset..][..length]
                                    .iter()
                                    .copied()
                                    .map(|v| v.try_into().unwrap()),
                            ),
                        )
                    })
                    .collect::<Vec<IdxItem>>()
                    .into()
            }

            fn from_slices<E: std::fmt::Debug, T: Copy + TryInto<IdxSize, Error = E>>(
                values: &[T],
            ) -> Vec<[IdxSize; 2]> {
                assert!(
                    values
                        .iter()
                        .all(|v| (*v).try_into().is_ok_and(|v| v < IdxSize::MAX))
                );

                assert_eq!(values.len() % 2, 0);
                let mut out = Vec::with_capacity(values.len() / 2);
                for chunk in values.chunks(2) {
                    let offset = chunk[0].try_into().unwrap();
                    let length = chunk[1].try_into().unwrap();

                    assert!(offset.checked_add(length).unwrap() < IdxSize::MAX);

                    out.push([offset, length]);
                }
                out
            }

            let groups: GroupsType = match groups.dtype() {
                ArrowDataType::LargeList(f) if matches!(f.dtype(), ArrowDataType::UInt32) => {
                    let groups = groups.as_any().downcast_ref::<ListArray<i64>>().unwrap();
                    assert!(groups.validity().is_none());
                    let values = groups
                        .values()
                        .as_any()
                        .downcast_ref::<PrimitiveArray<u32>>()
                        .unwrap();
                    assert!(values.validity().is_none());
                    GroupsType::Idx(from_indices(groups.offsets(), values.values().as_slice()))
                },
                ArrowDataType::LargeList(f) if matches!(f.dtype(), ArrowDataType::UInt64) => {
                    let groups = groups.as_any().downcast_ref::<ListArray<i64>>().unwrap();
                    assert!(groups.validity().is_none());
                    let values = groups
                        .values()
                        .as_any()
                        .downcast_ref::<PrimitiveArray<u64>>()
                        .unwrap();
                    assert!(values.validity().is_none());
                    GroupsType::Idx(from_indices(groups.offsets(), values.values().as_slice()))
                },
                ArrowDataType::FixedSizeList(f, 2)
                    if matches!(f.dtype(), ArrowDataType::UInt32) =>
                {
                    let groups = groups
                        .as_any()
                        .downcast_ref::<FixedSizeListArray>()
                        .unwrap();
                    assert!(groups.validity().is_none());
                    assert_eq!(groups.size(), 2);
                    let values = groups
                        .values()
                        .as_any()
                        .downcast_ref::<PrimitiveArray<u32>>()
                        .unwrap();
                    assert!(values.validity().is_none());
                    GroupsType::Slice {
                        groups: from_slices(values.values().as_slice()),
                        overlapping: true,
                    }
                },
                ArrowDataType::FixedSizeList(f, 2)
                    if matches!(f.dtype(), ArrowDataType::UInt64) =>
                {
                    let groups = groups
                        .as_any()
                        .downcast_ref::<FixedSizeListArray>()
                        .unwrap();
                    assert!(groups.validity().is_none());
                    assert_eq!(groups.size(), 2);
                    let values = groups
                        .values()
                        .as_any()
                        .downcast_ref::<PrimitiveArray<u64>>()
                        .unwrap();
                    assert!(values.validity().is_none());
                    GroupsType::Slice {
                        groups: from_slices(values.values().as_slice()),
                        overlapping: true,
                    }
                },
                _ => panic!("invalid groups type"),
            };
            let groups = groups.into_sliceable();

            if returns_scalar {
                dbg!("todo!");
                todo!();
            }

            Ok(AggregationContext {
                state: AggState::NotAggregated(series.into_column()),
                groups: Cow::Owned(groups),
                update_groups: UpdateGroups::No,
                original_len: false,
            })
        },
    }
}
