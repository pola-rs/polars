use std::borrow::Cow;
use std::sync::Arc;

use arrow::bitmap::Bitmap;
use arrow::bitmap::bitmask::BitMask;
use arrow::trusted_len::TrustMyLength;
use polars_core::POOL;
use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::prelude::{
    AnyValue, ChunkCast, Column, CompatLevel, GroupPositions, GroupsType, IDX_DTYPE, IntoColumn,
};
use polars_core::scalar::Scalar;
use polars_core::series::{ChunkCompareEq, Series};
use polars_utils::itertools::Itertools;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::{IdxSize, UnitVec};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::prelude::{AggState, AggregationContext, PhysicalExpr, UpdateGroups};
use crate::state::ExecutionState;

pub fn reverse<'a>(
    inputs: &[Arc<dyn PhysicalExpr>],
    df: &DataFrame,
    groups: &'a GroupPositions,
    state: &ExecutionState,
) -> PolarsResult<AggregationContext<'a>> {
    assert_eq!(inputs.len(), 1);

    let mut ac = inputs[0].evaluate_on_groups(df, groups, state)?;

    // Length preserving operation on scalars keeps scalar.
    if let AggState::AggregatedScalar(_) | AggState::LiteralScalar(_) = &ac.agg_state() {
        return Ok(ac);
    }

    POOL.install(|| {
        let positions = GroupsType::Idx(match &**ac.groups().as_ref() {
            GroupsType::Idx(idx) => idx
                .into_par_iter()
                .map(|(first, idx)| {
                    (
                        idx.last().copied().unwrap_or(first),
                        idx.iter().copied().rev().collect(),
                    )
                })
                .collect(),
            GroupsType::Slice {
                groups,
                overlapping: _,
            } => groups
                .into_par_iter()
                .map(|[start, len]| {
                    (
                        start + len.saturating_sub(1),
                        (*start..*start + *len).rev().collect(),
                    )
                })
                .collect(),
        })
        .into_sliceable();
        ac.with_groups(positions);
    });

    Ok(ac)
}

pub fn null_count<'a>(
    inputs: &[Arc<dyn PhysicalExpr>],
    df: &DataFrame,
    groups: &'a GroupPositions,
    state: &ExecutionState,
) -> PolarsResult<AggregationContext<'a>> {
    assert_eq!(inputs.len(), 1);

    let mut ac = inputs[0].evaluate_on_groups(df, groups, state)?;

    if let AggState::AggregatedScalar(s) | AggState::LiteralScalar(s) = &mut ac.state {
        *s = s.is_null().cast(&IDX_DTYPE).unwrap().into_column();
        return Ok(ac);
    }

    ac.groups();
    let values = ac.flat_naive();
    let name = values.name().clone();
    let Some(validity) = values.rechunk_validity() else {
        ac.state = AggState::AggregatedScalar(Column::new_scalar(
            name,
            (0 as IdxSize).into(),
            groups.len(),
        ));
        return Ok(ac);
    };

    POOL.install(|| {
        let validity = BitMask::from_bitmap(&validity);
        let null_count: Vec<IdxSize> = match &**ac.groups.as_ref() {
            GroupsType::Idx(idx) => idx
                .into_par_iter()
                .map(|(_, idx)| {
                    idx.iter()
                        .map(|i| IdxSize::from(!unsafe { validity.get_bit_unchecked(*i as usize) }))
                        .sum::<IdxSize>()
                })
                .collect(),
            GroupsType::Slice {
                groups,
                overlapping: _,
            } => groups
                .into_par_iter()
                .map(|[start, length]| {
                    unsafe { validity.sliced_unchecked(*start as usize, *length as usize) }
                        .unset_bits() as IdxSize
                })
                .collect(),
        };

        ac.state = AggState::AggregatedScalar(Column::new(name, null_count));
    });

    Ok(ac)
}

pub fn any<'a>(
    inputs: &[Arc<dyn PhysicalExpr>],
    df: &DataFrame,
    groups: &'a GroupPositions,
    state: &ExecutionState,
    ignore_nulls: bool,
) -> PolarsResult<AggregationContext<'a>> {
    assert_eq!(inputs.len(), 1);

    let mut ac = inputs[0].evaluate_on_groups(df, groups, state)?;

    if let AggState::AggregatedScalar(s) | AggState::LiteralScalar(s) = &mut ac.state {
        if ignore_nulls {
            *s = s
                .equal_missing(&Column::new_scalar(PlSmallStr::EMPTY, true.into(), 1))
                .unwrap()
                .into_column();
        } else {
            *s = s
                .equal(&Column::new_scalar(PlSmallStr::EMPTY, true.into(), 1))
                .unwrap()
                .into_column();
        }
        return Ok(ac);
    }

    ac.groups();
    let values = ac.flat_naive();
    let values = values.bool()?;
    let out = unsafe { values.agg_any(ac.groups.as_ref(), ignore_nulls) };
    ac.state = AggState::AggregatedScalar(out.into_column());

    Ok(ac)
}

pub fn all<'a>(
    inputs: &[Arc<dyn PhysicalExpr>],
    df: &DataFrame,
    groups: &'a GroupPositions,
    state: &ExecutionState,
    ignore_nulls: bool,
) -> PolarsResult<AggregationContext<'a>> {
    assert_eq!(inputs.len(), 1);

    let mut ac = inputs[0].evaluate_on_groups(df, groups, state)?;

    if let AggState::AggregatedScalar(s) | AggState::LiteralScalar(s) = &mut ac.state {
        if ignore_nulls {
            *s = s
                .equal_missing(&Column::new_scalar(PlSmallStr::EMPTY, true.into(), 1))
                .unwrap()
                .into_column();
        } else {
            *s = s
                .equal(&Column::new_scalar(PlSmallStr::EMPTY, true.into(), 1))
                .unwrap()
                .into_column();
        }
        return Ok(ac);
    }

    ac.groups();
    let values = ac.flat_naive();
    let values = values.bool()?;
    let out = unsafe { values.agg_all(groups.as_ref(), ignore_nulls) };
    ac.state = AggState::AggregatedScalar(out.into_column());

    Ok(ac)
}

pub fn drop_items<'a>(
    mut ac: AggregationContext<'a>,
    predicate: &Bitmap,
) -> PolarsResult<AggregationContext<'a>> {
    // No elements are filtered out.
    if predicate.unset_bits() == 0 {
        if let AggState::AggregatedScalar(c) | AggState::LiteralScalar(c) = &mut ac.state {
            *c = c.as_list().into_column();
        }
        return Ok(ac);
    }

    ac.set_original_len(false);

    // All elements are filtered out.
    if predicate.set_bits() == 0 {
        let name = ac.agg_state().name();
        let dtype = ac.agg_state().flat_dtype();

        ac.state = AggState::AggregatedList(Column::new_scalar(
            name.clone(),
            Scalar::new(
                dtype.clone().implode(),
                AnyValue::List(Series::new_empty(PlSmallStr::EMPTY, dtype)),
            ),
            ac.groups.len(),
        ));
        ac.with_update_groups(UpdateGroups::WithSeriesLen);
        return Ok(ac);
    }

    if let AggState::LiteralScalar(c) = &ac.state {
        ac.state =
            AggState::AggregatedList(c.as_list().into_column().new_from_index(0, predicate.len()));
        ac.groups = Cow::Owned(
            GroupsType::Slice {
                groups: predicate.iter().map(|p| [0, IdxSize::from(p)]).collect(),
                overlapping: true,
            }
            .into_sliceable(),
        );
        return Ok(ac);
    }

    if let AggState::AggregatedScalar(c) = &mut ac.state {
        ac.state = AggState::AggregatedList(c.as_list().into_column());
        ac.groups = Cow::Owned(
            GroupsType::Slice {
                groups: predicate
                    .iter()
                    .enumerate_idx()
                    .map(|(i, p)| [i, IdxSize::from(p)])
                    .collect(),
                overlapping: false,
            }
            .into_sliceable(),
        );
        return Ok(ac);
    }

    ac.groups();
    let predicate = BitMask::from_bitmap(predicate);
    POOL.install(|| {
        let positions = GroupsType::Idx(match &**ac.groups.as_ref() {
            GroupsType::Idx(idxs) => idxs
                .into_par_iter()
                .map(|(fst, idxs)| {
                    let out = idxs
                        .iter()
                        .copied()
                        .filter(|i| unsafe { predicate.get_bit_unchecked(*i as usize) })
                        .collect::<UnitVec<IdxSize>>();
                    (out.first().copied().unwrap_or(fst), out)
                })
                .collect(),
            GroupsType::Slice {
                groups,
                overlapping: _,
            } => groups
                .into_par_iter()
                .map(|[start, length]| {
                    let predicate =
                        unsafe { predicate.sliced_unchecked(*start as usize, *length as usize) };
                    let num_values = predicate.set_bits();

                    if num_values == 0 {
                        (*start, UnitVec::new())
                    } else if num_values == 1 {
                        let item = *start + predicate.leading_zeros() as IdxSize;
                        let mut out = UnitVec::with_capacity(1);
                        out.push(item);
                        (item, out)
                    } else if num_values == *length as usize {
                        (*start, (*start..*start + *length).collect())
                    } else {
                        let out = unsafe {
                            TrustMyLength::new(
                                (0..*length)
                                    .filter(|i| predicate.get_bit_unchecked(*i as usize))
                                    .map(|i| i + *start),
                                num_values,
                            )
                        };
                        let out = out.collect::<UnitVec<IdxSize>>();

                        (out.first().copied().unwrap(), out)
                    }
                })
                .collect(),
        })
        .into_sliceable();
        ac.with_groups(positions);
    });

    Ok(ac)
}

pub fn drop_nans<'a>(
    inputs: &[Arc<dyn PhysicalExpr>],
    df: &DataFrame,
    groups: &'a GroupPositions,
    state: &ExecutionState,
) -> PolarsResult<AggregationContext<'a>> {
    assert_eq!(inputs.len(), 1);
    let mut ac = inputs[0].evaluate_on_groups(df, groups, state)?;
    ac.groups();
    let predicate = if ac.agg_state().flat_dtype().is_float() {
        let values = ac.flat_naive();
        let mut values = values.is_nan().unwrap();
        values.rechunk_mut();
        values.downcast_as_array().values().clone()
    } else {
        Bitmap::new_with_value(true, 1)
    };
    drop_items(ac, &predicate)
}

pub fn drop_nulls<'a>(
    inputs: &[Arc<dyn PhysicalExpr>],
    df: &DataFrame,
    groups: &'a GroupPositions,
    state: &ExecutionState,
) -> PolarsResult<AggregationContext<'a>> {
    assert_eq!(inputs.len(), 1);
    let mut ac = inputs[0].evaluate_on_groups(df, groups, state)?;
    ac.groups();
    let predicate = ac.flat_naive().as_ref().clone();
    let predicate = predicate.rechunk_to_arrow(CompatLevel::newest());
    let predicate = predicate
        .validity()
        .cloned()
        .unwrap_or(Bitmap::new_with_value(true, 1));
    drop_items(ac, &predicate)
}
