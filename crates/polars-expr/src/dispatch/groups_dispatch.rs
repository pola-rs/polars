use std::sync::Arc;

use arrow::bitmap::bitmask::BitMask;
use polars_core::POOL;
use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::prelude::{ChunkCast, Column, GroupPositions, GroupsType, IDX_DTYPE, IntoColumn};
use polars_core::series::ChunkCompareEq;
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::prelude::{AggState, AggregationContext, PhysicalExpr};
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
