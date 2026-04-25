use std::borrow::Cow;
use std::sync::Arc;

use arrow::array::PrimitiveArray;
use arrow::bitmap::Bitmap;
use arrow::bitmap::bitmask::BitMask;
use arrow::trusted_len::TrustMyLength;
use polars_compute::unique::{AmortizedUnique, amortized_unique_from_dtype};
use polars_core::POOL;
use polars_core::chunked_array::builder::get_list_builder;
use polars_core::chunked_array::from_iterator_par::try_list_from_par_iter;
use polars_core::error::{PolarsResult, polars_bail, polars_ensure, polars_err};
use polars_core::frame::DataFrame;
use polars_core::prelude::row_encode::encode_rows_unordered;
use polars_core::prelude::{
    AnyValue, ChunkCast, Column, CompatLevel, DataType, Float64Chunked, GroupPositions,
    GroupsType, IDX_DTYPE, IntoColumn, IntoSeries, ListChunked,
};
use polars_core::scalar::Scalar;
use polars_core::series::{ChunkCompareEq, Series};
use polars_utils::itertools::Itertools;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::{IdxSize, UnitVec};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

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
                monotonic: _,
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
                monotonic: _,
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
    let out = unsafe { values.agg_all(ac.groups.as_ref(), ignore_nulls) };
    ac.state = AggState::AggregatedScalar(out.into_column());

    Ok(ac)
}

#[cfg(feature = "bitwise")]
pub fn bitwise_agg<'a>(
    inputs: &[Arc<dyn PhysicalExpr>],
    df: &DataFrame,
    groups: &'a GroupPositions,
    state: &ExecutionState,
    op: &'static str,
    f: impl Fn(&Column, &GroupsType) -> Column,
) -> PolarsResult<AggregationContext<'a>> {
    assert_eq!(inputs.len(), 1);

    let mut ac = inputs[0].evaluate_on_groups(df, groups, state)?;

    if let AggState::AggregatedScalar(s) | AggState::LiteralScalar(s) = &ac.state {
        let dtype = s.dtype();
        polars_ensure!(
            dtype.is_bool() | dtype.is_primitive_numeric(),
            op = op,
            dtype
        );
        return Ok(ac);
    }

    ac.groups();
    let values = ac.flat_naive();
    let out = f(values.as_ref(), ac.groups.as_ref());
    ac.state = AggState::AggregatedScalar(out.into_column());

    Ok(ac)
}

#[cfg(feature = "bitwise")]
pub fn bitwise_and<'a>(
    inputs: &[Arc<dyn PhysicalExpr>],
    df: &DataFrame,
    groups: &'a GroupPositions,
    state: &ExecutionState,
) -> PolarsResult<AggregationContext<'a>> {
    bitwise_agg(
        inputs,
        df,
        groups,
        state,
        "and_reduce",
        |v, groups| unsafe { v.agg_and(groups) },
    )
}

#[cfg(feature = "bitwise")]
pub fn bitwise_or<'a>(
    inputs: &[Arc<dyn PhysicalExpr>],
    df: &DataFrame,
    groups: &'a GroupPositions,
    state: &ExecutionState,
) -> PolarsResult<AggregationContext<'a>> {
    bitwise_agg(inputs, df, groups, state, "or_reduce", |v, groups| unsafe {
        v.agg_or(groups)
    })
}

#[cfg(feature = "bitwise")]
pub fn bitwise_xor<'a>(
    inputs: &[Arc<dyn PhysicalExpr>],
    df: &DataFrame,
    groups: &'a GroupPositions,
    state: &ExecutionState,
) -> PolarsResult<AggregationContext<'a>> {
    bitwise_agg(
        inputs,
        df,
        groups,
        state,
        "xor_reduce",
        |v, groups| unsafe { v.agg_xor(groups) },
    )
}

pub fn drop_items<'a>(
    mut ac: AggregationContext<'a>,
    predicate: &Bitmap,
) -> PolarsResult<AggregationContext<'a>> {
    // No elements are filtered out.
    if predicate.unset_bits() == 0 {
        if let AggState::AggregatedScalar(c) | AggState::LiteralScalar(c) = &mut ac.state {
            *c = c.as_list().into_column();
            if c.len() == 1 && ac.groups.len() != 1 {
                *c = c.new_from_index(0, ac.groups.len());
            }
            ac.state = AggState::AggregatedList(std::mem::take(c));
            ac.update_groups = UpdateGroups::WithSeriesLen;
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

    if let AggState::AggregatedScalar(c) = &mut ac.state {
        ac.state = AggState::NotAggregated(std::mem::take(c));
        ac.groups = Cow::Owned(
            {
                let groups = predicate
                    .iter()
                    .enumerate_idx()
                    .map(|(i, p)| [i, IdxSize::from(p)])
                    .collect();
                GroupsType::new_slice(groups, false, true)
            }
            .into_sliceable(),
        );
        ac.update_groups = UpdateGroups::No;
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
                monotonic: _,
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
        Bitmap::new_with_value(false, 1)
    };
    let predicate = !&predicate;
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

#[cfg(feature = "moment")]
pub fn moment_agg<'a, S: Default>(
    inputs: &[Arc<dyn PhysicalExpr>],
    df: &DataFrame,
    groups: &'a GroupPositions,
    state: &ExecutionState,

    insert_one: impl Fn(&mut S, f64) + Send + Sync,
    new_from_slice: impl Fn(&PrimitiveArray<f64>, usize, usize) -> S + Send + Sync,
    finalize: impl Fn(S) -> Option<f64> + Send + Sync,
) -> PolarsResult<AggregationContext<'a>> {
    assert_eq!(inputs.len(), 1);
    let mut ac = inputs[0].evaluate_on_groups(df, groups, state)?;

    if let AggState::AggregatedScalar(s) | AggState::LiteralScalar(s) = &mut ac.state {
        let ca = s.f64()?;
        *s = ca
            .iter()
            .map(|v| {
                v.and_then(|v| {
                    let mut state = S::default();
                    insert_one(&mut state, v);
                    finalize(state)
                })
            })
            .collect::<Float64Chunked>()
            .with_name(ca.name().clone())
            .into_column();
        return Ok(ac);
    }

    ac.groups();

    let name = ac.get_values().name().clone();
    let ca = ac.flat_naive();
    let ca = ca.f64()?;
    let ca = ca.rechunk();
    let arr = ca.downcast_as_array();

    let ca = POOL.install(|| match &**ac.groups.as_ref() {
        GroupsType::Idx(idx) => {
            if let Some(validity) = arr.validity().filter(|v| v.unset_bits() > 0) {
                idx.into_par_iter()
                    .map(|(_, idx)| {
                        let mut state = S::default();
                        for &i in idx.iter() {
                            if unsafe { validity.get_bit_unchecked(i as usize) } {
                                insert_one(&mut state, arr.values()[i as usize]);
                            }
                        }
                        finalize(state)
                    })
                    .collect::<Float64Chunked>()
            } else {
                idx.into_par_iter()
                    .map(|(_, idx)| {
                        let mut state = S::default();
                        for &i in idx.iter() {
                            insert_one(&mut state, arr.values()[i as usize]);
                        }
                        finalize(state)
                    })
                    .collect::<Float64Chunked>()
            }
        },
        GroupsType::Slice {
            groups,
            overlapping: _,
            monotonic: _,
        } => groups
            .into_par_iter()
            .map(|[start, length]| finalize(new_from_slice(arr, *start as usize, *length as usize)))
            .collect::<Float64Chunked>(),
    });

    ac.state = AggState::AggregatedScalar(ca.with_name(name).into_column());
    Ok(ac)
}

#[cfg(feature = "moment")]
pub fn skew<'a>(
    inputs: &[Arc<dyn PhysicalExpr>],
    df: &DataFrame,
    groups: &'a GroupPositions,
    state: &ExecutionState,
    bias: bool,
) -> PolarsResult<AggregationContext<'a>> {
    use polars_compute::moment::SkewState;
    moment_agg::<SkewState>(
        inputs,
        df,
        groups,
        state,
        SkewState::insert_one,
        SkewState::from_array,
        |s| s.finalize(bias),
    )
}

#[cfg(feature = "moment")]
pub fn kurtosis<'a>(
    inputs: &[Arc<dyn PhysicalExpr>],
    df: &DataFrame,
    groups: &'a GroupPositions,
    state: &ExecutionState,
    fisher: bool,
    bias: bool,
) -> PolarsResult<AggregationContext<'a>> {
    use polars_compute::moment::KurtosisState;
    moment_agg::<KurtosisState>(
        inputs,
        df,
        groups,
        state,
        KurtosisState::insert_one,
        KurtosisState::from_array,
        |s| s.finalize(fisher, bias),
    )
}

pub fn unique<'a>(
    inputs: &[Arc<dyn PhysicalExpr>],
    df: &DataFrame,
    groups: &'a GroupPositions,
    state: &ExecutionState,
    stable: bool,
) -> PolarsResult<AggregationContext<'a>> {
    _ = stable;

    assert_eq!(inputs.len(), 1);
    let mut ac = inputs[0].evaluate_on_groups(df, groups, state)?;
    ac.groups();

    if let AggState::AggregatedScalar(c) | AggState::LiteralScalar(c) = &mut ac.state {
        *c = c.as_list().into_column();
        if c.len() == 1 && ac.groups.len() != 1 {
            *c = c.new_from_index(0, ac.groups.len());
        }
        ac.state = AggState::AggregatedList(std::mem::take(c));
        ac.update_groups = UpdateGroups::WithSeriesLen;
        return Ok(ac);
    }

    let values = ac.flat_naive().to_physical_repr();
    let dtype = values.dtype();
    let values = if dtype.contains_objects() {
        polars_bail!(opq = unique, dtype);
    } else if let Some(ca) = values.try_str() {
        ca.as_binary().into_column()
    } else if dtype.is_nested() {
        encode_rows_unordered(&[values])?.into_column()
    } else {
        values
    };

    let values = values.rechunk_to_arrow(CompatLevel::newest());
    let values = values.as_ref();
    let state = amortized_unique_from_dtype(values.dtype());

    struct CloneWrapper(Box<dyn AmortizedUnique>);
    impl Clone for CloneWrapper {
        fn clone(&self) -> Self {
            Self(self.0.new_empty())
        }
    }

    POOL.install(|| {
        let positions = GroupsType::Idx(match &**ac.groups().as_ref() {
            GroupsType::Idx(idx) => idx
                .into_par_iter()
                .map_with(CloneWrapper(state), |state, (first, idx)| {
                    let mut idx = idx.clone();
                    unsafe { state.0.retain_unique(values, &mut idx) };
                    (idx.first().copied().unwrap_or(first), idx)
                })
                .collect(),
            GroupsType::Slice {
                groups,
                overlapping: _,
                monotonic: _,
            } => groups
                .into_par_iter()
                .map_with(CloneWrapper(state), |state, [start, len]| {
                    let mut idx = UnitVec::new();
                    state.0.arg_unique(values, &mut idx, *start, *len);
                    (idx.first().copied().unwrap_or(*start), idx)
                })
                .collect(),
        })
        .into_sliceable();
        ac.with_groups(positions);
    });

    Ok(ac)
}

fn fw_bw_fill_null<'a>(
    inputs: &[Arc<dyn PhysicalExpr>],
    df: &DataFrame,
    groups: &'a GroupPositions,
    state: &ExecutionState,
    f_idx: impl Fn(
        std::iter::Copied<std::slice::Iter<'_, IdxSize>>,
        BitMask<'_>,
        usize,
    ) -> UnitVec<IdxSize>
    + Send
    + Sync,
    f_range: impl Fn(std::ops::Range<IdxSize>, BitMask<'_>, usize) -> UnitVec<IdxSize> + Send + Sync,
) -> PolarsResult<AggregationContext<'a>> {
    assert_eq!(inputs.len(), 1);
    let mut ac = inputs[0].evaluate_on_groups(df, groups, state)?;
    ac.groups();

    if let AggState::AggregatedScalar(_) | AggState::LiteralScalar(_) = &mut ac.state {
        return Ok(ac);
    }

    let values = ac.flat_naive();
    let Some(validity) = values.rechunk_validity() else {
        return Ok(ac);
    };

    let validity = BitMask::from_bitmap(&validity);
    POOL.install(|| {
        let positions = GroupsType::Idx(match &**ac.groups().as_ref() {
            GroupsType::Idx(idx) => idx
                .into_par_iter()
                .map(|(first, idx)| {
                    let idx = f_idx(idx.iter().copied(), validity, idx.len());
                    (idx.first().copied().unwrap_or(first), idx)
                })
                .collect(),
            GroupsType::Slice {
                groups,
                overlapping: _,
                monotonic: _,
            } => groups
                .into_par_iter()
                .map(|[start, len]| {
                    let idx = f_range(*start..*start + *len, validity, *len as usize);
                    (idx.first().copied().unwrap_or(*start), idx)
                })
                .collect(),
        })
        .into_sliceable();
        ac.with_groups(positions);
    });

    Ok(ac)
}

pub fn forward_fill_null<'a>(
    inputs: &[Arc<dyn PhysicalExpr>],
    df: &DataFrame,
    groups: &'a GroupPositions,
    state: &ExecutionState,
    limit: Option<IdxSize>,
) -> PolarsResult<AggregationContext<'a>> {
    let limit = limit.unwrap_or(IdxSize::MAX);
    macro_rules! arg_forward_fill {
        (
            $iter:ident,
            $validity:ident,
            $length:ident
        ) => {{
            |$iter, $validity, $length| {
                let Some(start) = $iter
                    .clone()
                    .position(|i| unsafe { $validity.get_bit_unchecked(i as usize) })
                else {
                    return $iter.collect();
                };

                let mut idx = UnitVec::with_capacity($length);
                let mut iter = $iter;
                idx.extend((&mut iter).take(start));

                let mut current_limit = limit;
                let mut value = iter.next().unwrap();
                idx.push(value);

                idx.extend(iter.map(|i| {
                    if unsafe { $validity.get_bit_unchecked(i as usize) } {
                        current_limit = limit;
                        value = i;
                        i
                    } else if current_limit == 0 {
                        i
                    } else {
                        current_limit -= 1;
                        value
                    }
                }));
                idx
            }
        }};
    }

    fw_bw_fill_null(
        inputs,
        df,
        groups,
        state,
        arg_forward_fill!(iter, validity, length),
        arg_forward_fill!(iter, validity, length),
    )
}

pub fn backward_fill_null<'a>(
    inputs: &[Arc<dyn PhysicalExpr>],
    df: &DataFrame,
    groups: &'a GroupPositions,
    state: &ExecutionState,
    limit: Option<IdxSize>,
) -> PolarsResult<AggregationContext<'a>> {
    let limit = limit.unwrap_or(IdxSize::MAX);
    macro_rules! arg_backward_fill {
        (
            $iter:ident,
            $validity:ident,
            $length:ident
        ) => {{
            |$iter, $validity, $length| {
                let Some(start) = $iter
                    .clone()
                    .rev()
                    .position(|i| unsafe { $validity.get_bit_unchecked(i as usize) })
                else {
                    return $iter.collect();
                };

                let mut idx = UnitVec::from_iter($iter);
                let mut current_limit = limit;
                let mut value = idx[$length - start - 1];
                for i in idx[..$length - start].iter_mut().rev() {
                    if unsafe { $validity.get_bit_unchecked(*i as usize) } {
                        current_limit = limit;
                        value = *i;
                    } else if current_limit != 0 {
                        current_limit -= 1;
                        *i = value;
                    }
                }

                idx
            }
        }};
    }

    fw_bw_fill_null(
        inputs,
        df,
        groups,
        state,
        arg_backward_fill!(iter, validity, length),
        arg_backward_fill!(iter, validity, length),
    )
}

/// Build a seed vector aligned with `groups`, where each non-null group receives
/// a seed drawn in canonical (first-row-idx sorted) order. This decouples the
/// seed-to-group assignment from the hashmap iteration order of `groups` while
/// keeping `ac.groups` untouched (so the downstream `map_by_arg_sort` pointer-
/// equality path at `window.rs` stays on its correct branch).
#[cfg(feature = "random")]
fn canonical_per_group_seeds(
    items_mask: &[bool],
    groups_type: &GroupsType,
) -> Vec<Option<u64>> {
    let n_groups = items_mask.len();

    // first[i] = the first row-index of group i (in the iteration order that the
    // ListChunked was produced in, which matches ac.groups / gb.get_groups()).
    let first_rows: Vec<IdxSize> = match groups_type {
        GroupsType::Idx(groups) => groups.first().to_vec(),
        GroupsType::Slice { groups, .. } => groups.iter().map(|[f, _]| *f).collect(),
    };
    debug_assert_eq!(first_rows.len(), n_groups);

    // Canonical ordering over non-null groups: sort their indices by first-row-idx.
    let mut non_null: Vec<usize> = (0..n_groups).filter(|&i| items_mask[i]).collect();
    non_null.sort_by_key(|&i| first_rows[i]);

    // Draw exactly as many seeds as non-null groups (no wasted advancement).
    let drawn = polars_core::random::draw_n_global_seeds(non_null.len());

    // Distribute: the k-th non-null group (canonical order) gets drawn[k].
    let mut seed_of: Vec<Option<u64>> = vec![None; n_groups];
    for (k, &i) in non_null.iter().enumerate() {
        seed_of[i] = Some(drawn[k]);
    }

    seed_of
}

#[cfg(feature = "random")]
pub fn shuffle_over_groups<'a>(
    inputs: &[Arc<dyn PhysicalExpr>],
    df: &DataFrame,
    groups: &'a GroupPositions,
    state: &ExecutionState,
) -> PolarsResult<AggregationContext<'a>> {
    assert_eq!(inputs.len(), 1);
    let mut ac = inputs[0].evaluate_on_groups(df, groups, state)?;
    ac.groups();
    // NOTE: the per-group list produced below is in the iteration order of the
    // ORIGINAL `groups` parameter (upstream from WindowExpr), regardless of
    // whether the inner expression returned NotAggregated or AggregatedList.
    // We therefore derive canonical seed ordering from `groups` directly —
    // NOT from `ac.groups`, because for AggregatedList inputs `ac.groups()`
    // above runs a WithSeriesLen transform that replaces ac.groups with
    // Slice-based groups whose "first" values are cumulative list offsets
    // rather than original row indices.

    // IMPORTANT: we intentionally do NOT call `ac.groups.to_mut().sort()`.
    // Sorting clones the Cow, which breaks the `std::ptr::eq(ac.groups, gb.get_groups())`
    // check in `window.rs::map_list_agg_by_arg_sort`. When the pointer check fails
    // it takes the else-branch, which assumes ac.groups and gb.get_groups() describe
    // the *same groups in the same order* and pairs their row-indices positionally.
    // A reordered clone violates that contract and mis-maps values across groups
    // (see issue #27307 analysis). Instead, we leave ac.groups alone (preserving
    // pointer equality) and stabilise seed assignment via a canonical first-row-idx
    // ordering computed below — giving the same determinism guarantee without
    // disturbing the upstream group representation.

    // Mirror AggregatedScalar / LiteralScalar handling from apply_single_group_aware
    // (apply.rs:154-157).
    let agg = match ac.agg_state() {
        AggState::AggregatedScalar(s) | AggState::LiteralScalar(s) => s.as_list().into_column(),
        _ => ac.aggregated(),
    };
    let ca = agg.list().unwrap();
    let name = agg.name().clone();

    let items: Vec<Option<Series>> = ca.into_iter().collect();
    let n_groups = items.len();
    debug_assert_eq!(
        n_groups,
        ac.groups.len(),
        "aggregated ListChunked length ({}) must equal group count ({})",
        n_groups,
        ac.groups.len(),
    );

    // Pre-draw seeds in canonical order; see `canonical_per_group_seeds`.
    let items_mask: Vec<bool> = items.iter().map(|o| o.is_some()).collect();
    let seeds = canonical_per_group_seeds(&items_mask, groups.as_ref());

    // Parallel dispatch is now safe — each group's seed is pre-determined
    // regardless of rayon scheduling.
    let out: ListChunked = POOL.install(|| {
        try_list_from_par_iter(
            items
                .into_par_iter()
                .zip(seeds.into_par_iter())
                .map(|(opt_s, opt_seed)| -> PolarsResult<Option<Series>> {
                    Ok(match (opt_s, opt_seed) {
                        (Some(s), Some(seed)) => Some(s.shuffle(Some(seed))),
                        _ => None,
                    })
                }),
            PlSmallStr::EMPTY,
        )
    })?;

    // Finalise using the same contract as ApplyExpr::finish_apply_groups (apply.rs:99-103).
    let out_col = out.with_name(name).into_series().into_column();
    ac.with_update_groups(UpdateGroups::WithSeriesLen);
    ac.with_values_and_args(out_col, true, None, false, false)?;
    Ok(ac)
}

#[cfg(feature = "random")]
pub fn sample_over_groups<'a>(
    inputs: &[Arc<dyn PhysicalExpr>],
    df: &DataFrame,
    groups: &'a GroupPositions,
    state: &ExecutionState,
    is_fraction: bool,
    with_replacement: bool,
    shuffle: bool,
) -> PolarsResult<AggregationContext<'a>> {
    assert_eq!(inputs.len(), 2);

    let mut ac_data = inputs[0].evaluate_on_groups(df, groups, state)?;
    let mut ac_n = inputs[1].evaluate_on_groups(df, groups, state)?;
    ac_data.groups();
    ac_n.groups();

    // Same rationale as shuffle_over_groups: do NOT mutate ac_data.groups (it
    // would break pointer-equality in `map_list_agg_by_arg_sort`), and derive
    // canonical seed ordering from the ORIGINAL `groups` parameter so that
    // AggregatedList upstream states (which trigger a WithSeriesLen transform
    // replacing ac.groups with Slice-based list-offsets) are handled correctly.

    let agg_data = match ac_data.agg_state() {
        AggState::AggregatedScalar(s) | AggState::LiteralScalar(s) => s.as_list().into_column(),
        _ => ac_data.aggregated(),
    };
    let ca_data = agg_data.list().unwrap();
    let name = agg_data.name().clone();
    let n_groups = ca_data.len();

    // For `n`/`frac`: a LiteralScalar has length 1 regardless of group count and
    // must be broadcast to n_groups. This mirrors the broadcast in
    // `groups_dispatch::unique` for the scalar-state early-return path.
    let agg_n = match ac_n.agg_state() {
        AggState::LiteralScalar(s) => {
            let mut list = s.as_list().into_column();
            if list.len() == 1 && n_groups != 1 {
                list = list.new_from_index(0, n_groups);
            }
            list
        },
        AggState::AggregatedScalar(s) => s.as_list().into_column(),
        _ => ac_n.aggregated(),
    };
    let ca_n = agg_n.list().unwrap();

    debug_assert_eq!(
        n_groups,
        ac_data.groups.len(),
        "aggregated data ListChunked length ({}) must equal group count ({})",
        n_groups,
        ac_data.groups.len(),
    );
    debug_assert_eq!(n_groups, ca_n.len(), "data and n group counts must match");

    // Collect into parallel arrays so we can build per-group seeds.
    let items_data: Vec<Option<Series>> = ca_data.into_iter().collect();
    let items_n: Vec<Option<Series>> = ca_n.into_iter().collect();
    let items_mask: Vec<bool> = items_data
        .iter()
        .zip(items_n.iter())
        .map(|(d, n)| d.is_some() && n.is_some())
        .collect();
    let seeds = canonical_per_group_seeds(&items_mask, groups.as_ref());

    let inner_dtype = ca_data.inner_dtype().clone();
    let mut builder = get_list_builder(&inner_dtype, n_groups * 5, n_groups, name.clone());

    for ((opt_data, opt_n), opt_seed) in items_data
        .into_iter()
        .zip(items_n.into_iter())
        .zip(seeds.into_iter())
    {
        match (opt_data, opt_n, opt_seed) {
            (Some(data), Some(n_series), Some(seed)) => {
                let seed_opt = Some(seed);
                let out = if is_fraction {
                    let frac_s = n_series.cast(&DataType::Float64)?;
                    let frac = frac_s
                        .f64()?
                        .get(0)
                        .ok_or_else(|| polars_err!(ComputeError: "sample fraction is null"))?;
                    data.sample_frac(frac, with_replacement, shuffle, seed_opt)?
                } else {
                    let n_s = n_series.strict_cast(&IDX_DTYPE)?;
                    let n = n_s
                        .idx()?
                        .get(0)
                        .ok_or_else(|| polars_err!(ComputeError: "sample size is null"))?
                        as usize;
                    data.sample_n(n, with_replacement, shuffle, seed_opt)?
                };
                builder.append_series(&out)?;
            },
            _ => builder.append_null(),
        }
    }

    let ca = builder.finish();
    let out_col = ca.into_series().into_column();
    ac_data.with_update_groups(UpdateGroups::WithSeriesLen);
    ac_data.with_values_and_args(out_col, true, None, false, false)?;
    Ok(ac_data)
}
