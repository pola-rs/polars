use std::borrow::Cow;
use std::sync::Arc;

use arrow::array::PrimitiveArray;
use arrow::bitmap::Bitmap;
use arrow::bitmap::bitmask::BitMask;
use arrow::trusted_len::TrustMyLength;
use polars_compute::unique::{AmortizedUnique, amortized_unique_from_dtype};
use polars_core::POOL;
use polars_core::error::{PolarsResult, polars_bail, polars_ensure};
use polars_core::frame::DataFrame;
use polars_core::prelude::row_encode::encode_rows_unordered;
use polars_core::prelude::{
    AnyValue, ChunkCast, Column, CompatLevel, Float64Chunked, GroupPositions, GroupsType,
    IDX_DTYPE, IntoColumn,
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
