use arrow::array::Array;
use arrow::legacy::kernels::rolling;
use arrow::legacy::kernels::rolling::no_nulls::{MaxWindow, MinWindow};
use arrow::legacy::kernels::take_agg::{
    take_agg_no_null_primitive_iter_unchecked, take_agg_primitive_iter_unchecked,
};
use polars_core::export::num::Bounded;
use polars_core::frame::group_by::aggregations::{
    _agg_helper_idx, _agg_helper_slice, _rolling_apply_agg_window_no_nulls,
    _rolling_apply_agg_window_nulls, _slice_from_offsets, _use_rolling_kernels,
};
use polars_core::prelude::*;
use polars_utils::float::IsFloat;

#[inline(always)]
fn nan_min<T: IsFloat + PartialOrd + Copy>(a: T, b: T) -> T {
    // If b is nan, min is nan, because the comparison failed. We have
    // to poison the result if a is nan.
    let min = if a < b { a } else { b };
    if a.is_nan() {
        a
    } else {
        min
    }
}

#[inline(always)]
fn nan_max<T: IsFloat + PartialOrd + Copy>(a: T, b: T) -> T {
    // See nan_min.
    let max = if a > b { a } else { b };
    if a.is_nan() {
        a
    } else {
        max
    }
}

fn ca_nan_agg<T, Agg>(ca: &ChunkedArray<T>, min_or_max_fn: Agg) -> Option<T::Native>
where
    T: PolarsFloatType,
    Agg: Fn(T::Native, T::Native) -> T::Native + Copy,
{
    let mut cum_agg = None;
    ca.downcast_iter().for_each(|arr| {
        let agg = if arr.null_count() == 0 {
            arr.values().iter().copied().reduce(min_or_max_fn)
        } else {
            arr.iter()
                .unwrap_optional()
                .filter_map(|opt| opt.copied())
                .reduce(min_or_max_fn)
        };
        match cum_agg {
            None => cum_agg = agg,
            Some(a) => cum_agg = agg.map(|agg| agg + a),
        }
    });
    cum_agg
}

pub fn nan_min_s(s: &Series, name: &str) -> Series {
    match s.dtype() {
        DataType::Float32 => {
            let ca = s.f32().unwrap();
            Series::new(name, [ca_nan_agg(ca, nan_min)])
        },
        DataType::Float64 => {
            let ca = s.f64().unwrap();
            Series::new(name, [ca_nan_agg(ca, nan_min)])
        },
        _ => panic!("expected float"),
    }
}
pub fn nan_max_s(s: &Series, name: &str) -> Series {
    match s.dtype() {
        DataType::Float32 => {
            let ca = s.f32().unwrap();
            Series::new(name, [ca_nan_agg(ca, nan_max)])
        },
        DataType::Float64 => {
            let ca = s.f64().unwrap();
            Series::new(name, [ca_nan_agg(ca, nan_max)])
        },
        _ => panic!("expected float"),
    }
}

unsafe fn group_nan_max<T>(ca: &ChunkedArray<T>, groups: &GroupsProxy) -> Series
where
    T: PolarsFloatType,
    ChunkedArray<T>: IntoSeries,
{
    match groups {
        GroupsProxy::Idx(groups) => _agg_helper_idx::<T, _>(groups, |(first, idx)| {
            debug_assert!(idx.len() <= ca.len());
            if idx.is_empty() {
                None
            } else if idx.len() == 1 {
                ca.get(first as usize)
            } else {
                match (ca.has_validity(), ca.chunks().len()) {
                    (false, 1) => Some({
                        take_agg_no_null_primitive_iter_unchecked(
                            ca.downcast_iter().next().unwrap(),
                            idx.iter().map(|i| *i as usize),
                            nan_max,
                            T::Native::min_value(),
                        )
                    }),
                    (_, 1) => take_agg_primitive_iter_unchecked::<T::Native, _, _>(
                        ca.downcast_iter().next().unwrap(),
                        idx.iter().map(|i| *i as usize),
                        nan_max,
                        T::Native::min_value(),
                        idx.len() as IdxSize,
                    ),
                    _ => {
                        let take = { ca.take_unchecked(idx) };
                        ca_nan_agg(&take, nan_max)
                    },
                }
            }
        }),
        GroupsProxy::Slice {
            groups: groups_slice,
            ..
        } => {
            if _use_rolling_kernels(groups_slice, ca.chunks()) {
                let arr = ca.downcast_iter().next().unwrap();
                let values = arr.values().as_slice();
                let offset_iter = groups_slice.iter().map(|[first, len]| (*first, *len));
                let arr = match arr.validity() {
                    None => _rolling_apply_agg_window_no_nulls::<MaxWindow<_>, _, _>(
                        values,
                        offset_iter,
                        None,
                    ),
                    Some(validity) => _rolling_apply_agg_window_nulls::<
                        rolling::nulls::MaxWindow<_>,
                        _,
                        _,
                    >(values, validity, offset_iter, None),
                };
                ChunkedArray::from(arr).into_series()
            } else {
                _agg_helper_slice::<T, _>(groups_slice, |[first, len]| {
                    debug_assert!(len <= ca.len() as IdxSize);
                    match len {
                        0 => None,
                        1 => ca.get(first as usize),
                        _ => {
                            let arr_group = _slice_from_offsets(ca, first, len);
                            ca_nan_agg(&arr_group, nan_max)
                        },
                    }
                })
            }
        },
    }
}

unsafe fn group_nan_min<T>(ca: &ChunkedArray<T>, groups: &GroupsProxy) -> Series
where
    T: PolarsFloatType,
    ChunkedArray<T>: IntoSeries,
{
    match groups {
        GroupsProxy::Idx(groups) => _agg_helper_idx::<T, _>(groups, |(first, idx)| {
            debug_assert!(idx.len() <= ca.len());
            if idx.is_empty() {
                None
            } else if idx.len() == 1 {
                ca.get(first as usize)
            } else {
                match (ca.has_validity(), ca.chunks().len()) {
                    (false, 1) => Some(take_agg_no_null_primitive_iter_unchecked(
                        ca.downcast_iter().next().unwrap(),
                        idx.iter().map(|i| *i as usize),
                        nan_min,
                        T::Native::max_value(),
                    )),
                    (_, 1) => take_agg_primitive_iter_unchecked::<T::Native, _, _>(
                        ca.downcast_iter().next().unwrap(),
                        idx.iter().map(|i| *i as usize),
                        nan_min,
                        T::Native::max_value(),
                        idx.len() as IdxSize,
                    ),
                    _ => {
                        let take = { ca.take_unchecked(idx) };
                        ca_nan_agg(&take, nan_min)
                    },
                }
            }
        }),
        GroupsProxy::Slice {
            groups: groups_slice,
            ..
        } => {
            if _use_rolling_kernels(groups_slice, ca.chunks()) {
                let arr = ca.downcast_iter().next().unwrap();
                let values = arr.values().as_slice();
                let offset_iter = groups_slice.iter().map(|[first, len]| (*first, *len));
                let arr = match arr.validity() {
                    None => _rolling_apply_agg_window_no_nulls::<MinWindow<_>, _, _>(
                        values,
                        offset_iter,
                        None,
                    ),
                    Some(validity) => _rolling_apply_agg_window_nulls::<
                        rolling::nulls::MinWindow<_>,
                        _,
                        _,
                    >(values, validity, offset_iter, None),
                };
                ChunkedArray::from(arr).into_series()
            } else {
                _agg_helper_slice::<T, _>(groups_slice, |[first, len]| {
                    debug_assert!(len <= ca.len() as IdxSize);
                    match len {
                        0 => None,
                        1 => ca.get(first as usize),
                        _ => {
                            let arr_group = _slice_from_offsets(ca, first, len);
                            ca_nan_agg(&arr_group, nan_min)
                        },
                    }
                })
            }
        },
    }
}

/// # Safety
/// `groups` must be in bounds.
pub unsafe fn group_agg_nan_min_s(s: &Series, groups: &GroupsProxy) -> Series {
    match s.dtype() {
        DataType::Float32 => {
            let ca = s.f32().unwrap();
            group_nan_min(ca, groups)
        },
        DataType::Float64 => {
            let ca = s.f64().unwrap();
            group_nan_min(ca, groups)
        },
        _ => panic!("expected float"),
    }
}

/// # Safety
/// `groups` must be in bounds.
pub unsafe fn group_agg_nan_max_s(s: &Series, groups: &GroupsProxy) -> Series {
    match s.dtype() {
        DataType::Float32 => {
            let ca = s.f32().unwrap();
            group_nan_max(ca, groups)
        },
        DataType::Float64 => {
            let ca = s.f64().unwrap();
            group_nan_max(ca, groups)
        },
        _ => panic!("expected float"),
    }
}
