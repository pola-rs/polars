#![allow(unsafe_op_in_unsafe_fn)]
use arrow::legacy::kernels::take_agg::{
    take_agg_no_null_primitive_iter_unchecked, take_agg_primitive_iter_unchecked,
};
use polars_compute::rolling;
use polars_compute::rolling::no_nulls::{MaxWindow, MinWindow};
use polars_core::chunked_array::arrow_bridge::{chunk_from_arrow, chunk_to_arrow};
use polars_core::frame::group_by::aggregations::{
    _agg_helper_idx, _agg_helper_slice, _rolling_apply_agg_window_no_nulls,
    _rolling_apply_agg_window_nulls, _slice_from_offsets, _use_rolling_kernels,
};
use polars_core::prelude::*;
use polars_utils::min_max::MinMax;

pub fn ca_nan_agg<T, Agg>(ca: &ChunkedArray<T>, min_or_max_fn: Agg) -> Option<T::Native>
where
    T: PolarsFloatType,
    Agg: Fn(T::Native, T::Native) -> T::Native + Copy,
{
    ca.downcast_iter()
        .filter_map(|arr| {
            if arr.null_count() == 0 {
                arr.values_iter().reduce(min_or_max_fn)
            } else {
                arr.iter().flatten().reduce(min_or_max_fn)
            }
        })
        .reduce(min_or_max_fn)
}

pub fn nan_min_s(s: &Series, name: PlSmallStr) -> Series {
    match s.dtype() {
        #[cfg(feature = "dtype-f16")]
        DataType::Float16 => {
            let ca = s.f16().unwrap();
            Series::new(name, [ca_nan_agg(ca, MinMax::min_propagate_nan)])
        },
        DataType::Float32 => {
            let ca = s.f32().unwrap();
            Series::new(name, [ca_nan_agg(ca, MinMax::min_propagate_nan)])
        },
        DataType::Float64 => {
            let ca = s.f64().unwrap();
            Series::new(name, [ca_nan_agg(ca, MinMax::min_propagate_nan)])
        },
        _ => panic!("expected float"),
    }
}

pub fn nan_max_s(s: &Series, name: PlSmallStr) -> Series {
    match s.dtype() {
        #[cfg(feature = "dtype-f16")]
        DataType::Float16 => {
            let ca = s.f16().unwrap();
            Series::new(name, [ca_nan_agg(ca, MinMax::max_propagate_nan)])
        },
        DataType::Float32 => {
            let ca = s.f32().unwrap();
            Series::new(name, [ca_nan_agg(ca, MinMax::max_propagate_nan)])
        },
        DataType::Float64 => {
            let ca = s.f64().unwrap();
            Series::new(name, [ca_nan_agg(ca, MinMax::max_propagate_nan)])
        },
        _ => panic!("expected float"),
    }
}

unsafe fn group_nan_max<T: PolarsFloatType>(ca: &ChunkedArray<T>, groups: &GroupsType) -> Series {
    match groups {
        GroupsType::Idx(groups) => _agg_helper_idx::<T, _>(groups, |(first, idx)| {
            debug_assert!(idx.len() <= ca.len());
            if idx.is_empty() {
                None
            } else if idx.len() == 1 {
                ca.get(first as usize)
            } else {
                match (ca.has_nulls(), ca.chunks().len()) {
                    // TODO(polars-array-scalar): the take kernels are Arrow ones, so a scalar
                    // chunk is written out here rather than the one value it stands for being
                    // taken however many times the group asks for it.
                    (false, 1) => take_agg_no_null_primitive_iter_unchecked(
                        &chunk_to_arrow(ca.downcast_iter().next().unwrap()),
                        idx.iter().map(|i| *i as usize),
                    )
                    .reduce(MinMax::max_propagate_nan),
                    (_, 1) => take_agg_primitive_iter_unchecked(
                        &chunk_to_arrow(ca.downcast_iter().next().unwrap()),
                        idx.iter().map(|i| *i as usize),
                    )
                    .reduce(MinMax::max_propagate_nan),
                    _ => {
                        let take = { ca.take_unchecked(idx) };
                        ca_nan_agg(&take, MinMax::max_propagate_nan)
                    },
                }
            }
        }),
        GroupsType::Slice {
            groups: groups_slice,
            overlapping,
            monotonic,
        } => {
            if _use_rolling_kernels(groups_slice, *overlapping, *monotonic, ca.chunks()) {
                // TODO(polars-array-scalar): the rolling kernels read the values as a slice, so
                // a scalar chunk is written out here rather than aggregated once.
                let flat = ca.to_flat();
                let arr = flat
                    .flat_chunk(0)
                    .expect("the rolling kernels need one chunk");
                let values = arr.as_slice();
                let offset_iter = groups_slice.iter().map(|[first, len]| (*first, *len));
                let arr = match arr.validity() {
                    None => _rolling_apply_agg_window_no_nulls::<MaxWindow<_>, _, _, _>(
                        values,
                        offset_iter,
                        None,
                    ),
                    Some(validity) => _rolling_apply_agg_window_nulls::<
                        rolling::nulls::MaxWindow<_>,
                        _,
                        _,
                        _,
                    >(values, validity, offset_iter, None),
                };
                ChunkedArray::<T>::with_chunk(PlSmallStr::EMPTY, chunk_from_arrow(&arr))
                    .into_series()
            } else {
                _agg_helper_slice::<T, _>(groups_slice, |[first, len]| {
                    debug_assert!(len <= ca.len() as IdxSize);
                    match len {
                        0 => None,
                        1 => ca.get(first as usize),
                        _ => {
                            let arr_group = _slice_from_offsets(ca, first, len);
                            ca_nan_agg(&arr_group, MinMax::max_propagate_nan)
                        },
                    }
                })
            }
        },
    }
}

unsafe fn group_nan_min<T: PolarsFloatType>(ca: &ChunkedArray<T>, groups: &GroupsType) -> Series {
    match groups {
        GroupsType::Idx(groups) => _agg_helper_idx::<T, _>(groups, |(first, idx)| {
            debug_assert!(idx.len() <= ca.len());
            if idx.is_empty() {
                None
            } else if idx.len() == 1 {
                ca.get(first as usize)
            } else {
                match (ca.has_nulls(), ca.chunks().len()) {
                    // TODO(polars-array-scalar): the take kernels are Arrow ones, so a scalar
                    // chunk is written out here rather than the one value it stands for being
                    // taken however many times the group asks for it.
                    (false, 1) => take_agg_no_null_primitive_iter_unchecked(
                        &chunk_to_arrow(ca.downcast_iter().next().unwrap()),
                        idx.iter().map(|i| *i as usize),
                    )
                    .reduce(MinMax::min_propagate_nan),
                    (_, 1) => take_agg_primitive_iter_unchecked(
                        &chunk_to_arrow(ca.downcast_iter().next().unwrap()),
                        idx.iter().map(|i| *i as usize),
                    )
                    .reduce(MinMax::min_propagate_nan),
                    _ => {
                        let take = { ca.take_unchecked(idx) };
                        ca_nan_agg(&take, MinMax::min_propagate_nan)
                    },
                }
            }
        }),
        GroupsType::Slice {
            groups: groups_slice,
            overlapping,
            monotonic,
        } => {
            if _use_rolling_kernels(groups_slice, *overlapping, *monotonic, ca.chunks()) {
                // TODO(polars-array-scalar): the rolling kernels read the values as a slice, so
                // a scalar chunk is written out here rather than aggregated once.
                let flat = ca.to_flat();
                let arr = flat
                    .flat_chunk(0)
                    .expect("the rolling kernels need one chunk");
                let values = arr.as_slice();
                let offset_iter = groups_slice.iter().map(|[first, len]| (*first, *len));
                let arr = match arr.validity() {
                    None => _rolling_apply_agg_window_no_nulls::<MinWindow<_>, _, _, _>(
                        values,
                        offset_iter,
                        None,
                    ),
                    Some(validity) => _rolling_apply_agg_window_nulls::<
                        rolling::nulls::MinWindow<_>,
                        _,
                        _,
                        _,
                    >(values, validity, offset_iter, None),
                };
                ChunkedArray::<T>::with_chunk(PlSmallStr::EMPTY, chunk_from_arrow(&arr))
                    .into_series()
            } else {
                _agg_helper_slice::<T, _>(groups_slice, |[first, len]| {
                    debug_assert!(len <= ca.len() as IdxSize);
                    match len {
                        0 => None,
                        1 => ca.get(first as usize),
                        _ => {
                            let arr_group = _slice_from_offsets(ca, first, len);
                            ca_nan_agg(&arr_group, MinMax::min_propagate_nan)
                        },
                    }
                })
            }
        },
    }
}

/// # Safety
/// `groups` must be in bounds.
pub unsafe fn group_agg_nan_min_s(s: &Series, groups: &GroupsType) -> Series {
    match s.dtype() {
        #[cfg(feature = "dtype-f16")]
        DataType::Float16 => {
            let ca = s.f16().unwrap();
            group_nan_min(ca, groups)
        },
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
pub unsafe fn group_agg_nan_max_s(s: &Series, groups: &GroupsType) -> Series {
    match s.dtype() {
        #[cfg(feature = "dtype-f16")]
        DataType::Float16 => {
            let ca = s.f16().unwrap();
            group_nan_max(ca, groups)
        },
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
