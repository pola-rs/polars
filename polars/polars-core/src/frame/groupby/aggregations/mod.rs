mod agg_list;

use crate::POOL;
pub use agg_list::*;
use arrow::bitmap::MutableBitmap;
use num::{Bounded, Num, NumCast, ToPrimitive, Zero};
use rayon::prelude::*;

use crate::apply_method_physical_integer;
use arrow::types::{simd::Simd, NativeType};

#[cfg(feature = "object")]
use crate::chunked_array::object::extension::create_extension;
use crate::frame::groupby::GroupsIdx;
#[cfg(feature = "object")]
use crate::frame::groupby::GroupsIndicator;
use crate::prelude::*;
use crate::series::implementations::SeriesWrap;
use polars_arrow::kernels::take_agg::*;
use polars_arrow::prelude::QuantileInterpolOptions;
use polars_arrow::trusted_len::PushUnchecked;

fn slice_from_offsets<T>(ca: &ChunkedArray<T>, first: IdxSize, len: IdxSize) -> ChunkedArray<T>
where
    ChunkedArray<T>: ChunkOps,
{
    ca.slice(first as i64, len as usize)
}

// helper that combines the groups into a parallel iterator over `(first, all): (u32, &Vec<u32>)`
fn agg_helper_idx<T, F>(groups: &GroupsIdx, f: F) -> Series
where
    F: Fn((IdxSize, &Vec<IdxSize>)) -> Option<T::Native> + Send + Sync,
    T: PolarsNumericType,
    ChunkedArray<T>: IntoSeries,
{
    let ca: ChunkedArray<T> = POOL.install(|| groups.into_par_iter().map(f).collect());
    ca.into_series()
}

// helper that iterates on the `all: Vec<Vec<u32>` collection
// this doesn't have traverse the `first: Vec<u32>` memory and is therefore faster
fn agg_helper_idx_on_all<T, F>(groups: &GroupsIdx, f: F) -> Series
where
    F: Fn(&Vec<IdxSize>) -> Option<T::Native> + Send + Sync,
    T: PolarsNumericType,
    ChunkedArray<T>: IntoSeries,
{
    let ca: ChunkedArray<T> = POOL.install(|| groups.all().into_par_iter().map(f).collect());
    ca.into_series()
}

fn agg_helper_slice<T, F>(groups: &[[IdxSize; 2]], f: F) -> Series
where
    F: Fn([IdxSize; 2]) -> Option<T::Native> + Send + Sync,
    T: PolarsNumericType,
    ChunkedArray<T>: IntoSeries,
{
    let ca: ChunkedArray<T> = POOL.install(|| groups.par_iter().copied().map(f).collect());
    ca.into_series()
}

impl BooleanChunked {
    pub(crate) fn agg_min(&self, groups: &GroupsProxy) -> Series {
        self.cast(&IDX_DTYPE).unwrap().agg_min(groups)
    }
    pub(crate) fn agg_max(&self, groups: &GroupsProxy) -> Series {
        self.cast(&IDX_DTYPE).unwrap().agg_max(groups)
    }
    pub(crate) fn agg_sum(&self, groups: &GroupsProxy) -> Series {
        self.cast(&IDX_DTYPE).unwrap().agg_sum(groups)
    }
}

// implemented on the series because we don't need types
impl Series {
    fn slice_from_offsets(&self, first: IdxSize, len: IdxSize) -> Self {
        self.slice(first as i64, len as usize)
    }

    fn restore_logical(&self, out: Series) -> Series {
        if self.is_logical() {
            out.cast(self.dtype()).unwrap()
        } else {
            out
        }
    }

    #[doc(hidden)]
    pub fn agg_valid_count(&self, groups: &GroupsProxy) -> Series {
        match groups {
            GroupsProxy::Idx(groups) => agg_helper_idx_on_all::<IdxType, _>(groups, |idx| {
                debug_assert!(idx.len() <= self.len());
                if idx.is_empty() {
                    None
                } else if !self.has_validity() {
                    Some(idx.len() as IdxSize)
                } else {
                    let take =
                        unsafe { self.take_iter_unchecked(&mut idx.iter().map(|i| *i as usize)) };
                    Some((take.len() - take.null_count()) as IdxSize)
                }
            }),
            GroupsProxy::Slice(groups) => agg_helper_slice::<IdxType, _>(groups, |[first, len]| {
                debug_assert!(len <= self.len() as IdxSize);
                if len == 0 {
                    None
                } else if !self.has_validity() {
                    Some(len)
                } else {
                    let take = self.slice_from_offsets(first, len);
                    Some((take.len() - take.null_count()) as IdxSize)
                }
            }),
        }
    }

    #[doc(hidden)]
    pub fn agg_first(&self, groups: &GroupsProxy) -> Series {
        let out = match groups {
            GroupsProxy::Idx(groups) => {
                let mut iter = groups.iter().map(|(first, idx)| {
                    if idx.is_empty() {
                        None
                    } else {
                        Some(first as usize)
                    }
                });
                // Safety:
                // groups are always in bounds
                unsafe { self.take_opt_iter_unchecked(&mut iter) }
            }
            GroupsProxy::Slice(groups) => {
                let mut iter =
                    groups.iter().map(
                        |&[first, len]| {
                            if len == 0 {
                                None
                            } else {
                                Some(first as usize)
                            }
                        },
                    );
                // Safety:
                // groups are always in bounds
                unsafe { self.take_opt_iter_unchecked(&mut iter) }
            }
        };
        self.restore_logical(out)
    }

    #[doc(hidden)]
    pub fn agg_n_unique(&self, groups: &GroupsProxy) -> Series {
        match groups {
            GroupsProxy::Idx(groups) => agg_helper_idx_on_all::<IdxType, _>(groups, |idx| {
                debug_assert!(idx.len() <= self.len());
                if idx.is_empty() {
                    None
                } else {
                    let take =
                        unsafe { self.take_iter_unchecked(&mut idx.iter().map(|i| *i as usize)) };
                    take.n_unique().ok().map(|v| v as IdxSize)
                }
            }),
            GroupsProxy::Slice(groups) => agg_helper_slice::<IdxType, _>(groups, |[first, len]| {
                debug_assert!(len <= self.len() as IdxSize);
                if len == 0 {
                    None
                } else {
                    let take = self.slice_from_offsets(first, len);
                    take.n_unique().ok().map(|v| v as IdxSize)
                }
            }),
        }
    }

    #[doc(hidden)]
    pub fn agg_mean(&self, groups: &GroupsProxy) -> Series {
        use DataType::*;
        if self.dtype().is_numeric() {
            match self.dtype() {
                Float32 => SeriesWrap(self.f32().unwrap().clone()).agg_mean(groups),
                Float64 => SeriesWrap(self.f64().unwrap().clone()).agg_mean(groups),
                _ => {
                    apply_method_physical_integer!(self, agg_mean, groups)
                }
            }
        } else {
            Series::full_null(self.name(), groups.len(), self.dtype())
        }
    }

    #[doc(hidden)]
    pub fn agg_last(&self, groups: &GroupsProxy) -> Series {
        let out = match groups {
            GroupsProxy::Idx(groups) => {
                let mut iter = groups.all().iter().map(|idx| {
                    if idx.is_empty() {
                        None
                    } else {
                        Some(idx[idx.len() - 1] as usize)
                    }
                });
                unsafe { self.take_opt_iter_unchecked(&mut iter) }
            }
            GroupsProxy::Slice(groups) => {
                let mut iter = groups.iter().map(|&[first, len]| {
                    if len == 0 {
                        None
                    } else {
                        Some((first + len - 1) as usize)
                    }
                });
                unsafe { self.take_opt_iter_unchecked(&mut iter) }
            }
        };
        self.restore_logical(out)
    }
}

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType + Sync,
    T::Native:
        NativeType + PartialOrd + Num + NumCast + Zero + Simd + Bounded + std::iter::Sum<T::Native>,
    <T::Native as Simd>::Simd: std::ops::Add<Output = <T::Native as Simd>::Simd>
        + arrow::compute::aggregate::Sum<T::Native>
        + arrow::compute::aggregate::SimdOrd<T::Native>,
    ChunkedArray<T>: IntoSeries,
{
    pub(crate) fn agg_min(&self, groups: &GroupsProxy) -> Series {
        match groups {
            GroupsProxy::Idx(groups) => agg_helper_idx::<T, _>(groups, |(first, idx)| {
                debug_assert!(idx.len() <= self.len());
                if idx.is_empty() {
                    None
                } else if idx.len() == 1 {
                    self.get(first as usize)
                } else {
                    match (self.has_validity(), self.chunks.len()) {
                        (false, 1) => unsafe {
                            Some(take_agg_no_null_primitive_iter_unchecked(
                                self.downcast_iter().next().unwrap(),
                                idx.iter().map(|i| *i as usize),
                                |a, b| if a < b { a } else { b },
                                T::Native::max_value(),
                            ))
                        },
                        (_, 1) => unsafe {
                            take_agg_primitive_iter_unchecked::<T::Native, _, _>(
                                self.downcast_iter().next().unwrap(),
                                idx.iter().map(|i| *i as usize),
                                |a, b| if a < b { a } else { b },
                                T::Native::max_value(),
                            )
                        },
                        _ => {
                            let take = unsafe { self.take_unchecked(idx.into()) };
                            take.min()
                        }
                    }
                }
            }),
            GroupsProxy::Slice(groups) => agg_helper_slice::<T, _>(groups, |[first, len]| {
                debug_assert!(len <= self.len() as IdxSize);
                match len {
                    0 => None,
                    1 => self.get(first as usize),
                    _ => {
                        let arr_group = slice_from_offsets(self, first, len);
                        arr_group.min()
                    }
                }
            }),
        }
    }

    pub(crate) fn agg_max(&self, groups: &GroupsProxy) -> Series {
        match groups {
            GroupsProxy::Idx(groups) => agg_helper_idx::<T, _>(groups, |(first, idx)| {
                debug_assert!(idx.len() <= self.len());
                if idx.is_empty() {
                    None
                } else if idx.len() == 1 {
                    self.get(first as usize)
                } else {
                    match (self.has_validity(), self.chunks.len()) {
                        (false, 1) => Some(unsafe {
                            take_agg_no_null_primitive_iter_unchecked(
                                self.downcast_iter().next().unwrap(),
                                idx.iter().map(|i| *i as usize),
                                |a, b| if a > b { a } else { b },
                                T::Native::min_value(),
                            )
                        }),
                        (_, 1) => unsafe {
                            take_agg_primitive_iter_unchecked::<T::Native, _, _>(
                                self.downcast_iter().next().unwrap(),
                                idx.iter().map(|i| *i as usize),
                                |a, b| if a > b { a } else { b },
                                T::Native::min_value(),
                            )
                        },
                        _ => {
                            let take = unsafe { self.take_unchecked(idx.into()) };
                            take.max()
                        }
                    }
                }
            }),
            GroupsProxy::Slice(groups) => agg_helper_slice::<T, _>(groups, |[first, len]| {
                debug_assert!(len <= self.len() as IdxSize);
                match len {
                    0 => None,
                    1 => self.get(first as usize),
                    _ => {
                        let arr_group = slice_from_offsets(self, first, len);
                        arr_group.max()
                    }
                }
            }),
        }
    }

    pub(crate) fn agg_sum(&self, groups: &GroupsProxy) -> Series {
        match groups {
            GroupsProxy::Idx(groups) => agg_helper_idx::<T, _>(groups, |(first, idx)| {
                debug_assert!(idx.len() <= self.len());
                if idx.is_empty() {
                    None
                } else if idx.len() == 1 {
                    self.get(first as usize)
                } else {
                    match (self.has_validity(), self.chunks.len()) {
                        (false, 1) => Some(unsafe {
                            take_agg_no_null_primitive_iter_unchecked(
                                self.downcast_iter().next().unwrap(),
                                idx.iter().map(|i| *i as usize),
                                |a, b| a + b,
                                T::Native::zero(),
                            )
                        }),
                        (_, 1) => unsafe {
                            take_agg_primitive_iter_unchecked::<T::Native, _, _>(
                                self.downcast_iter().next().unwrap(),
                                idx.iter().map(|i| *i as usize),
                                |a, b| a + b,
                                T::Native::zero(),
                            )
                        },
                        _ => {
                            let take = unsafe { self.take_unchecked(idx.into()) };
                            take.sum()
                        }
                    }
                }
            }),
            GroupsProxy::Slice(groups) => agg_helper_slice::<T, _>(groups, |[first, len]| {
                debug_assert!(len <= self.len() as IdxSize);
                match len {
                    0 => None,
                    1 => self.get(first as usize),
                    _ => {
                        let arr_group = slice_from_offsets(self, first, len);
                        arr_group.sum()
                    }
                }
            }),
        }
    }
}

impl<T> SeriesWrap<ChunkedArray<T>>
where
    T: PolarsFloatType,
    ChunkedArray<T>: IntoSeries
        + ChunkVar<T::Native>
        + VarAggSeries
        + ChunkQuantile<T::Native>
        + QuantileAggSeries,
    T::Native: NativeType + PartialOrd + Num + NumCast + Simd + std::iter::Sum<T::Native>,
    <T::Native as Simd>::Simd: std::ops::Add<Output = <T::Native as Simd>::Simd>
        + arrow::compute::aggregate::Sum<T::Native>
        + arrow::compute::aggregate::SimdOrd<T::Native>,
{
    pub(crate) fn agg_mean(&self, groups: &GroupsProxy) -> Series {
        match groups {
            GroupsProxy::Idx(groups) => {
                agg_helper_idx::<T, _>(groups, |(first, idx)| {
                    // this can fail due to a bug in lazy code.
                    // here users can create filters in aggregations
                    // and thereby creating shorter columns than the original group tuples.
                    // the group tuples are modified, but if that's done incorrect there can be out of bounds
                    // access
                    debug_assert!(idx.len() <= self.len());
                    let out = if idx.is_empty() {
                        None
                    } else if idx.len() == 1 {
                        self.get(first as usize).map(|sum| sum.to_f64().unwrap())
                    } else {
                        match (self.has_validity(), self.chunks.len()) {
                            (false, 1) => unsafe {
                                take_agg_no_null_primitive_iter_unchecked(
                                    self.downcast_iter().next().unwrap(),
                                    idx.iter().map(|i| *i as usize),
                                    |a, b| a + b,
                                    T::Native::zero(),
                                )
                            }
                            .to_f64()
                            .map(|sum| sum / idx.len() as f64),
                            (_, 1) => unsafe {
                                take_agg_primitive_iter_unchecked_count_nulls::<T::Native, _, _, _>(
                                    self.downcast_iter().next().unwrap(),
                                    idx.iter().map(|i| *i as usize),
                                    |a, b| a + b,
                                    T::Native::zero(),
                                )
                            }
                            .map(|(sum, null_count)| {
                                sum.to_f64()
                                    .map(|sum| sum / (idx.len() as f64 - null_count as f64))
                                    .unwrap()
                            }),
                            _ => {
                                let take = unsafe { self.take_unchecked(idx.into()) };
                                take.mean()
                            }
                        }
                    };
                    out.map(|flt| NumCast::from(flt).unwrap())
                })
            }
            GroupsProxy::Slice(groups) => agg_helper_slice::<T, _>(groups, |[first, len]| {
                debug_assert!(len <= self.len() as IdxSize);
                match len {
                    0 => None,
                    1 => self.get(first as usize),
                    _ => {
                        let arr_group = slice_from_offsets(self, first, len);
                        arr_group.mean().map(|flt| NumCast::from(flt).unwrap())
                    }
                }
            }),
        }
    }

    pub(crate) fn agg_var(&self, groups: &GroupsProxy) -> Series {
        let ca = &self.0;
        match groups {
            GroupsProxy::Idx(groups) => agg_helper_idx_on_all::<T, _>(groups, |idx| {
                debug_assert!(idx.len() <= ca.len());
                if idx.is_empty() {
                    return None;
                }
                let take = unsafe { ca.take_unchecked(idx.into()) };
                take.var_as_series().unpack::<T>().unwrap().get(0)
            }),
            GroupsProxy::Slice(groups) => agg_helper_slice::<T, _>(groups, |[first, len]| {
                debug_assert!(len <= self.len() as IdxSize);
                match len {
                    0 => None,
                    1 => self.get(first as usize),
                    _ => {
                        let arr_group = slice_from_offsets(self, first, len);
                        arr_group.var().map(|flt| NumCast::from(flt).unwrap())
                    }
                }
            }),
        }
    }
    pub(crate) fn agg_std(&self, groups: &GroupsProxy) -> Series {
        let ca = &self.0;
        match groups {
            GroupsProxy::Idx(groups) => agg_helper_idx_on_all::<T, _>(groups, |idx| {
                debug_assert!(idx.len() <= ca.len());
                if idx.is_empty() {
                    return None;
                }
                let take = unsafe { ca.take_unchecked(idx.into()) };
                take.std_as_series().unpack::<T>().unwrap().get(0)
            }),
            GroupsProxy::Slice(groups) => agg_helper_slice::<T, _>(groups, |[first, len]| {
                debug_assert!(len <= self.len() as IdxSize);
                match len {
                    0 => None,
                    1 => self.get(first as usize).map(|v| NumCast::from(v).unwrap()),
                    _ => {
                        let arr_group = slice_from_offsets(self, first, len);
                        arr_group.std().map(|flt| NumCast::from(flt).unwrap())
                    }
                }
            }),
        }
    }

    pub(crate) fn agg_quantile(
        &self,
        groups: &GroupsProxy,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> Series {
        let ca = &self.0;
        let invalid_quantile = !(0.0..=1.0).contains(&quantile);
        match groups {
            GroupsProxy::Idx(groups) => agg_helper_idx_on_all::<T, _>(groups, |idx| {
                debug_assert!(idx.len() <= ca.len());
                if idx.is_empty() | invalid_quantile {
                    return None;
                }
                let take = unsafe { ca.take_unchecked(idx.into()) };
                take.quantile_as_series(quantile, interpol)
                    .unwrap() // checked with invalid quantile check
                    .unpack::<T>()
                    .unwrap()
                    .get(0)
            }),
            GroupsProxy::Slice(groups) => agg_helper_slice::<T, _>(groups, |[first, len]| {
                debug_assert!(len <= self.len() as IdxSize);
                match len {
                    0 => None,
                    1 => self.get(first as usize),
                    _ => {
                        let arr_group = slice_from_offsets(self, first, len);
                        // unwrap checked with invalid quantile check
                        arr_group
                            .quantile(quantile, interpol)
                            .unwrap()
                            .map(|flt| NumCast::from(flt).unwrap())
                    }
                }
            }),
        }
    }
    pub(crate) fn agg_median(&self, groups: &GroupsProxy) -> Series {
        let ca = &self.0;
        match groups {
            GroupsProxy::Idx(groups) => agg_helper_idx_on_all::<T, _>(groups, |idx| {
                debug_assert!(idx.len() <= ca.len());
                if idx.is_empty() {
                    return None;
                }
                let take = unsafe { ca.take_unchecked(idx.into()) };
                take.median_as_series().unpack::<T>().unwrap().get(0)
            }),
            GroupsProxy::Slice(groups) => agg_helper_slice::<T, _>(groups, |[first, len]| {
                debug_assert!(len <= self.len() as IdxSize);
                match len {
                    0 => None,
                    1 => self.get(first as usize).map(|v| NumCast::from(v).unwrap()),
                    _ => {
                        let arr_group = slice_from_offsets(self, first, len);
                        arr_group.median().map(|flt| NumCast::from(flt).unwrap())
                    }
                }
            }),
        }
    }
}

impl<T> ChunkedArray<T>
where
    T: PolarsIntegerType,
    ChunkedArray<T>: IntoSeries,
    T::Native: NativeType + Num + NumCast + Zero + Simd + Bounded + std::iter::Sum<T::Native> + Ord,
    <T::Native as Simd>::Simd: std::ops::Add<Output = <T::Native as Simd>::Simd>
        + arrow::compute::aggregate::Sum<T::Native>
        + arrow::compute::aggregate::SimdOrd<T::Native>,
{
    pub(crate) fn agg_mean(&self, groups: &GroupsProxy) -> Series {
        match groups {
            GroupsProxy::Idx(groups) => {
                agg_helper_idx::<Float64Type, _>(groups, |(first, idx)| {
                    // this can fail due to a bug in lazy code.
                    // here users can create filters in aggregations
                    // and thereby creating shorter columns than the original group tuples.
                    // the group tuples are modified, but if that's done incorrect there can be out of bounds
                    // access
                    debug_assert!(idx.len() <= self.len());
                    if idx.is_empty() {
                        None
                    } else if idx.len() == 1 {
                        self.get(first as usize).map(|sum| sum.to_f64().unwrap())
                    } else {
                        match (self.has_validity(), self.chunks.len()) {
                            (false, 1) => unsafe {
                                take_agg_no_null_primitive_iter_unchecked(
                                    self.downcast_iter().next().unwrap(),
                                    idx.iter().map(|i| *i as usize),
                                    |a, b| a + b,
                                    0.0f64,
                                )
                            }
                            .to_f64()
                            .map(|sum| sum / idx.len() as f64),
                            (_, 1) => {
                                unsafe {
                                    take_agg_primitive_iter_unchecked_count_nulls::<
                                        T::Native,
                                        f64,
                                        _,
                                        _,
                                    >(
                                        self.downcast_iter().next().unwrap(),
                                        idx.iter().map(|i| *i as usize),
                                        |a, b| a + b,
                                        0.0,
                                    )
                                }
                                .map(|(sum, null_count)| {
                                    sum / (idx.len() as f64 - null_count as f64)
                                })
                            }
                            _ => {
                                let take = unsafe { self.take_unchecked(idx.into()) };
                                take.mean()
                            }
                        }
                    }
                })
            }
            GroupsProxy::Slice(groups) => {
                agg_helper_slice::<Float64Type, _>(groups, |[first, len]| {
                    debug_assert!(len < self.len() as IdxSize);
                    match len {
                        0 => None,
                        1 => self.get(first as usize).map(|v| NumCast::from(v).unwrap()),
                        _ => {
                            let arr_group = slice_from_offsets(self, first, len);
                            arr_group.mean()
                        }
                    }
                })
            }
        }
    }

    pub(crate) fn agg_var(&self, groups: &GroupsProxy) -> Series {
        match groups {
            GroupsProxy::Idx(groups) => agg_helper_idx_on_all::<Float64Type, _>(groups, |idx| {
                debug_assert!(idx.len() <= self.len());
                if idx.is_empty() {
                    return None;
                }
                let take = unsafe { self.take_unchecked(idx.into()) };
                take.var_as_series().unpack::<Float64Type>().unwrap().get(0)
            }),
            GroupsProxy::Slice(groups) => {
                agg_helper_slice::<Float64Type, _>(groups, |[first, len]| {
                    debug_assert!(len <= self.len() as IdxSize);
                    match len {
                        0 => None,
                        1 => self.get(first as usize).map(|v| NumCast::from(v).unwrap()),
                        _ => {
                            let arr_group = slice_from_offsets(self, first, len);
                            arr_group.var()
                        }
                    }
                })
            }
        }
    }
    pub(crate) fn agg_std(&self, groups: &GroupsProxy) -> Series {
        match groups {
            GroupsProxy::Idx(groups) => agg_helper_idx_on_all::<Float64Type, _>(groups, |idx| {
                debug_assert!(idx.len() <= self.len());
                if idx.is_empty() {
                    return None;
                }
                let take = unsafe { self.take_unchecked(idx.into()) };
                take.std_as_series().unpack::<Float64Type>().unwrap().get(0)
            }),
            GroupsProxy::Slice(groups) => {
                agg_helper_slice::<Float64Type, _>(groups, |[first, len]| {
                    debug_assert!(len <= self.len() as IdxSize);
                    match len {
                        0 => None,
                        1 => self.get(first as usize).map(|v| NumCast::from(v).unwrap()),
                        _ => {
                            let arr_group = slice_from_offsets(self, first, len);
                            arr_group.std()
                        }
                    }
                })
            }
        }
    }

    pub(crate) fn agg_quantile(
        &self,
        groups: &GroupsProxy,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> Series {
        match groups {
            GroupsProxy::Idx(groups) => agg_helper_idx_on_all::<Float64Type, _>(groups, |idx| {
                debug_assert!(idx.len() <= self.len());
                if idx.is_empty() {
                    return None;
                }
                let take = unsafe { self.take_unchecked(idx.into()) };
                take.quantile_as_series(quantile, interpol)
                    .unwrap()
                    .unpack::<Float64Type>()
                    .unwrap()
                    .get(0)
            }),
            GroupsProxy::Slice(groups) => {
                agg_helper_slice::<Float64Type, _>(groups, |[first, len]| {
                    debug_assert!(len <= self.len() as IdxSize);
                    match len {
                        0 => None,
                        1 => self.get(first as usize).map(|v| NumCast::from(v).unwrap()),
                        _ => {
                            let arr_group = slice_from_offsets(self, first, len);
                            arr_group.quantile(quantile, interpol).unwrap()
                        }
                    }
                })
            }
        }
    }
    pub(crate) fn agg_median(&self, groups: &GroupsProxy) -> Series {
        match groups {
            GroupsProxy::Idx(groups) => agg_helper_idx_on_all::<Float64Type, _>(groups, |idx| {
                debug_assert!(idx.len() <= self.len());
                if idx.is_empty() {
                    return None;
                }
                let take = unsafe { self.take_unchecked(idx.into()) };
                take.median_as_series()
                    .unpack::<Float64Type>()
                    .unwrap()
                    .get(0)
            }),
            GroupsProxy::Slice(groups) => {
                agg_helper_slice::<Float64Type, _>(groups, |[first, len]| {
                    debug_assert!(len <= self.len() as IdxSize);
                    match len {
                        0 => None,
                        1 => self.get(first as usize).map(|v| NumCast::from(v).unwrap()),
                        _ => {
                            let arr_group = slice_from_offsets(self, first, len);
                            arr_group.median()
                        }
                    }
                })
            }
        }
    }
}

impl<T> ChunkedArray<T> where ChunkedArray<T>: ChunkTake + IntoSeries {}
