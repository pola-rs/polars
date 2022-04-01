use crate::POOL;
use num::{Bounded, Num, NumCast, ToPrimitive, Zero};
use rayon::prelude::*;

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
fn agg_helper_idx<T, F>(groups: &GroupsIdx, f: F) -> Option<Series>
where
    F: Fn((IdxSize, &Vec<IdxSize>)) -> Option<T::Native> + Send + Sync,
    T: PolarsNumericType,
    ChunkedArray<T>: IntoSeries,
{
    let ca: ChunkedArray<T> = POOL.install(|| groups.into_par_iter().map(f).collect());
    Some(ca.into_series())
}

// helper that iterates on the `all: Vec<Vec<u32>` collection
// this doesn't have traverse the `first: Vec<u32>` memory and is therefore faster
fn agg_helper_idx_on_all<T, F>(groups: &GroupsIdx, f: F) -> Option<Series>
where
    F: Fn(&Vec<IdxSize>) -> Option<T::Native> + Send + Sync,
    T: PolarsNumericType,
    ChunkedArray<T>: IntoSeries,
{
    let ca: ChunkedArray<T> = POOL.install(|| groups.all().into_par_iter().map(f).collect());
    Some(ca.into_series())
}

fn agg_helper_slice<T, F>(groups: &[[IdxSize; 2]], f: F) -> Option<Series>
where
    F: Fn([IdxSize; 2]) -> Option<T::Native> + Send + Sync,
    T: PolarsNumericType,
    ChunkedArray<T>: IntoSeries,
{
    let ca: ChunkedArray<T> = POOL.install(|| groups.par_iter().copied().map(f).collect());
    Some(ca.into_series())
}

impl BooleanChunked {
    pub(crate) fn agg_min(&self, groups: &GroupsProxy) -> Option<Series> {
        self.cast(&IDX_DTYPE).unwrap().agg_min(groups)
    }
    pub(crate) fn agg_max(&self, groups: &GroupsProxy) -> Option<Series> {
        self.cast(&IDX_DTYPE).unwrap().agg_max(groups)
    }
    pub(crate) fn agg_sum(&self, groups: &GroupsProxy) -> Option<Series> {
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

    #[cfg(feature = "private")]
    pub fn agg_valid_count(&self, groups: &GroupsProxy) -> Option<Series> {
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

    #[cfg(feature = "private")]
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

    #[cfg(feature = "private")]
    pub fn agg_n_unique(&self, groups: &GroupsProxy) -> Option<Series> {
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

    #[cfg(feature = "private")]
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
    pub(crate) fn agg_min(&self, groups: &GroupsProxy) -> Option<Series> {
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
                                |a, b| if a < b { a } else { b },
                                T::Native::max_value(),
                            )
                        }),
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

    pub(crate) fn agg_max(&self, groups: &GroupsProxy) -> Option<Series> {
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

    pub(crate) fn agg_sum(&self, groups: &GroupsProxy) -> Option<Series> {
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
    pub(crate) fn agg_mean(&self, groups: &GroupsProxy) -> Option<Series> {
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
                                take_agg_primitive_iter_unchecked_count_nulls::<T::Native, _, _>(
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
                                let opt_sum: Option<T::Native> = take.sum();
                                opt_sum.map(|sum| sum.to_f64().unwrap() / idx.len() as f64)
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

    pub(crate) fn agg_var(&self, groups: &GroupsProxy) -> Option<Series> {
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
    pub(crate) fn agg_std(&self, groups: &GroupsProxy) -> Option<Series> {
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
    ) -> Option<Series> {
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
    pub(crate) fn agg_median(&self, groups: &GroupsProxy) -> Option<Series> {
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
    pub(crate) fn agg_mean(&self, groups: &GroupsProxy) -> Option<Series> {
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
                                    T::Native::zero(),
                                )
                            }
                            .to_f64()
                            .map(|sum| sum / idx.len() as f64),
                            (_, 1) => unsafe {
                                take_agg_primitive_iter_unchecked_count_nulls::<T::Native, _, _>(
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
                                let opt_sum: Option<T::Native> = take.sum();
                                opt_sum.map(|sum| sum.to_f64().unwrap() / idx.len() as f64)
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

    pub(crate) fn agg_var(&self, groups: &GroupsProxy) -> Option<Series> {
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
    pub(crate) fn agg_std(&self, groups: &GroupsProxy) -> Option<Series> {
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
    ) -> Option<Series> {
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
    pub(crate) fn agg_median(&self, groups: &GroupsProxy) -> Option<Series> {
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

pub trait AggList {
    fn agg_list(&self, _groups: &GroupsProxy) -> Option<Series> {
        None
    }
}

impl<T> AggList for ChunkedArray<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: IntoSeries,
{
    fn agg_list(&self, groups: &GroupsProxy) -> Option<Series> {
        match groups {
            GroupsProxy::Idx(groups) => {
                let mut can_fast_explode = true;
                let arr = match self.cont_slice() {
                    Ok(values) => {
                        let mut offsets = Vec::<i64>::with_capacity(groups.len() + 1);
                        let mut length_so_far = 0i64;
                        offsets.push(length_so_far);

                        let mut list_values = Vec::<T::Native>::with_capacity(self.len());
                        groups.iter().for_each(|(_, idx)| {
                            let idx_len = idx.len();
                            if idx_len == 0 {
                                can_fast_explode = false;
                            }

                            length_so_far += idx_len as i64;
                            // Safety:
                            // group tuples are in bounds
                            unsafe {
                                list_values.extend(idx.iter().map(|idx| {
                                    debug_assert!((*idx as usize) < values.len());
                                    *values.get_unchecked(*idx as usize)
                                }));
                                // Safety:
                                // we know that offsets has allocated enough slots
                                offsets.push_unchecked(length_so_far);
                            }
                        });
                        let array = PrimitiveArray::from_data(
                            T::get_dtype().to_arrow(),
                            list_values.into(),
                            None,
                        );
                        let data_type =
                            ListArray::<i64>::default_datatype(T::get_dtype().to_arrow());
                        ListArray::<i64>::from_data(
                            data_type,
                            offsets.into(),
                            Arc::new(array),
                            None,
                        )
                    }
                    _ => {
                        let mut builder = ListPrimitiveChunkedBuilder::<T::Native>::new(
                            self.name(),
                            groups.len(),
                            self.len(),
                            self.dtype().clone(),
                        );
                        for idx in groups.all().iter() {
                            let s = unsafe { self.take_unchecked(idx.into()).into_series() };
                            builder.append_series(&s);
                        }
                        return Some(builder.finish().into_series());
                    }
                };
                let mut ca = ListChunked::from_chunks(self.name(), vec![Arc::new(arr)]);
                if can_fast_explode {
                    ca.set_fast_explode()
                }
                Some(ca.into())
            }
            GroupsProxy::Slice(groups) => {
                let mut can_fast_explode = true;
                let arr = match self.cont_slice() {
                    Ok(values) => {
                        let mut offsets = Vec::<i64>::with_capacity(groups.len() + 1);
                        let mut length_so_far = 0i64;
                        offsets.push(length_so_far);

                        let mut list_values = Vec::<T::Native>::with_capacity(self.len());
                        groups.iter().for_each(|&[first, len]| {
                            if len == 0 {
                                can_fast_explode = false;
                            }

                            length_so_far += len as i64;
                            list_values
                                .extend_from_slice(&values[first as usize..(first + len) as usize]);
                            unsafe {
                                // Safety:
                                // we know that offsets has allocated enough slots
                                offsets.push_unchecked(length_so_far);
                            }
                        });
                        let array = PrimitiveArray::from_data(
                            T::get_dtype().to_arrow(),
                            list_values.into(),
                            None,
                        );
                        let data_type =
                            ListArray::<i64>::default_datatype(T::get_dtype().to_arrow());
                        ListArray::<i64>::from_data(
                            data_type,
                            offsets.into(),
                            Arc::new(array),
                            None,
                        )
                    }
                    _ => {
                        let mut builder = ListPrimitiveChunkedBuilder::<T::Native>::new(
                            self.name(),
                            groups.len(),
                            self.len(),
                            self.dtype().clone(),
                        );
                        for &[first, len] in groups {
                            let s = self.slice(first as i64, len as usize).into_series();
                            builder.append_series(&s);
                        }
                        return Some(builder.finish().into_series());
                    }
                };
                let mut ca = ListChunked::from_chunks(self.name(), vec![Arc::new(arr)]);
                if can_fast_explode {
                    ca.set_fast_explode()
                }
                Some(ca.into())
            }
        }
    }
}

impl AggList for BooleanChunked {
    fn agg_list(&self, groups: &GroupsProxy) -> Option<Series> {
        match groups {
            GroupsProxy::Idx(groups) => {
                let mut builder =
                    ListBooleanChunkedBuilder::new(self.name(), groups.len(), self.len());
                for idx in groups.all().iter() {
                    let ca = unsafe { self.take_unchecked(idx.into()) };
                    builder.append(&ca)
                }
                Some(builder.finish().into_series())
            }
            GroupsProxy::Slice(groups) => {
                let mut builder =
                    ListBooleanChunkedBuilder::new(self.name(), groups.len(), self.len());
                for [first, len] in groups {
                    let ca = self.slice(*first as i64, *len as usize);
                    builder.append(&ca)
                }
                Some(builder.finish().into_series())
            }
        }
    }
}

impl AggList for Utf8Chunked {
    fn agg_list(&self, groups: &GroupsProxy) -> Option<Series> {
        match groups {
            GroupsProxy::Idx(groups) => {
                let mut builder =
                    ListUtf8ChunkedBuilder::new(self.name(), groups.len(), self.len());
                for idx in groups.all().iter() {
                    let ca = unsafe { self.take_unchecked(idx.into()) };
                    builder.append(&ca)
                }
                Some(builder.finish().into_series())
            }
            GroupsProxy::Slice(groups) => {
                let mut builder =
                    ListUtf8ChunkedBuilder::new(self.name(), groups.len(), self.len());
                for [first, len] in groups {
                    let ca = self.slice(*first as i64, *len as usize);
                    builder.append(&ca)
                }
                Some(builder.finish().into_series())
            }
        }
    }
}

fn agg_list_list<F: Fn(&ListChunked, bool, &mut Vec<i64>, &mut i64, &mut Vec<ArrayRef>) -> bool>(
    ca: &ListChunked,
    groups_len: usize,
    func: F,
) -> Option<Series> {
    let can_fast_explode = true;
    let mut offsets = Vec::<i64>::with_capacity(groups_len + 1);
    let mut length_so_far = 0i64;
    offsets.push(length_so_far);

    let mut list_values = Vec::with_capacity(groups_len);

    let can_fast_explode = func(
        ca,
        can_fast_explode,
        &mut offsets,
        &mut length_so_far,
        &mut list_values,
    );
    if groups_len == 0 {
        list_values.push(ca.chunks[0].slice(0, 0).into())
    }
    let arrays = list_values.iter().map(|arr| &**arr).collect::<Vec<_>>();
    let list_values: ArrayRef = arrow::compute::concatenate::concatenate(&arrays)
        .unwrap()
        .into();
    let data_type = ListArray::<i64>::default_datatype(list_values.data_type().clone());
    let arr = Arc::new(ListArray::<i64>::from_data(
        data_type,
        offsets.into(),
        list_values,
        None,
    )) as ArrayRef;
    let mut listarr = ListChunked::from_chunks(ca.name(), vec![arr]);
    if can_fast_explode {
        listarr.set_fast_explode()
    }
    Some(listarr.into_series())
}

impl AggList for ListChunked {
    fn agg_list(&self, groups: &GroupsProxy) -> Option<Series> {
        match groups {
            GroupsProxy::Idx(groups) => {
                let func = |ca: &ListChunked,
                            mut can_fast_explode: bool,
                            offsets: &mut Vec<i64>,
                            length_so_far: &mut i64,
                            list_values: &mut Vec<ArrayRef>| {
                    groups.iter().for_each(|(_, idx)| {
                        let idx_len = idx.len();
                        if idx_len == 0 {
                            can_fast_explode = false;
                        }

                        *length_so_far += idx_len as i64;
                        // Safety:
                        // group tuples are in bounds
                        unsafe {
                            let mut s = ca.take_unchecked(idx.into());
                            let arr = s.chunks.pop().unwrap();
                            list_values.push(arr);

                            // Safety:
                            // we know that offsets has allocated enough slots
                            offsets.push_unchecked(*length_so_far);
                        }
                    });
                    can_fast_explode
                };

                agg_list_list(self, groups.len(), func)
            }
            GroupsProxy::Slice(groups) => {
                let func = |ca: &ListChunked,
                            mut can_fast_explode: bool,
                            offsets: &mut Vec<i64>,
                            length_so_far: &mut i64,
                            list_values: &mut Vec<ArrayRef>| {
                    groups.iter().for_each(|&[first, len]| {
                        if len == 0 {
                            can_fast_explode = false;
                        }

                        *length_so_far += len as i64;
                        let mut s = ca.slice(first as i64, len as usize);
                        let arr = s.chunks.pop().unwrap();
                        list_values.push(arr);

                        unsafe {
                            // Safety:
                            // we know that offsets has allocated enough slots
                            offsets.push_unchecked(*length_so_far);
                        }
                    });
                    can_fast_explode
                };

                agg_list_list(self, groups.len(), func)
            }
        }
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> AggList for ObjectChunked<T> {
    fn agg_list(&self, groups: &GroupsProxy) -> Option<Series> {
        let mut can_fast_explode = true;
        let mut offsets = Vec::<i64>::with_capacity(groups.len() + 1);
        let mut length_so_far = 0i64;
        offsets.push(length_so_far);

        //  we know that iterators length
        let iter = unsafe {
            groups
                .iter()
                .flat_map(|indicator| {
                    let (group_vals, len) = match indicator {
                        GroupsIndicator::Idx((_first, idx)) => {
                            // Safety:
                            // group tuples always in bounds
                            let group_vals = self.take_unchecked(idx.into());

                            (group_vals, idx.len() as IdxSize)
                        }
                        GroupsIndicator::Slice([first, len]) => {
                            let group_vals = slice_from_offsets(self, first, len);

                            (group_vals, len)
                        }
                    };

                    if len == 0 {
                        can_fast_explode = false;
                    }
                    length_so_far += len as i64;
                    // Safety:
                    // we know that offsets has allocated enough slots
                    offsets.push_unchecked(length_so_far);

                    let arr = group_vals.downcast_iter().next().unwrap().clone();
                    arr.into_iter_cloned()
                })
                .trust_my_length(self.len())
        };

        let mut pe = create_extension(iter);

        // Safety:
        // this is safe because we just created the PolarsExtension
        // meaning that the sentinel is heap allocated and the dereference of the
        // pointer does not fail
        unsafe { pe.set_to_series_fn::<T>() };
        let extension_array = Arc::new(pe.take_and_forget()) as ArrayRef;
        let extension_dtype = extension_array.data_type();

        let data_type = ListArray::<i64>::default_datatype(extension_dtype.clone());
        let arr = Arc::new(ListArray::<i64>::from_data(
            data_type,
            offsets.into(),
            extension_array,
            None,
        )) as ArrayRef;

        let mut listarr = ListChunked::from_chunks(self.name(), vec![arr]);
        if can_fast_explode {
            listarr.set_fast_explode()
        }
        Some(listarr.into_series())
    }
}
