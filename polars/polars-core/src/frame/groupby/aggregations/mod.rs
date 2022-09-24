mod agg_list;

pub use agg_list::*;
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::types::simd::Simd;
use arrow::types::NativeType;
use num::{Bounded, Num, NumCast, ToPrimitive, Zero};
use polars_arrow::data_types::IsFloat;
use polars_arrow::kernels::rolling;
use polars_arrow::kernels::rolling::no_nulls::{
    MaxWindow, MeanWindow, MinWindow, RollingAggWindowNoNulls, StdWindow, SumWindow, VarWindow,
};
use polars_arrow::kernels::rolling::nulls::RollingAggWindowNulls;
use polars_arrow::kernels::take_agg::*;
use polars_arrow::prelude::QuantileInterpolOptions;
use polars_arrow::trusted_len::PushUnchecked;
use rayon::prelude::*;

#[cfg(feature = "object")]
use crate::chunked_array::object::extension::create_extension;
use crate::frame::groupby::GroupsIdx;
#[cfg(feature = "object")]
use crate::frame::groupby::GroupsIndicator;
use crate::prelude::*;
use crate::series::implementations::SeriesWrap;
use crate::series::IsSorted;
use crate::{apply_method_physical_integer, POOL};

// if the windows overlap, we can use the rolling_<agg> kernels
// they maintain state, which saves a lot of compute by not naively traversing all elements every
// window
//
// if the windows don't overlap, we should not use these kernels as they are single threaded, so
// we miss out on easy parallelization.
pub fn _use_rolling_kernels(groups: &GroupsSlice, chunks: &[ArrayRef]) -> bool {
    match groups.len() {
        0 | 1 => false,
        _ => {
            let [first_offset, first_len] = groups[0];
            let second_offset = groups[1][0];

            second_offset < (first_offset + first_len) && chunks.len() == 1
        }
    }
}

// Use an aggregation window that maintains the state
pub fn _rolling_apply_agg_window_nulls<'a, Agg, T, O>(
    values: &'a [T],
    validity: &'a Bitmap,
    offsets: O,
) -> ArrayRef
where
    O: Iterator<Item = (IdxSize, IdxSize)> + TrustedLen,
    Agg: RollingAggWindowNulls<'a, T>,
    T: IsFloat + NativeType,
{
    if values.is_empty() {
        let out: Vec<T> = vec![];
        return Box::new(PrimitiveArray::from_data(
            T::PRIMITIVE.into(),
            out.into(),
            None,
        ));
    }

    // This iterators length can be trusted
    // these represent the number of groups in the groupby operation
    let output_len = offsets.size_hint().0;
    // start with a dummy index, will be overwritten on first iteration.
    // Safety:
    // we are in bounds
    let mut agg_window = unsafe { Agg::new(values, validity, 0, 0) };

    let mut validity = MutableBitmap::with_capacity(output_len);
    validity.extend_constant(output_len, true);

    let out = offsets
        .enumerate()
        .map(|(idx, (start, len))| {
            let end = start + len;

            // safety:
            // we are in bounds

            let agg = if start == end {
                None
            } else {
                unsafe { agg_window.update(start as usize, end as usize) }
            };

            match agg {
                Some(val) => val,
                None => {
                    // safety: we are in bounds
                    unsafe { validity.set_unchecked(idx, false) };
                    T::default()
                }
            }
        })
        .collect_trusted::<Vec<_>>();

    Box::new(PrimitiveArray::from_data(
        T::PRIMITIVE.into(),
        out.into(),
        Some(validity.into()),
    ))
}

// Use an aggregation window that maintains the state
pub fn _rolling_apply_agg_window_no_nulls<'a, Agg, T, O>(values: &'a [T], offsets: O) -> ArrayRef
where
    // items (offset, len) -> so offsets are offset, offset + len
    Agg: RollingAggWindowNoNulls<'a, T>,
    O: Iterator<Item = (IdxSize, IdxSize)> + TrustedLen,
    T: IsFloat + NativeType,
{
    if values.is_empty() {
        let out: Vec<T> = vec![];
        return Box::new(PrimitiveArray::from_data(
            T::PRIMITIVE.into(),
            out.into(),
            None,
        ));
    }
    // start with a dummy index, will be overwritten on first iteration.
    let mut agg_window = Agg::new(values, 0, 0);

    let out = offsets
        .map(|(start, len)| {
            let end = start + len;

            if start == end {
                None
            } else {
                // safety:
                // we are in bounds
                Some(unsafe { agg_window.update(start as usize, end as usize) })
            }
        })
        .collect::<PrimitiveArray<T>>();

    Box::new(out)
}

pub fn _slice_from_offsets<T>(ca: &ChunkedArray<T>, first: IdxSize, len: IdxSize) -> ChunkedArray<T>
where
    T: PolarsDataType,
{
    ca.slice(first as i64, len as usize)
}

// helper that combines the groups into a parallel iterator over `(first, all): (u32, &Vec<u32>)`
pub fn _agg_helper_idx<T, F>(groups: &GroupsIdx, f: F) -> Series
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

pub fn _agg_helper_slice<T, F>(groups: &[[IdxSize; 2]], f: F) -> Series
where
    F: Fn([IdxSize; 2]) -> Option<T::Native> + Send + Sync,
    T: PolarsNumericType,
    ChunkedArray<T>: IntoSeries,
{
    let ca: ChunkedArray<T> = POOL.install(|| groups.par_iter().copied().map(f).collect());
    ca.into_series()
}

impl BooleanChunked {
    pub(crate) unsafe fn agg_min(&self, groups: &GroupsProxy) -> Series {
        self.cast(&IDX_DTYPE).unwrap().agg_min(groups)
    }
    pub(crate) unsafe fn agg_max(&self, groups: &GroupsProxy) -> Series {
        self.cast(&IDX_DTYPE).unwrap().agg_max(groups)
    }
    pub(crate) unsafe fn agg_sum(&self, groups: &GroupsProxy) -> Series {
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
            GroupsProxy::Slice { groups, .. } => {
                _agg_helper_slice::<IdxType, _>(groups, |[first, len]| {
                    debug_assert!(len <= self.len() as IdxSize);
                    if len == 0 {
                        None
                    } else if !self.has_validity() {
                        Some(len)
                    } else {
                        let take = self.slice_from_offsets(first, len);
                        Some((take.len() - take.null_count()) as IdxSize)
                    }
                })
            }
        }
    }

    #[doc(hidden)]
    pub unsafe fn agg_first(&self, groups: &GroupsProxy) -> Series {
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
                self.take_opt_iter_unchecked(&mut iter)
            }
            GroupsProxy::Slice { groups, .. } => {
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
                self.take_opt_iter_unchecked(&mut iter)
            }
        };
        self.restore_logical(out)
    }

    #[doc(hidden)]
    pub unsafe fn agg_n_unique(&self, groups: &GroupsProxy) -> Series {
        match groups {
            GroupsProxy::Idx(groups) => agg_helper_idx_on_all::<IdxType, _>(groups, |idx| {
                debug_assert!(idx.len() <= self.len());
                if idx.is_empty() {
                    None
                } else {
                    let take = self.take_iter_unchecked(&mut idx.iter().map(|i| *i as usize));
                    take.n_unique().ok().map(|v| v as IdxSize)
                }
            }),
            GroupsProxy::Slice { groups, .. } => {
                _agg_helper_slice::<IdxType, _>(groups, |[first, len]| {
                    debug_assert!(len <= self.len() as IdxSize);
                    if len == 0 {
                        None
                    } else {
                        let take = self.slice_from_offsets(first, len);
                        take.n_unique().ok().map(|v| v as IdxSize)
                    }
                })
            }
        }
    }

    #[doc(hidden)]
    pub unsafe fn agg_median(&self, groups: &GroupsProxy) -> Series {
        use DataType::*;

        match self.dtype() {
            Float32 => SeriesWrap(self.f32().unwrap().clone()).agg_median(groups),
            Float64 => SeriesWrap(self.f64().unwrap().clone()).agg_median(groups),
            dt if dt.is_numeric() || dt.is_temporal() => {
                let ca = self.to_physical_repr();
                let physical_type = ca.dtype();
                let s = apply_method_physical_integer!(ca, agg_median, groups);
                if dt.is_logical() {
                    // back to physical and then
                    // back to logical type
                    s.cast(physical_type).unwrap().cast(dt).unwrap()
                } else {
                    s
                }
            }
            _ => Series::full_null("", groups.len(), self.dtype()),
        }
    }

    #[doc(hidden)]
    pub unsafe fn agg_quantile(
        &self,
        groups: &GroupsProxy,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> Series {
        use DataType::*;

        match self.dtype() {
            Float32 => {
                SeriesWrap(self.f32().unwrap().clone()).agg_quantile(groups, quantile, interpol)
            }
            Float64 => {
                SeriesWrap(self.f64().unwrap().clone()).agg_quantile(groups, quantile, interpol)
            }
            dt if dt.is_numeric() || dt.is_temporal() => {
                let ca = self.to_physical_repr();
                let physical_type = ca.dtype();
                let s =
                    apply_method_physical_integer!(ca, agg_quantile, groups, quantile, interpol);
                if dt.is_logical() {
                    // back to physical and then
                    // back to logical type
                    s.cast(physical_type).unwrap().cast(dt).unwrap()
                } else {
                    s
                }
            }
            _ => Series::full_null("", groups.len(), self.dtype()),
        }
    }

    #[doc(hidden)]
    pub unsafe fn agg_mean(&self, groups: &GroupsProxy) -> Series {
        use DataType::*;

        match self.dtype() {
            Float32 => SeriesWrap(self.f32().unwrap().clone()).agg_mean(groups),
            Float64 => SeriesWrap(self.f64().unwrap().clone()).agg_mean(groups),
            dt if dt.is_numeric() => {
                apply_method_physical_integer!(self, agg_mean, groups)
            }
            dt @ Duration(_) => {
                let s = self.to_physical_repr();
                // agg_mean returns Float64
                let out = s.agg_mean(groups);
                // cast back to Int64 and then to logical duration type
                out.cast(&Int64).unwrap().cast(dt).unwrap()
            }
            _ => Series::full_null("", groups.len(), self.dtype()),
        }
    }

    #[doc(hidden)]
    pub unsafe fn agg_last(&self, groups: &GroupsProxy) -> Series {
        let out = match groups {
            GroupsProxy::Idx(groups) => {
                let mut iter = groups.all().iter().map(|idx| {
                    if idx.is_empty() {
                        None
                    } else {
                        Some(idx[idx.len() - 1] as usize)
                    }
                });
                self.take_opt_iter_unchecked(&mut iter)
            }
            GroupsProxy::Slice { groups, .. } => {
                let mut iter = groups.iter().map(|&[first, len]| {
                    if len == 0 {
                        None
                    } else {
                        Some((first + len - 1) as usize)
                    }
                });
                self.take_opt_iter_unchecked(&mut iter)
            }
        };
        self.restore_logical(out)
    }
}

#[inline(always)]
fn take_min<T: PartialOrd>(a: T, b: T) -> T {
    if a < b {
        a
    } else {
        b
    }
}

#[inline(always)]
fn take_max<T: PartialOrd>(a: T, b: T) -> T {
    if a > b {
        a
    } else {
        b
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
    pub(crate) unsafe fn agg_min(&self, groups: &GroupsProxy) -> Series {
        // faster paths
        match (self.is_sorted2(), self.null_count()) {
            (IsSorted::Ascending, 0) => {
                return self.clone().into_series().agg_first(groups);
            }
            (IsSorted::Descending, 0) => {
                return self.clone().into_series().agg_last(groups);
            }
            _ => {}
        }
        match groups {
            GroupsProxy::Idx(groups) => _agg_helper_idx::<T, _>(groups, |(first, idx)| {
                debug_assert!(idx.len() <= self.len());
                if idx.is_empty() {
                    None
                } else if idx.len() == 1 {
                    self.get(first as usize)
                } else {
                    match (self.has_validity(), self.chunks.len()) {
                        (false, 1) => Some(take_agg_no_null_primitive_iter_unchecked(
                            self.downcast_iter().next().unwrap(),
                            idx.iter().map(|i| *i as usize),
                            take_min,
                            T::Native::max_value(),
                        )),
                        (_, 1) => take_agg_primitive_iter_unchecked::<T::Native, _, _>(
                            self.downcast_iter().next().unwrap(),
                            idx.iter().map(|i| *i as usize),
                            take_min,
                            T::Native::max_value(),
                            idx.len() as IdxSize,
                        ),
                        _ => {
                            let take = { self.take_unchecked(idx.into()) };
                            take.min()
                        }
                    }
                }
            }),
            GroupsProxy::Slice {
                groups: groups_slice,
                ..
            } => {
                if _use_rolling_kernels(groups_slice, self.chunks()) {
                    let arr = self.downcast_iter().next().unwrap();
                    let values = arr.values().as_slice();
                    let offset_iter = groups_slice.iter().map(|[first, len]| (*first, *len));
                    let arr = match arr.validity() {
                        None => _rolling_apply_agg_window_no_nulls::<MinWindow<_>, _, _>(
                            values,
                            offset_iter,
                        ),
                        Some(validity) => _rolling_apply_agg_window_nulls::<
                            rolling::nulls::MinWindow<_>,
                            _,
                            _,
                        >(values, validity, offset_iter),
                    };
                    Self::from_chunks("", vec![arr]).into_series()
                } else {
                    _agg_helper_slice::<T, _>(groups_slice, |[first, len]| {
                        debug_assert!(len <= self.len() as IdxSize);
                        match len {
                            0 => None,
                            1 => self.get(first as usize),
                            _ => {
                                let arr_group = _slice_from_offsets(self, first, len);
                                arr_group.min()
                            }
                        }
                    })
                }
            }
        }
    }

    pub(crate) unsafe fn agg_max(&self, groups: &GroupsProxy) -> Series {
        // faster paths
        match (self.is_sorted2(), self.null_count()) {
            (IsSorted::Ascending, 0) => {
                return self.clone().into_series().agg_last(groups);
            }
            (IsSorted::Descending, 0) => {
                return self.clone().into_series().agg_first(groups);
            }
            _ => {}
        }

        match groups {
            GroupsProxy::Idx(groups) => _agg_helper_idx::<T, _>(groups, |(first, idx)| {
                debug_assert!(idx.len() <= self.len());
                if idx.is_empty() {
                    None
                } else if idx.len() == 1 {
                    self.get(first as usize)
                } else {
                    match (self.has_validity(), self.chunks.len()) {
                        (false, 1) => Some({
                            take_agg_no_null_primitive_iter_unchecked(
                                self.downcast_iter().next().unwrap(),
                                idx.iter().map(|i| *i as usize),
                                take_max,
                                T::Native::min_value(),
                            )
                        }),
                        (_, 1) => take_agg_primitive_iter_unchecked::<T::Native, _, _>(
                            self.downcast_iter().next().unwrap(),
                            idx.iter().map(|i| *i as usize),
                            take_max,
                            T::Native::min_value(),
                            idx.len() as IdxSize,
                        ),
                        _ => {
                            let take = { self.take_unchecked(idx.into()) };
                            take.max()
                        }
                    }
                }
            }),
            GroupsProxy::Slice {
                groups: groups_slice,
                ..
            } => {
                if _use_rolling_kernels(groups_slice, self.chunks()) {
                    let arr = self.downcast_iter().next().unwrap();
                    let values = arr.values().as_slice();
                    let offset_iter = groups_slice.iter().map(|[first, len]| (*first, *len));
                    let arr = match arr.validity() {
                        None => _rolling_apply_agg_window_no_nulls::<MaxWindow<_>, _, _>(
                            values,
                            offset_iter,
                        ),
                        Some(validity) => _rolling_apply_agg_window_nulls::<
                            rolling::nulls::MaxWindow<_>,
                            _,
                            _,
                        >(values, validity, offset_iter),
                    };
                    Self::from_chunks("", vec![arr]).into_series()
                } else {
                    _agg_helper_slice::<T, _>(groups_slice, |[first, len]| {
                        debug_assert!(len <= self.len() as IdxSize);
                        match len {
                            0 => None,
                            1 => self.get(first as usize),
                            _ => {
                                let arr_group = _slice_from_offsets(self, first, len);
                                arr_group.max()
                            }
                        }
                    })
                }
            }
        }
    }

    pub(crate) unsafe fn agg_sum(&self, groups: &GroupsProxy) -> Series {
        match groups {
            GroupsProxy::Idx(groups) => _agg_helper_idx::<T, _>(groups, |(first, idx)| {
                debug_assert!(idx.len() <= self.len());
                if idx.is_empty() {
                    None
                } else if idx.len() == 1 {
                    self.get(first as usize)
                } else {
                    match (self.has_validity(), self.chunks.len()) {
                        (false, 1) => Some({
                            take_agg_no_null_primitive_iter_unchecked(
                                self.downcast_iter().next().unwrap(),
                                idx.iter().map(|i| *i as usize),
                                |a, b| a + b,
                                T::Native::zero(),
                            )
                        }),
                        (_, 1) => take_agg_primitive_iter_unchecked::<T::Native, _, _>(
                            self.downcast_iter().next().unwrap(),
                            idx.iter().map(|i| *i as usize),
                            |a, b| a + b,
                            T::Native::zero(),
                            idx.len() as IdxSize,
                        ),
                        _ => {
                            let take = { self.take_unchecked(idx.into()) };
                            take.sum()
                        }
                    }
                }
            }),
            GroupsProxy::Slice { groups, .. } => {
                if _use_rolling_kernels(groups, self.chunks()) {
                    let arr = self.downcast_iter().next().unwrap();
                    let values = arr.values().as_slice();
                    let offset_iter = groups.iter().map(|[first, len]| (*first, *len));
                    let arr = match arr.validity() {
                        None => _rolling_apply_agg_window_no_nulls::<SumWindow<_>, _, _>(
                            values,
                            offset_iter,
                        ),
                        Some(validity) => _rolling_apply_agg_window_nulls::<
                            rolling::nulls::SumWindow<_>,
                            _,
                            _,
                        >(values, validity, offset_iter),
                    };
                    Self::from_chunks("", vec![arr]).into_series()
                } else {
                    _agg_helper_slice::<T, _>(groups, |[first, len]| {
                        debug_assert!(len <= self.len() as IdxSize);
                        match len {
                            0 => None,
                            1 => self.get(first as usize),
                            _ => {
                                let arr_group = _slice_from_offsets(self, first, len);
                                arr_group.sum()
                            }
                        }
                    })
                }
            }
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
    T::Native: Simd + NumericNative + num::pow::Pow<T::Native, Output = T::Native>,
    <T::Native as Simd>::Simd: std::ops::Add<Output = <T::Native as Simd>::Simd>
        + arrow::compute::aggregate::Sum<T::Native>
        + arrow::compute::aggregate::SimdOrd<T::Native>,
{
    pub(crate) unsafe fn agg_mean(&self, groups: &GroupsProxy) -> Series {
        match groups {
            GroupsProxy::Idx(groups) => {
                _agg_helper_idx::<T, _>(groups, |(first, idx)| {
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
                            (false, 1) => {
                                take_agg_no_null_primitive_iter_unchecked(
                                    self.downcast_iter().next().unwrap(),
                                    idx.iter().map(|i| *i as usize),
                                    |a, b| a + b,
                                    T::Native::zero(),
                                )
                            }
                            .to_f64()
                            .map(|sum| sum / idx.len() as f64),
                            (_, 1) => {
                                take_agg_primitive_iter_unchecked_count_nulls::<T::Native, _, _, _>(
                                    self.downcast_iter().next().unwrap(),
                                    idx.iter().map(|i| *i as usize),
                                    |a, b| a + b,
                                    T::Native::zero(),
                                    idx.len() as IdxSize,
                                )
                            }
                            .map(|(sum, null_count)| {
                                sum.to_f64()
                                    .map(|sum| sum / (idx.len() as f64 - null_count as f64))
                                    .unwrap()
                            }),
                            _ => {
                                let take = { self.take_unchecked(idx.into()) };
                                take.mean()
                            }
                        }
                    };
                    out.map(|flt| NumCast::from(flt).unwrap())
                })
            }
            GroupsProxy::Slice { groups, .. } => {
                if _use_rolling_kernels(groups, self.chunks()) {
                    let arr = self.downcast_iter().next().unwrap();
                    let values = arr.values().as_slice();
                    let offset_iter = groups.iter().map(|[first, len]| (*first, *len));
                    let arr = match arr.validity() {
                        None => _rolling_apply_agg_window_no_nulls::<MeanWindow<_>, _, _>(
                            values,
                            offset_iter,
                        ),
                        Some(validity) => _rolling_apply_agg_window_nulls::<
                            rolling::nulls::MeanWindow<_>,
                            _,
                            _,
                        >(values, validity, offset_iter),
                    };
                    ChunkedArray::<T>::from_chunks("", vec![arr]).into_series()
                } else {
                    _agg_helper_slice::<T, _>(groups, |[first, len]| {
                        debug_assert!(len <= self.len() as IdxSize);
                        match len {
                            0 => None,
                            1 => self.get(first as usize),
                            _ => {
                                let arr_group = _slice_from_offsets(self, first, len);
                                arr_group.mean().map(|flt| NumCast::from(flt).unwrap())
                            }
                        }
                    })
                }
            }
        }
    }

    pub(crate) unsafe fn agg_var(&self, groups: &GroupsProxy, ddof: u8) -> Series {
        let ca = &self.0;
        match groups {
            GroupsProxy::Idx(groups) => agg_helper_idx_on_all::<T, _>(groups, |idx| {
                debug_assert!(idx.len() <= ca.len());
                if idx.is_empty() {
                    return None;
                }
                let take = { ca.take_unchecked(idx.into()) };
                take.var_as_series(ddof).unpack::<T>().unwrap().get(0)
            }),
            GroupsProxy::Slice { groups, .. } => {
                if _use_rolling_kernels(groups, self.chunks()) {
                    let arr = self.downcast_iter().next().unwrap();
                    let values = arr.values().as_slice();
                    let offset_iter = groups.iter().map(|[first, len]| (*first, *len));
                    let arr = match arr.validity() {
                        None => _rolling_apply_agg_window_no_nulls::<VarWindow<_>, _, _>(
                            values,
                            offset_iter,
                        ),
                        Some(validity) => _rolling_apply_agg_window_nulls::<
                            rolling::nulls::VarWindow<_>,
                            _,
                            _,
                        >(values, validity, offset_iter),
                    };
                    ChunkedArray::<T>::from_chunks("", vec![arr]).into_series()
                } else {
                    _agg_helper_slice::<T, _>(groups, |[first, len]| {
                        debug_assert!(len <= self.len() as IdxSize);
                        match len {
                            0 => None,
                            1 => NumCast::from(0),
                            _ => {
                                let arr_group = _slice_from_offsets(self, first, len);
                                arr_group.var(ddof).map(|flt| NumCast::from(flt).unwrap())
                            }
                        }
                    })
                }
            }
        }
    }
    pub(crate) unsafe fn agg_std(&self, groups: &GroupsProxy, ddof: u8) -> Series {
        let ca = &self.0;
        match groups {
            GroupsProxy::Idx(groups) => agg_helper_idx_on_all::<T, _>(groups, |idx| {
                debug_assert!(idx.len() <= ca.len());
                if idx.is_empty() {
                    return None;
                }
                let take = { ca.take_unchecked(idx.into()) };
                take.std_as_series(ddof).unpack::<T>().unwrap().get(0)
            }),
            GroupsProxy::Slice { groups, .. } => {
                if _use_rolling_kernels(groups, self.chunks()) {
                    let arr = self.downcast_iter().next().unwrap();
                    let values = arr.values().as_slice();
                    let offset_iter = groups.iter().map(|[first, len]| (*first, *len));
                    let arr = match arr.validity() {
                        None => _rolling_apply_agg_window_no_nulls::<StdWindow<_>, _, _>(
                            values,
                            offset_iter,
                        ),
                        Some(validity) => _rolling_apply_agg_window_nulls::<
                            rolling::nulls::StdWindow<_>,
                            _,
                            _,
                        >(values, validity, offset_iter),
                    };
                    ChunkedArray::<T>::from_chunks("", vec![arr]).into_series()
                } else {
                    _agg_helper_slice::<T, _>(groups, |[first, len]| {
                        debug_assert!(len <= self.len() as IdxSize);
                        match len {
                            0 => None,
                            1 => NumCast::from(0),
                            _ => {
                                let arr_group = _slice_from_offsets(self, first, len);
                                arr_group.std(ddof).map(|flt| NumCast::from(flt).unwrap())
                            }
                        }
                    })
                }
            }
        }
    }

    pub(crate) unsafe fn agg_quantile(
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
                let take = { ca.take_unchecked(idx.into()) };
                take.quantile_as_series(quantile, interpol)
                    .unwrap() // checked with invalid quantile check
                    .unpack::<T>()
                    .unwrap()
                    .get(0)
            }),
            GroupsProxy::Slice { groups, .. } => {
                if _use_rolling_kernels(groups, self.chunks()) {
                    let arr = self.downcast_iter().next().unwrap();
                    let values = arr.values().as_slice();
                    let offset_iter = groups.iter().map(|[first, len]| (*first, *len));
                    let arr = match arr.validity() {
                        None => rolling::no_nulls::rolling_quantile_by_iter(
                            values,
                            quantile,
                            interpol,
                            offset_iter,
                        ),
                        Some(validity) => rolling::nulls::rolling_quantile_by_iter(
                            values,
                            validity,
                            quantile,
                            interpol,
                            offset_iter,
                        ),
                    };
                    ChunkedArray::<T>::from_chunks("", vec![arr]).into_series()
                } else {
                    _agg_helper_slice::<T, _>(groups, |[first, len]| {
                        debug_assert!(len <= self.len() as IdxSize);
                        match len {
                            0 => None,
                            1 => self.get(first as usize),
                            _ => {
                                let arr_group = _slice_from_offsets(self, first, len);
                                // unwrap checked with invalid quantile check
                                arr_group
                                    .quantile(quantile, interpol)
                                    .unwrap()
                                    .map(|flt| NumCast::from(flt).unwrap())
                            }
                        }
                    })
                }
            }
        }
    }
    pub(crate) unsafe fn agg_median(&self, groups: &GroupsProxy) -> Series {
        let ca = &self.0;
        match groups {
            GroupsProxy::Idx(groups) => agg_helper_idx_on_all::<T, _>(groups, |idx| {
                debug_assert!(idx.len() <= ca.len());
                if idx.is_empty() {
                    return None;
                }
                let take = { ca.take_unchecked(idx.into()) };
                take.median_as_series().unpack::<T>().unwrap().get(0)
            }),
            GroupsProxy::Slice { .. } => {
                self.agg_quantile(groups, 0.5, QuantileInterpolOptions::Linear)
            }
        }
    }
}

impl<T> ChunkedArray<T>
where
    T: PolarsIntegerType,
    ChunkedArray<T>: IntoSeries,
    T::Native: NumericNative + Ord,
    <T::Native as Simd>::Simd: std::ops::Add<Output = <T::Native as Simd>::Simd>
        + arrow::compute::aggregate::Sum<T::Native>
        + arrow::compute::aggregate::SimdOrd<T::Native>,
{
    pub(crate) unsafe fn agg_mean(&self, groups: &GroupsProxy) -> Series {
        match groups {
            GroupsProxy::Idx(groups) => {
                _agg_helper_idx::<Float64Type, _>(groups, |(first, idx)| {
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
                            (false, 1) => {
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
                                {
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
                                        idx.len() as IdxSize,
                                    )
                                }
                                .map(|(sum, null_count)| {
                                    sum / (idx.len() as f64 - null_count as f64)
                                })
                            }
                            _ => {
                                let take = { self.take_unchecked(idx.into()) };
                                take.mean()
                            }
                        }
                    }
                })
            }
            GroupsProxy::Slice {
                groups: groups_slice,
                ..
            } => {
                if _use_rolling_kernels(groups_slice, self.chunks()) {
                    let ca = self.cast(&DataType::Float64).unwrap();
                    ca.agg_mean(groups)
                } else {
                    _agg_helper_slice::<Float64Type, _>(groups_slice, |[first, len]| {
                        debug_assert!(len < self.len() as IdxSize);
                        match len {
                            0 => None,
                            1 => self.get(first as usize).map(|v| NumCast::from(v).unwrap()),
                            _ => {
                                let arr_group = _slice_from_offsets(self, first, len);
                                arr_group.mean()
                            }
                        }
                    })
                }
            }
        }
    }

    pub(crate) unsafe fn agg_var(&self, groups: &GroupsProxy, ddof: u8) -> Series {
        match groups {
            GroupsProxy::Idx(groups) => agg_helper_idx_on_all::<Float64Type, _>(groups, |idx| {
                debug_assert!(idx.len() <= self.len());
                if idx.is_empty() {
                    return None;
                }
                let take = { self.take_unchecked(idx.into()) };
                take.var_as_series(ddof)
                    .unpack::<Float64Type>()
                    .unwrap()
                    .get(0)
            }),
            GroupsProxy::Slice {
                groups: groups_slice,
                ..
            } => {
                if _use_rolling_kernels(groups_slice, self.chunks()) {
                    let ca = self.cast(&DataType::Float64).unwrap();
                    ca.agg_var(groups, ddof)
                } else {
                    _agg_helper_slice::<Float64Type, _>(groups_slice, |[first, len]| {
                        debug_assert!(len <= self.len() as IdxSize);
                        match len {
                            0 => None,
                            1 => NumCast::from(0),
                            _ => {
                                let arr_group = _slice_from_offsets(self, first, len);
                                arr_group.var(ddof)
                            }
                        }
                    })
                }
            }
        }
    }
    pub(crate) unsafe fn agg_std(&self, groups: &GroupsProxy, ddof: u8) -> Series {
        match groups {
            GroupsProxy::Idx(groups) => agg_helper_idx_on_all::<Float64Type, _>(groups, |idx| {
                debug_assert!(idx.len() <= self.len());
                if idx.is_empty() {
                    return None;
                }
                let take = { self.take_unchecked(idx.into()) };
                take.std_as_series(ddof)
                    .unpack::<Float64Type>()
                    .unwrap()
                    .get(0)
            }),
            GroupsProxy::Slice {
                groups: groups_slice,
                ..
            } => {
                if _use_rolling_kernels(groups_slice, self.chunks()) {
                    let ca = self.cast(&DataType::Float64).unwrap();
                    ca.agg_std(groups, ddof)
                } else {
                    _agg_helper_slice::<Float64Type, _>(groups_slice, |[first, len]| {
                        debug_assert!(len <= self.len() as IdxSize);
                        match len {
                            0 => None,
                            1 => NumCast::from(0),
                            _ => {
                                let arr_group = _slice_from_offsets(self, first, len);
                                arr_group.std(ddof)
                            }
                        }
                    })
                }
            }
        }
    }

    pub(crate) unsafe fn agg_quantile(
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
                let take = self.take_unchecked(idx.into());
                take.quantile_as_series(quantile, interpol)
                    .unwrap()
                    .unpack::<Float64Type>()
                    .unwrap()
                    .get(0)
            }),
            GroupsProxy::Slice {
                groups: groups_slice,
                ..
            } => {
                if _use_rolling_kernels(groups_slice, self.chunks()) {
                    let ca = self.cast(&DataType::Float64).unwrap();
                    ca.agg_quantile(groups, quantile, interpol)
                } else {
                    _agg_helper_slice::<Float64Type, _>(groups_slice, |[first, len]| {
                        debug_assert!(len <= self.len() as IdxSize);
                        match len {
                            0 => None,
                            1 => self.get(first as usize).map(|v| NumCast::from(v).unwrap()),
                            _ => {
                                let arr_group = _slice_from_offsets(self, first, len);
                                arr_group.quantile(quantile, interpol).unwrap()
                            }
                        }
                    })
                }
            }
        }
    }
    pub(crate) unsafe fn agg_median(&self, groups: &GroupsProxy) -> Series {
        match groups {
            GroupsProxy::Idx(groups) => agg_helper_idx_on_all::<Float64Type, _>(groups, |idx| {
                debug_assert!(idx.len() <= self.len());
                if idx.is_empty() {
                    return None;
                }
                let take = self.take_unchecked(idx.into());
                take.median_as_series()
                    .unpack::<Float64Type>()
                    .unwrap()
                    .get(0)
            }),
            GroupsProxy::Slice {
                groups: groups_slice,
                ..
            } => {
                if _use_rolling_kernels(groups_slice, self.chunks()) {
                    let ca = self.cast(&DataType::Float64).unwrap();
                    ca.agg_median(groups)
                } else {
                    _agg_helper_slice::<Float64Type, _>(groups_slice, |[first, len]| {
                        debug_assert!(len <= self.len() as IdxSize);
                        match len {
                            0 => None,
                            1 => self.get(first as usize).map(|v| NumCast::from(v).unwrap()),
                            _ => {
                                let arr_group = _slice_from_offsets(self, first, len);
                                arr_group.median()
                            }
                        }
                    })
                }
            }
        }
    }
}

impl<T: PolarsDataType> ChunkedArray<T> where ChunkedArray<T>: ChunkTake + IntoSeries {}
