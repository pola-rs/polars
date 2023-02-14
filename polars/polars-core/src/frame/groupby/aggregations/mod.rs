mod agg_list;
mod dispatch;

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
        return Box::new(PrimitiveArray::new(T::PRIMITIVE.into(), out.into(), None));
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

    Box::new(PrimitiveArray::new(
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
        return Box::new(PrimitiveArray::new(T::PRIMITIVE.into(), out.into(), None));
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

pub fn _agg_helper_idx_bool<F>(groups: &GroupsIdx, f: F) -> Series
where
    F: Fn((IdxSize, &Vec<IdxSize>)) -> Option<bool> + Send + Sync,
{
    let ca: BooleanChunked = POOL.install(|| groups.into_par_iter().map(f).collect());
    ca.into_series()
}

pub fn _agg_helper_idx_utf8<'a, F>(groups: &'a GroupsIdx, f: F) -> Series
where
    F: Fn((IdxSize, &'a Vec<IdxSize>)) -> Option<&'a str> + Send + Sync,
{
    let ca: Utf8Chunked = POOL.install(|| groups.into_par_iter().map(f).collect());
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

pub fn _agg_helper_slice_bool<F>(groups: &[[IdxSize; 2]], f: F) -> Series
where
    F: Fn([IdxSize; 2]) -> Option<bool> + Send + Sync,
{
    let ca: BooleanChunked = POOL.install(|| groups.par_iter().copied().map(f).collect());
    ca.into_series()
}

pub fn _agg_helper_slice_utf8<'a, F>(groups: &'a [[IdxSize; 2]], f: F) -> Series
where
    F: Fn([IdxSize; 2]) -> Option<&'a str> + Send + Sync,
{
    let ca: Utf8Chunked = POOL.install(|| groups.par_iter().copied().map(f).collect());
    ca.into_series()
}

impl BooleanChunked {
    pub(crate) unsafe fn agg_min(&self, groups: &GroupsProxy) -> Series {
        // faster paths
        match (self.is_sorted_flag2(), self.null_count()) {
            (IsSorted::Ascending, 0) => {
                return self.clone().into_series().agg_first(groups);
            }
            (IsSorted::Descending, 0) => {
                return self.clone().into_series().agg_last(groups);
            }
            _ => {}
        }
        match groups {
            GroupsProxy::Idx(groups) => _agg_helper_idx_bool(groups, |(first, idx)| {
                debug_assert!(idx.len() <= self.len());
                if idx.is_empty() {
                    None
                } else if idx.len() == 1 {
                    self.get(first as usize)
                } else {
                    // TODO! optimize this
                    // can just check if any is false and early stop
                    let take = { self.take_unchecked(idx.into()) };
                    take.min().map(|v| v == 1)
                }
            }),
            GroupsProxy::Slice {
                groups: groups_slice,
                ..
            } => _agg_helper_slice_bool(groups_slice, |[first, len]| {
                debug_assert!(len <= self.len() as IdxSize);
                match len {
                    0 => None,
                    1 => self.get(first as usize),
                    _ => {
                        let arr_group = _slice_from_offsets(self, first, len);
                        arr_group.min().map(|v| v == 1)
                    }
                }
            }),
        }
    }
    pub(crate) unsafe fn agg_max(&self, groups: &GroupsProxy) -> Series {
        // faster paths
        match (self.is_sorted_flag2(), self.null_count()) {
            (IsSorted::Ascending, 0) => {
                return self.clone().into_series().agg_last(groups);
            }
            (IsSorted::Descending, 0) => {
                return self.clone().into_series().agg_first(groups);
            }
            _ => {}
        }

        match groups {
            GroupsProxy::Idx(groups) => _agg_helper_idx_bool(groups, |(first, idx)| {
                debug_assert!(idx.len() <= self.len());
                if idx.is_empty() {
                    None
                } else if idx.len() == 1 {
                    self.get(first as usize)
                } else {
                    // TODO! optimize this
                    // can just check if any is true and early stop
                    let take = { self.take_unchecked(idx.into()) };
                    take.max().map(|v| v == 1)
                }
            }),
            GroupsProxy::Slice {
                groups: groups_slice,
                ..
            } => _agg_helper_slice_bool(groups_slice, |[first, len]| {
                debug_assert!(len <= self.len() as IdxSize);
                match len {
                    0 => None,
                    1 => self.get(first as usize),
                    _ => {
                        let arr_group = _slice_from_offsets(self, first, len);
                        arr_group.max().map(|v| v == 1)
                    }
                }
            }),
        }
    }
    pub(crate) unsafe fn agg_sum(&self, groups: &GroupsProxy) -> Series {
        self.cast(&IDX_DTYPE).unwrap().agg_sum(groups)
    }
}

impl Utf8Chunked {
    #[allow(clippy::needless_lifetimes)]
    pub(crate) unsafe fn agg_min<'a>(&'a self, groups: &GroupsProxy) -> Series {
        // faster paths
        match (&self.is_sorted_flag2(), &self.null_count()) {
            (IsSorted::Ascending, 0) => {
                return self.clone().into_series().agg_first(groups);
            }
            (IsSorted::Descending, 0) => {
                return self.clone().into_series().agg_last(groups);
            }
            _ => {}
        }

        match groups {
            GroupsProxy::Idx(groups) => {
                let ca_self = self.rechunk();
                let arr = ca_self.downcast_iter().next().unwrap();
                _agg_helper_idx_utf8(groups, |(first, idx)| {
                    debug_assert!(idx.len() <= ca_self.len());
                    if idx.is_empty() {
                        None
                    } else if idx.len() == 1 {
                        ca_self.get(first as usize)
                    } else if self.null_count() == 0 {
                        take_agg_utf8_iter_unchecked_no_null(
                            arr,
                            indexes_to_usizes(idx),
                            |acc, v| if acc < v { acc } else { v },
                        )
                    } else {
                        take_agg_utf8_iter_unchecked(
                            arr,
                            indexes_to_usizes(idx),
                            |acc, v| if acc < v { acc } else { v },
                            idx.len() as IdxSize,
                        )
                    }
                })
            }
            GroupsProxy::Slice {
                groups: groups_slice,
                ..
            } => _agg_helper_slice_utf8(groups_slice, |[first, len]| {
                debug_assert!(len <= self.len() as IdxSize);
                match len {
                    0 => None,
                    1 => self.get(first as usize),
                    _ => {
                        let arr_group = _slice_from_offsets(self, first, len);
                        let borrowed = arr_group.min_str();

                        // Safety:
                        // The borrowed has `arr_group`s lifetime, but it actually points to data
                        // hold by self. Here we tell the compiler that.
                        unsafe { std::mem::transmute::<Option<&str>, Option<&'a str>>(borrowed) }
                    }
                }
            }),
        }
    }

    #[allow(clippy::needless_lifetimes)]
    pub(crate) unsafe fn agg_max<'a>(&'a self, groups: &GroupsProxy) -> Series {
        // faster paths
        match (self.is_sorted_flag2(), self.null_count()) {
            (IsSorted::Ascending, 0) => {
                return self.clone().into_series().agg_last(groups);
            }
            (IsSorted::Descending, 0) => {
                return self.clone().into_series().agg_first(groups);
            }
            _ => {}
        }

        match groups {
            GroupsProxy::Idx(groups) => {
                let ca_self = self.rechunk();
                let arr = ca_self.downcast_iter().next().unwrap();
                _agg_helper_idx_utf8(groups, |(first, idx)| {
                    debug_assert!(idx.len() <= self.len());
                    if idx.is_empty() {
                        None
                    } else if idx.len() == 1 {
                        ca_self.get(first as usize)
                    } else if ca_self.null_count() == 0 {
                        take_agg_utf8_iter_unchecked_no_null(
                            arr,
                            indexes_to_usizes(idx),
                            |acc, v| if acc > v { acc } else { v },
                        )
                    } else {
                        take_agg_utf8_iter_unchecked(
                            arr,
                            indexes_to_usizes(idx),
                            |acc, v| if acc > v { acc } else { v },
                            idx.len() as IdxSize,
                        )
                    }
                })
            }
            GroupsProxy::Slice {
                groups: groups_slice,
                ..
            } => _agg_helper_slice_utf8(groups_slice, |[first, len]| {
                debug_assert!(len <= self.len() as IdxSize);
                match len {
                    0 => None,
                    1 => self.get(first as usize),
                    _ => {
                        let arr_group = _slice_from_offsets(self, first, len);
                        let borrowed = arr_group.max_str();

                        // Safety:
                        // The borrowed has `arr_group`s lifetime, but it actually points to data
                        // hold by self. Here we tell the compiler that.
                        unsafe { std::mem::transmute::<Option<&str>, Option<&'a str>>(borrowed) }
                    }
                }
            }),
        }
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

/// Intermediate helper trait so we can have a single generic implementation
/// This trait will ensure the specific dispatch works without complicating
/// the trait bounds.
trait QuantileDispatcher<K> {
    fn _quantile(self, quantile: f64, interpol: QuantileInterpolOptions)
        -> PolarsResult<Option<K>>;

    fn _median(self) -> Option<K>;
}

impl<T> QuantileDispatcher<f64> for ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: Ord,
    ChunkedArray<T>: IntoSeries,
{
    fn _quantile(
        self,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> PolarsResult<Option<f64>> {
        self.quantile_faster(quantile, interpol)
    }
    fn _median(self) -> Option<f64> {
        self.median_faster()
    }
}

impl QuantileDispatcher<f32> for Float32Chunked {
    fn _quantile(
        self,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> PolarsResult<Option<f32>> {
        self.quantile_faster(quantile, interpol)
    }
    fn _median(self) -> Option<f32> {
        self.median_faster()
    }
}
impl QuantileDispatcher<f64> for Float64Chunked {
    fn _quantile(
        self,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> PolarsResult<Option<f64>> {
        self.quantile_faster(quantile, interpol)
    }
    fn _median(self) -> Option<f64> {
        self.median_faster()
    }
}

unsafe fn agg_quantile_generic<T, K>(
    ca: &ChunkedArray<T>,
    groups: &GroupsProxy,
    quantile: f64,
    interpol: QuantileInterpolOptions,
) -> Series
where
    T: PolarsNumericType,
    ChunkedArray<T>: QuantileDispatcher<K::Native>,
    ChunkedArray<K>: IntoSeries,
    K: PolarsNumericType,
{
    let invalid_quantile = !(0.0..=1.0).contains(&quantile);
    if invalid_quantile {
        return Series::full_null(ca.name(), groups.len(), ca.dtype());
    }
    match groups {
        GroupsProxy::Idx(groups) => agg_helper_idx_on_all::<K, _>(groups, |idx| {
            debug_assert!(idx.len() <= ca.len());
            if idx.is_empty() {
                return None;
            }
            let take = { ca.take_unchecked(idx.into()) };
            // checked with invalid quantile check
            take._quantile(quantile, interpol).unwrap_unchecked()
        }),
        GroupsProxy::Slice { groups, .. } => {
            if _use_rolling_kernels(groups, ca.chunks()) {
                // this cast is a no-op for floats
                let s = ca.cast(&K::get_dtype()).unwrap();
                let ca: &ChunkedArray<K> = s.as_ref().as_ref();
                let arr = ca.downcast_iter().next().unwrap();
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
                // the rolling kernels works on the dtype, this is not yet the float
                // output type we need.
                ChunkedArray::<K>::from_chunks("", vec![arr]).into_series()
            } else {
                _agg_helper_slice::<K, _>(groups, |[first, len]| {
                    debug_assert!(first + len <= ca.len() as IdxSize);
                    match len {
                        0 => None,
                        1 => ca.get(first as usize).map(|v| NumCast::from(v).unwrap()),
                        _ => {
                            let arr_group = _slice_from_offsets(ca, first, len);
                            // unwrap checked with invalid quantile check
                            arr_group
                                ._quantile(quantile, interpol)
                                .unwrap_unchecked()
                                .map(|flt| NumCast::from(flt).unwrap_unchecked())
                        }
                    }
                })
            }
        }
    }
}

unsafe fn agg_median_generic<T, K>(ca: &ChunkedArray<T>, groups: &GroupsProxy) -> Series
where
    T: PolarsNumericType,
    ChunkedArray<T>: QuantileDispatcher<K::Native>,
    ChunkedArray<K>: IntoSeries,
    K: PolarsNumericType,
{
    match groups {
        GroupsProxy::Idx(groups) => agg_helper_idx_on_all::<K, _>(groups, |idx| {
            debug_assert!(idx.len() <= ca.len());
            if idx.is_empty() {
                return None;
            }
            let take = { ca.take_unchecked(idx.into()) };
            take._median()
        }),
        GroupsProxy::Slice { .. } => {
            agg_quantile_generic::<T, K>(ca, groups, 0.5, QuantileInterpolOptions::Linear)
        }
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
        match (self.is_sorted_flag2(), self.null_count()) {
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
        match (self.is_sorted_flag2(), self.null_count()) {
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
                take.var(ddof)
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
                take.std(ddof)
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
}

impl Float32Chunked {
    pub(crate) unsafe fn agg_quantile(
        &self,
        groups: &GroupsProxy,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> Series {
        agg_quantile_generic::<_, Float32Type>(self, groups, quantile, interpol)
    }
    pub(crate) unsafe fn agg_median(&self, groups: &GroupsProxy) -> Series {
        agg_median_generic::<_, Float32Type>(self, groups)
    }
}
impl Float64Chunked {
    pub(crate) unsafe fn agg_quantile(
        &self,
        groups: &GroupsProxy,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> Series {
        agg_quantile_generic::<_, Float64Type>(self, groups, quantile, interpol)
    }
    pub(crate) unsafe fn agg_median(&self, groups: &GroupsProxy) -> Series {
        agg_median_generic::<_, Float64Type>(self, groups)
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
                        debug_assert!(first + len <= self.len() as IdxSize);
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
                take.var(ddof)
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
                        debug_assert!(first + len <= self.len() as IdxSize);
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
                take.std(ddof)
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
                        debug_assert!(first + len <= self.len() as IdxSize);
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
        agg_quantile_generic::<_, Float64Type>(self, groups, quantile, interpol)
    }
    pub(crate) unsafe fn agg_median(&self, groups: &GroupsProxy) -> Series {
        agg_median_generic::<_, Float64Type>(self, groups)
    }
}

impl<T: PolarsDataType> ChunkedArray<T> where ChunkedArray<T>: ChunkTake + IntoSeries {}
