use crate::POOL;
use ahash::RandomState;
use num::{Bounded, Num, NumCast, ToPrimitive, Zero};
use rayon::prelude::*;
use std::collections::HashSet;
use std::hash::Hash;

use arrow::types::{simd::Simd, NativeType};

use crate::prelude::*;
use crate::series::implementations::SeriesWrap;
use crate::utils::NoNull;
use arrow::buffer::MutableBuffer;
use polars_arrow::kernels::take_agg::*;
use polars_arrow::trusted_len::PushUnchecked;

fn agg_helper<T, F>(groups: &[(u32, Vec<u32>)], f: F) -> Option<Series>
where
    F: Fn(&(u32, Vec<u32>)) -> Option<T::Native> + Send + Sync,
    T: PolarsNumericType,
    ChunkedArray<T>: IntoSeries,
{
    let ca: ChunkedArray<T> = POOL.install(|| groups.par_iter().map(f).collect());
    Some(ca.into_series())
}

impl BooleanChunked {
    pub(crate) fn agg_min(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        self.cast::<UInt32Type>().unwrap().agg_min(groups)
    }
    pub(crate) fn agg_max(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        self.cast::<UInt32Type>().unwrap().agg_max(groups)
    }
    pub(crate) fn agg_sum(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        self.cast::<UInt32Type>().unwrap().agg_sum(groups)
    }
}

impl<T> ChunkedArray<T>
where
    ChunkedArray<T>: ChunkTake,
    T: PolarsDataType + Sync,
{
    #[cfg(feature = "lazy")]
    pub(crate) fn agg_valid_count(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        agg_helper::<UInt32Type, _>(groups, |(_first, idx)| {
            debug_assert!(idx.len() <= self.len());
            if idx.is_empty() {
                None
            } else if self.null_count() == 0 {
                Some(idx.len() as u32)
            } else {
                let take = unsafe { self.take_unchecked(idx.iter().map(|i| *i as usize).into()) };
                Some((take.len() - take.null_count()) as u32)
            }
        })
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
    pub(crate) fn agg_min(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        agg_helper::<T, _>(groups, |(first, idx)| {
            debug_assert!(idx.len() <= self.len());
            if idx.is_empty() {
                None
            } else if idx.len() == 1 {
                self.get(*first as usize)
            } else {
                match (self.null_count(), self.chunks.len()) {
                    (0, 1) => Some(unsafe {
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
                        let take =
                            unsafe { self.take_unchecked(idx.iter().map(|i| *i as usize).into()) };
                        take.min()
                    }
                }
            }
        })
    }

    pub(crate) fn agg_max(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        agg_helper::<T, _>(groups, |(first, idx)| {
            debug_assert!(idx.len() <= self.len());
            if idx.is_empty() {
                None
            } else if idx.len() == 1 {
                self.get(*first as usize)
            } else {
                match (self.null_count(), self.chunks.len()) {
                    (0, 1) => Some(unsafe {
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
                        let take =
                            unsafe { self.take_unchecked(idx.iter().map(|i| *i as usize).into()) };
                        take.max()
                    }
                }
            }
        })
    }

    pub(crate) fn agg_sum(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        agg_helper::<T, _>(groups, |(first, idx)| {
            debug_assert!(idx.len() <= self.len());
            if idx.is_empty() {
                None
            } else if idx.len() == 1 {
                self.get(*first as usize)
            } else {
                match (self.null_count(), self.chunks.len()) {
                    (0, 1) => Some(unsafe {
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
                        let take =
                            unsafe { self.take_unchecked(idx.iter().map(|i| *i as usize).into()) };
                        take.sum()
                    }
                }
            }
        })
    }
}

impl<T> SeriesWrap<ChunkedArray<T>>
where
    T: PolarsFloatType,
    ChunkedArray<T>: IntoSeries,
    T::Native: NativeType + PartialOrd + Num + NumCast + Simd + std::iter::Sum<T::Native>,
    <T::Native as Simd>::Simd: std::ops::Add<Output = <T::Native as Simd>::Simd>
        + arrow::compute::aggregate::Sum<T::Native>
        + arrow::compute::aggregate::SimdOrd<T::Native>,
{
    pub(crate) fn agg_mean(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        agg_helper::<T, _>(groups, |(first, idx)| {
            // this can fail due to a bug in lazy code.
            // here users can create filters in aggregations
            // and thereby creating shorter columns than the original group tuples.
            // the group tuples are modified, but if that's done incorrect there can be out of bounds
            // access
            debug_assert!(idx.len() <= self.len());
            let out = if idx.is_empty() {
                None
            } else if idx.len() == 1 {
                self.get(*first as usize).map(|sum| sum.to_f64().unwrap())
            } else {
                match (self.null_count(), self.chunks.len()) {
                    (0, 1) => unsafe {
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
                        let take =
                            unsafe { self.take_unchecked(idx.iter().map(|i| *i as usize).into()) };
                        let opt_sum: Option<T::Native> = take.sum();
                        opt_sum.map(|sum| sum.to_f64().unwrap() / idx.len() as f64)
                    }
                }
            };
            out.map(|flt| NumCast::from(flt).unwrap())
        })
    }

    pub(crate) fn agg_var(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        let ca = &self.0;
        agg_helper::<T, _>(groups, |(_first, idx)| {
            debug_assert!(idx.len() <= ca.len());
            if idx.is_empty() {
                return None;
            }
            let take = unsafe { ca.take_unchecked(idx.iter().map(|i| *i as usize).into()) };
            take.into_series()
                .var_as_series()
                .unpack::<T>()
                .unwrap()
                .get(0)
        })
    }
    pub(crate) fn agg_std(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        let ca = &self.0;
        agg_helper::<T, _>(groups, |(_first, idx)| {
            debug_assert!(idx.len() <= ca.len());
            if idx.is_empty() {
                return None;
            }
            let take = unsafe { ca.take_unchecked(idx.iter().map(|i| *i as usize).into()) };
            take.into_series()
                .std_as_series()
                .unpack::<T>()
                .unwrap()
                .get(0)
        })
    }
}

impl<T> ChunkedArray<T>
where
    T: PolarsIntegerType,
    ChunkedArray<T>: IntoSeries,
    T::Native:
        NativeType + PartialOrd + Num + NumCast + Zero + Simd + Bounded + std::iter::Sum<T::Native>,
    <T::Native as Simd>::Simd: std::ops::Add<Output = <T::Native as Simd>::Simd>
        + arrow::compute::aggregate::Sum<T::Native>
        + arrow::compute::aggregate::SimdOrd<T::Native>,
{
    pub(crate) fn agg_mean(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        agg_helper::<Float64Type, _>(groups, |(first, idx)| {
            // this can fail due to a bug in lazy code.
            // here users can create filters in aggregations
            // and thereby creating shorter columns than the original group tuples.
            // the group tuples are modified, but if that's done incorrect there can be out of bounds
            // access
            debug_assert!(idx.len() <= self.len());
            if idx.is_empty() {
                None
            } else if idx.len() == 1 {
                self.get(*first as usize).map(|sum| sum.to_f64().unwrap())
            } else {
                match (self.null_count(), self.chunks.len()) {
                    (0, 1) => unsafe {
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
                        let take =
                            unsafe { self.take_unchecked(idx.iter().map(|i| *i as usize).into()) };
                        let opt_sum: Option<T::Native> = take.sum();
                        opt_sum.map(|sum| sum.to_f64().unwrap() / idx.len() as f64)
                    }
                }
            }
        })
    }

    pub(crate) fn agg_var(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        agg_helper::<Float64Type, _>(groups, |(_first, idx)| {
            debug_assert!(idx.len() <= self.len());
            if idx.is_empty() {
                return None;
            }
            let take = unsafe { self.take_unchecked(idx.iter().map(|i| *i as usize).into()) };
            take.into_series()
                .var_as_series()
                .unpack::<Float64Type>()
                .unwrap()
                .get(0)
        })
    }
    pub(crate) fn agg_std(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        agg_helper::<Float64Type, _>(groups, |(_first, idx)| {
            debug_assert!(idx.len() <= self.len());
            if idx.is_empty() {
                return None;
            }
            let take = unsafe { self.take_unchecked(idx.iter().map(|i| *i as usize).into()) };
            take.into_series()
                .std_as_series()
                .unpack::<Float64Type>()
                .unwrap()
                .get(0)
        })
    }
}

impl<T> ChunkedArray<T>
where
    ChunkedArray<T>: ChunkTake + IntoSeries,
{
    pub(crate) fn agg_first(&self, groups: &[(u32, Vec<u32>)]) -> Series {
        let iter = groups.iter().map(|(first, idx)| {
            if idx.is_empty() {
                None
            } else {
                Some(*first as usize)
            }
        });
        unsafe { self.take_unchecked(iter.into()) }.into_series()
    }

    pub(crate) fn agg_last(&self, groups: &[(u32, Vec<u32>)]) -> Series {
        let iter = groups.iter().map(|(_, idx)| {
            if idx.is_empty() {
                None
            } else {
                Some(idx[idx.len() - 1] as usize)
            }
        });
        unsafe { self.take_unchecked(iter.into()) }.into_series()
    }
}

pub(crate) trait AggNUnique {
    fn agg_n_unique(&self, _groups: &[(u32, Vec<u32>)]) -> Option<UInt32Chunked> {
        None
    }
}

macro_rules! impl_agg_n_unique {
    ($self:ident, $groups:ident, $ca_type:ty) => {{
        $groups
            .into_par_iter()
            .map(|(_first, idx)| {
                debug_assert!(idx.len() <= $self.len());
                if idx.is_empty() {
                    return 0;
                }
                let taker = $self.take_rand();

                if $self.null_count() == 0 {
                    let mut set = HashSet::with_hasher(RandomState::new());
                    for i in idx {
                        let v = unsafe { taker.get_unchecked(*i as usize) };
                        set.insert(v);
                    }
                    set.len() as u32
                } else {
                    let mut set = HashSet::with_hasher(RandomState::new());
                    for i in idx {
                        let opt_v = taker.get(*i as usize);
                        set.insert(opt_v);
                    }
                    set.len() as u32
                }
            })
            .collect::<$ca_type>()
            .into_inner()
    }};
}

impl<T> AggNUnique for ChunkedArray<T>
where
    T: PolarsIntegerType + Sync,
    T::Native: Hash + Eq,
{
    fn agg_n_unique(&self, groups: &[(u32, Vec<u32>)]) -> Option<UInt32Chunked> {
        Some(impl_agg_n_unique!(self, groups, NoNull<UInt32Chunked>))
    }
}

// todo! could use mantissa method here
impl AggNUnique for Float32Chunked {}
impl AggNUnique for Float64Chunked {}
impl AggNUnique for ListChunked {}
#[cfg(feature = "dtype-categorical")]
impl AggNUnique for CategoricalChunked {
    fn agg_n_unique(&self, groups: &[(u32, Vec<u32>)]) -> Option<UInt32Chunked> {
        self.cast::<UInt32Type>()
            .unwrap()
            .agg_n_unique(groups)
            .map(|mut ca| {
                ca.categorical_map = self.categorical_map.clone();
                ca
            })
    }
}
#[cfg(feature = "object")]
impl<T> AggNUnique for ObjectChunked<T> {}

// TODO: could be faster as it can only be null, true, or false
impl AggNUnique for BooleanChunked {
    fn agg_n_unique(&self, groups: &[(u32, Vec<u32>)]) -> Option<UInt32Chunked> {
        Some(impl_agg_n_unique!(self, groups, NoNull<UInt32Chunked>))
    }
}

impl AggNUnique for Utf8Chunked {
    fn agg_n_unique(&self, groups: &[(u32, Vec<u32>)]) -> Option<UInt32Chunked> {
        Some(impl_agg_n_unique!(self, groups, NoNull<UInt32Chunked>))
    }
}

pub trait AggList {
    fn agg_list(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        None
    }
}

impl<T> AggList for ChunkedArray<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: IntoSeries,
{
    fn agg_list(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        let mut can_fast_explode = true;
        let arr = match self.cont_slice() {
            Ok(values) => {
                let mut offsets = MutableBuffer::<i64>::with_capacity(groups.len() + 1);
                let mut length_so_far = 0i64;
                offsets.push(length_so_far);

                let mut list_values = MutableBuffer::<T::Native>::with_capacity(self.len());
                groups.iter().for_each(|(_, idx)| {
                    let idx_len = idx.len();
                    if idx_len == 0 {
                        can_fast_explode = false;
                    }

                    length_so_far += idx_len as i64;
                    // Safety:
                    // group tuples are in bounds
                    unsafe {
                        list_values.extend_from_trusted_len_iter(
                            idx.iter().map(|idx| *values.get_unchecked(*idx as usize)),
                        );
                        // Safety:
                        // we know that offsets has allocated enough slots
                        offsets.push_unchecked(length_so_far);
                    }
                });
                let array =
                    PrimitiveArray::from_data(T::get_dtype().to_arrow(), list_values.into(), None);
                let data_type = ListArray::<i64>::default_datatype(T::get_dtype().to_arrow());
                ListArray::<i64>::from_data(data_type, offsets.into(), Arc::new(array), None)
            }
            _ => {
                let mut builder =
                    ListPrimitiveChunkedBuilder::<T>::new(self.name(), groups.len(), self.len());
                for (_first, idx) in groups {
                    let s = unsafe {
                        self.take_unchecked(idx.iter().map(|i| *i as usize).into())
                            .into_series()
                    };
                    builder.append_opt_series(Some(&s));
                }
                return Some(builder.finish().into_series());
            }
        };
        let mut ca = ListChunked::new_from_chunks(self.name(), vec![Arc::new(arr)]);
        if can_fast_explode {
            ca.set_fast_explode()
        }
        Some(ca.into())
    }
}

impl AggList for BooleanChunked {
    fn agg_list(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        let mut builder = ListBooleanChunkedBuilder::new(self.name(), groups.len(), self.len());
        for (_first, idx) in groups {
            let s = unsafe {
                self.take_unchecked(idx.iter().map(|i| *i as usize).into())
                    .into_series()
            };
            builder.append_series(&s)
        }
        Some(builder.finish().into_series())
    }
}

impl AggList for Utf8Chunked {
    fn agg_list(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        let mut builder = ListUtf8ChunkedBuilder::new(self.name(), groups.len(), self.len());
        for (_first, idx) in groups {
            let s = unsafe {
                self.take_unchecked(idx.iter().map(|i| *i as usize).into())
                    .into_series()
            };
            builder.append_series(&s)
        }
        Some(builder.finish().into_series())
    }
}

impl AggList for ListChunked {}
#[cfg(feature = "object")]
impl<T> AggList for ObjectChunked<T> {}

pub(crate) trait AggQuantile {
    fn agg_quantile(&self, _groups: &[(u32, Vec<u32>)], _quantile: f64) -> Option<Series> {
        None
    }

    fn agg_median(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        None
    }
}

impl<T> AggQuantile for ChunkedArray<T>
where
    T: PolarsNumericType + Sync,
    T::Native: PartialOrd + Num + NumCast + Zero + Simd + std::iter::Sum<T::Native>,
    <T::Native as Simd>::Simd: std::ops::Add<Output = <T::Native as Simd>::Simd>
        + arrow::compute::aggregate::Sum<T::Native>
        + arrow::compute::aggregate::SimdOrd<T::Native>,
    ChunkedArray<T>: IntoSeries,
{
    fn agg_quantile(&self, groups: &[(u32, Vec<u32>)], quantile: f64) -> Option<Series> {
        agg_helper::<T, _>(groups, |(_first, idx)| {
            if idx.is_empty() {
                return None;
            }

            let group_vals = unsafe { self.take_unchecked(idx.iter().map(|i| *i as usize).into()) };
            group_vals.quantile(quantile).unwrap()
        })
    }

    fn agg_median(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        agg_helper::<Float64Type, _>(groups, |(_first, idx)| {
            if idx.is_empty() {
                return None;
            }

            let group_vals = unsafe { self.take_unchecked(idx.iter().map(|i| *i as usize).into()) };
            group_vals.median()
        })
    }
}

impl AggQuantile for Utf8Chunked {}
impl AggQuantile for BooleanChunked {}
impl AggQuantile for ListChunked {}
#[cfg(feature = "dtype-categorical")]
impl AggQuantile for CategoricalChunked {}
#[cfg(feature = "object")]
impl<T> AggQuantile for ObjectChunked<T> {}
