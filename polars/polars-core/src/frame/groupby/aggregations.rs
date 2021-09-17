use crate::POOL;
use ahash::RandomState;
use num::{Bounded, Num, NumCast, ToPrimitive, Zero};
use rayon::prelude::*;
use std::collections::HashSet;
use std::hash::Hash;

use arrow::types::{simd::Simd, NativeType};

use crate::chunked_array::kernels::take_agg::{
    take_agg_no_null_primitive_iter_unchecked, take_agg_primitive_iter_unchecked,
    take_agg_primitive_iter_unchecked_count_nulls,
};
use crate::prelude::*;
use crate::utils::NoNull;
use arrow::buffer::MutableBuffer;
use polars_arrow::trusted_len::PushUnchecked;

pub(crate) trait NumericAggSync {
    fn agg_mean(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        None
    }
    fn agg_min(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        None
    }
    fn agg_max(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        None
    }
    fn agg_sum(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        None
    }
    fn agg_std(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        None
    }
    fn agg_var(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        None
    }

    /// Count the valid values. That is length - null_count
    /// Used in partitioned aggregation to compute the mean values.
    fn agg_valid_count(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        None
    }
}

fn agg_helper<T, F>(groups: &[(u32, Vec<u32>)], f: F) -> Option<Series>
where
    F: Fn(&(u32, Vec<u32>)) -> Option<T::Native> + Send + Sync,
    T: PolarsNumericType,
    ChunkedArray<T>: IntoSeries,
{
    let ca: ChunkedArray<T> = POOL.install(|| groups.par_iter().map(f).collect());
    Some(ca.into_series())
}

impl NumericAggSync for BooleanChunked {
    fn agg_min(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        self.cast::<UInt32Type>().unwrap().agg_min(groups)
    }
    fn agg_max(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        self.cast::<UInt32Type>().unwrap().agg_max(groups)
    }
    fn agg_sum(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        self.cast::<UInt32Type>().unwrap().agg_sum(groups)
    }
}
impl NumericAggSync for Utf8Chunked {}
impl NumericAggSync for ListChunked {}
#[cfg(feature = "dtype-categorical")]
impl NumericAggSync for CategoricalChunked {}
#[cfg(feature = "object")]
impl<T> NumericAggSync for ObjectChunked<T> {}

impl<T> NumericAggSync for ChunkedArray<T>
where
    T: PolarsNumericType + Sync,
    T::Native: NativeType + PartialOrd + Num + NumCast + Zero + Simd + Bounded,
    <T::Native as Simd>::Simd: std::ops::Add<Output = <T::Native as Simd>::Simd>
        + arrow::compute::aggregate::Sum<T::Native>
        + arrow::compute::aggregate::SimdOrd<T::Native>,
    ChunkedArray<T>: IntoSeries,
{
    fn agg_mean(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
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
                        take_agg_primitive_iter_unchecked_count_nulls::<T, _, _>(
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

    fn agg_min(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
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
                        take_agg_primitive_iter_unchecked::<T, _, _>(
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

    fn agg_max(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
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
                        take_agg_primitive_iter_unchecked::<T, _, _>(
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

    fn agg_sum(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
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
                        take_agg_primitive_iter_unchecked::<T, _, _>(
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
    fn agg_var(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
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
    fn agg_std(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
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
    #[cfg(feature = "lazy")]
    fn agg_valid_count(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
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

pub(crate) trait AggFirst {
    fn agg_first(&self, _groups: &[(u32, Vec<u32>)]) -> Series;
}

macro_rules! impl_agg_first {
    ($self:ident, $groups:ident, $ca_type:ty) => {{
        let mut ca = $groups
            .iter()
            .map(|(first, idx)| {
                debug_assert!(idx.len() <= $self.len());
                if idx.is_empty() {
                    return None;
                }
                $self.get(*first as usize)
            })
            .collect::<$ca_type>();

        ca.categorical_map = $self.categorical_map.clone();
        ca.into_series()
    }};
}

impl<T> AggFirst for ChunkedArray<T>
where
    T: PolarsNumericType + Send,
    ChunkedArray<T>: IntoSeries,
{
    fn agg_first(&self, groups: &[(u32, Vec<u32>)]) -> Series {
        impl_agg_first!(self, groups, ChunkedArray<T>)
    }
}

impl AggFirst for BooleanChunked {
    fn agg_first(&self, groups: &[(u32, Vec<u32>)]) -> Series {
        impl_agg_first!(self, groups, BooleanChunked)
    }
}

impl AggFirst for Utf8Chunked {
    fn agg_first(&self, groups: &[(u32, Vec<u32>)]) -> Series {
        impl_agg_first!(self, groups, Utf8Chunked)
    }
}

impl AggFirst for ListChunked {
    fn agg_first(&self, groups: &[(u32, Vec<u32>)]) -> Series {
        impl_agg_first!(self, groups, ListChunked)
    }
}

#[cfg(feature = "dtype-categorical")]
impl AggFirst for CategoricalChunked {
    fn agg_first(&self, groups: &[(u32, Vec<u32>)]) -> Series {
        let out = self
            .cast::<UInt32Type>()
            .unwrap()
            .agg_first(groups)
            .cast::<CategoricalType>()
            .unwrap();

        debug_assert!(out.categorical().unwrap().categorical_map.is_some());
        out
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> AggFirst for ObjectChunked<T> {
    fn agg_first(&self, groups: &[(u32, Vec<u32>)]) -> Series {
        let ca: Self = groups
            .iter()
            .map(|(first, idx)| {
                if idx.is_empty() {
                    return None;
                }
                self.get(*first as usize).cloned()
            })
            .collect();

        ca.into_series()
    }
}

pub(crate) trait AggLast {
    fn agg_last(&self, _groups: &[(u32, Vec<u32>)]) -> Series;
}

macro_rules! impl_agg_last {
    ($self:ident, $groups:ident, $ca_type:ty) => {{
        let mut ca = $groups
            .iter()
            .map(|(_first, idx)| {
                debug_assert!(idx.len() <= $self.len());
                if idx.is_empty() {
                    return None;
                }

                $self.get(idx[idx.len() - 1] as usize)
            })
            .collect::<$ca_type>();

        ca.categorical_map = $self.categorical_map.clone();
        ca.into_series()
    }};
}

impl<T> AggLast for ChunkedArray<T>
where
    T: PolarsNumericType + Send,
    ChunkedArray<T>: IntoSeries,
{
    fn agg_last(&self, groups: &[(u32, Vec<u32>)]) -> Series {
        impl_agg_last!(self, groups, ChunkedArray<T>)
    }
}

impl AggLast for BooleanChunked {
    fn agg_last(&self, groups: &[(u32, Vec<u32>)]) -> Series {
        impl_agg_last!(self, groups, BooleanChunked)
    }
}

impl AggLast for Utf8Chunked {
    fn agg_last(&self, groups: &[(u32, Vec<u32>)]) -> Series {
        impl_agg_last!(self, groups, Utf8Chunked)
    }
}

#[cfg(feature = "dtype-categorical")]
impl AggLast for CategoricalChunked {
    fn agg_last(&self, groups: &[(u32, Vec<u32>)]) -> Series {
        self.cast::<UInt32Type>()
            .unwrap()
            .agg_last(groups)
            .cast::<CategoricalType>()
            .unwrap()
    }
}

impl AggLast for ListChunked {
    fn agg_last(&self, groups: &[(u32, Vec<u32>)]) -> Series {
        impl_agg_last!(self, groups, ListChunked)
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> AggLast for ObjectChunked<T> {
    fn agg_last(&self, groups: &[(u32, Vec<u32>)]) -> Series {
        let ca: Self = groups
            .iter()
            .map(|(_first, idx)| {
                if idx.is_empty() {
                    return None;
                }
                self.get((idx.len() - 1) as usize).cloned()
            })
            .collect();

        ca.into_series()
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
    T::Native: PartialOrd + Num + NumCast + Zero + Simd,
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
