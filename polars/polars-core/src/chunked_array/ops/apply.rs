//! Implementations of the ChunkApply Trait.
use std::borrow::Cow;
use std::convert::TryFrom;

use arrow::array::{BooleanArray, PrimitiveArray};
use polars_arrow::array::PolarsArray;
use polars_arrow::trusted_len::PushUnchecked;

use crate::prelude::*;
use crate::series::IsSorted;
use crate::utils::{CustomIterTools, NoNull};

macro_rules! try_apply {
    ($self:expr, $f:expr) => {{
        if !$self.has_validity() {
            $self.into_no_null_iter().map($f).collect()
        } else {
            $self
                .into_iter()
                .map(|opt_v| opt_v.map($f).transpose())
                .collect()
        }
    }};
}

macro_rules! apply {
    ($self:expr, $f:expr) => {{
        if !$self.has_validity() {
            $self.into_no_null_iter().map($f).collect_trusted()
        } else {
            $self
                .into_iter()
                .map(|opt_v| opt_v.map($f))
                .collect_trusted()
        }
    }};
}

macro_rules! apply_enumerate {
    ($self:expr, $f:expr) => {{
        if !$self.has_validity() {
            $self
                .into_no_null_iter()
                .enumerate()
                .map($f)
                .collect_trusted()
        } else {
            $self
                .into_iter()
                .enumerate()
                .map(|(idx, opt_v)| opt_v.map(|v| $f((idx, v))))
                .collect_trusted()
        }
    }};
}

fn apply_in_place_impl<S, F>(name: &str, chunks: Vec<ArrayRef>, f: F) -> ChunkedArray<S>
where
    F: Fn(S::Native) -> S::Native + Copy,
    S: PolarsNumericType,
{
    use arrow::Either::*;
    let chunks = chunks
        .into_iter()
        .map(|arr| {
            let owned_arr = arr
                .as_any()
                .downcast_ref::<PrimitiveArray<S::Native>>()
                .unwrap()
                .clone();
            // make sure we have a single ref count coming in.
            drop(arr);

            match owned_arr.into_mut() {
                Left(immutable) => Box::new(arrow::compute::arity::unary(
                    &immutable,
                    f,
                    S::get_dtype().to_arrow(),
                )),
                Right(mut mutable) => {
                    let vals = mutable.values_mut_slice();
                    vals.iter_mut().for_each(|v| *v = f(*v));
                    let a: PrimitiveArray<_> = mutable.into();
                    Box::new(a) as ArrayRef
                }
            }
        })
        .collect();
    unsafe { ChunkedArray::<S>::from_chunks(name, chunks) }
}

impl<T: PolarsNumericType> ChunkedArray<T> {
    /// Cast a numeric array to another numeric data type and apply a function in place.
    /// This saves an allocation.
    pub fn cast_and_apply_in_place<F, S>(&self, f: F) -> ChunkedArray<S>
    where
        F: Fn(S::Native) -> S::Native + Copy,
        S: PolarsNumericType,
    {
        // if we cast, we create a new arrow buffer
        // then we clone the arrays and drop the casted arrays
        // this will ensure we have a single ref count
        // and we can mutate in place
        let chunks = {
            let s = self.cast(&S::get_dtype()).unwrap();
            s.chunks().clone()
        };
        apply_in_place_impl(self.name(), chunks, f)
    }
}

impl<T: PolarsNumericType> ChunkedArray<T> {
    pub fn apply_mut<F>(&mut self, f: F)
    where
        F: Fn(T::Native) -> T::Native + Copy,
    {
        // safety, we do no t change the lengths
        unsafe {
            self.downcast_iter_mut()
                .for_each(|arr| arrow::compute::arity_assign::unary(arr, f))
        };
        // can be in any order now
        self.set_sorted_flag(IsSorted::Not);
    }
}

impl<'a, T> ChunkApply<'a, T::Native, T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn apply_cast_numeric<F, S>(&self, f: F) -> ChunkedArray<S>
    where
        F: Fn(T::Native) -> S::Native + Copy,
        S: PolarsNumericType,
    {
        let chunks = self
            .data_views()
            .zip(self.iter_validities())
            .map(|(slice, validity)| {
                let values = Vec::<_>::from_trusted_len_iter(slice.iter().map(|&v| f(v)));
                to_array::<S>(values, validity.cloned())
            })
            .collect();
        unsafe { ChunkedArray::<S>::from_chunks(self.name(), chunks) }
    }

    fn branch_apply_cast_numeric_no_null<F, S>(&self, f: F) -> ChunkedArray<S>
    where
        F: Fn(Option<T::Native>) -> S::Native,
        S: PolarsNumericType,
    {
        let chunks = self
            .downcast_iter()
            .map(|array| {
                let values = if !array.has_validity() {
                    let values = array.values().iter().map(|&v| f(Some(v)));
                    Vec::<_>::from_trusted_len_iter(values)
                } else {
                    let values = array.into_iter().map(|v| f(v.copied()));
                    Vec::<_>::from_trusted_len_iter(values)
                };
                to_array::<S>(values, None)
            })
            .collect();
        unsafe { ChunkedArray::<S>::from_chunks(self.name(), chunks) }
    }

    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(T::Native) -> T::Native + Copy,
    {
        let chunks = self
            .data_views()
            .zip(self.iter_validities())
            .map(|(slice, validity)| {
                let values = slice.iter().copied().map(f);
                let values = Vec::<_>::from_trusted_len_iter(values);
                to_array::<T>(values, validity.cloned())
            })
            .collect();
        unsafe { ChunkedArray::<T>::from_chunks(self.name(), chunks) }
    }

    fn try_apply<F>(&'a self, f: F) -> PolarsResult<Self>
    where
        F: Fn(T::Native) -> PolarsResult<T::Native> + Copy,
    {
        let mut ca: ChunkedArray<T> = self
            .data_views()
            .zip(self.iter_validities())
            .map(|(slice, validity)| {
                let vec: PolarsResult<Vec<_>> = slice.iter().copied().map(f).collect();
                Ok((vec?, validity.cloned()))
            })
            .collect::<PolarsResult<_>>()?;
        ca.rename(self.name());
        Ok(ca)
    }

    fn apply_on_opt<F>(&'a self, f: F) -> Self
    where
        F: Fn(Option<T::Native>) -> Option<T::Native> + Copy,
    {
        let chunks = self
            .downcast_iter()
            .map(|arr| {
                let iter = arr.into_iter().map(|opt_v| f(opt_v.copied()));
                let arr = PrimitiveArray::<T::Native>::from_trusted_len_iter(iter)
                    .to(T::get_dtype().to_arrow());
                Box::new(arr) as ArrayRef
            })
            .collect();
        unsafe { Self::from_chunks(self.name(), chunks) }
    }

    fn apply_with_idx<F>(&'a self, f: F) -> Self
    where
        F: Fn((usize, T::Native)) -> T::Native + Copy,
    {
        if !self.has_validity() {
            let ca: NoNull<_> = self
                .into_no_null_iter()
                .enumerate()
                .map(f)
                .collect_trusted();
            ca.into_inner()
        } else {
            // we know that we only iterate over length == self.len()
            unsafe {
                self.downcast_iter()
                    .flatten()
                    .trust_my_length(self.len())
                    .enumerate()
                    .map(|(idx, opt_v)| opt_v.map(|v| f((idx, *v))))
                    .collect_trusted()
            }
        }
    }

    fn apply_with_idx_on_opt<F>(&'a self, f: F) -> Self
    where
        F: Fn((usize, Option<T::Native>)) -> Option<T::Native> + Copy,
    {
        // we know that we only iterate over length == self.len()
        unsafe {
            self.downcast_iter()
                .flatten()
                .trust_my_length(self.len())
                .enumerate()
                .map(|(idx, v)| f((idx, v.copied())))
                .collect_trusted()
        }
    }
    fn apply_to_slice<F, V>(&'a self, f: F, slice: &mut [V])
    where
        F: Fn(Option<T::Native>, &V) -> V,
    {
        assert!(slice.len() >= self.len());

        let mut idx = 0;
        self.downcast_iter().for_each(|arr| {
            arr.into_iter().for_each(|opt_val| {
                // Safety:
                // length asserted above
                let item = unsafe { slice.get_unchecked_mut(idx) };
                *item = f(opt_val.copied(), item);
                idx += 1;
            })
        });
    }
}

impl<'a> ChunkApply<'a, bool, bool> for BooleanChunked {
    fn apply_cast_numeric<F, S>(&self, f: F) -> ChunkedArray<S>
    where
        F: Fn(bool) -> S::Native + Copy,
        S: PolarsNumericType,
    {
        let f = |array: &BooleanArray| {
            let values = array.values().iter().map(f);
            let values = Vec::<_>::from_trusted_len_iter(values);
            let validity = array.validity().cloned();
            to_array::<S>(values, validity)
        };

        self.apply_kernel_cast(&f)
    }

    fn branch_apply_cast_numeric_no_null<F, S>(&self, f: F) -> ChunkedArray<S>
    where
        F: Fn(Option<bool>) -> S::Native + Copy,
        S: PolarsNumericType,
    {
        self.apply_kernel_cast(&|array: &BooleanArray| {
            let values = Vec::<_>::from_trusted_len_iter(array.into_iter().map(f));
            to_array::<S>(values, None)
        })
    }

    fn apply<F>(&self, f: F) -> Self
    where
        F: Fn(bool) -> bool + Copy,
    {
        apply!(self, f)
    }

    fn try_apply<F>(&self, f: F) -> PolarsResult<Self>
    where
        F: Fn(bool) -> PolarsResult<bool> + Copy,
    {
        try_apply!(self, f)
    }

    fn apply_on_opt<F>(&'a self, f: F) -> Self
    where
        F: Fn(Option<bool>) -> Option<bool> + Copy,
    {
        self.into_iter().map(f).collect_trusted()
    }

    fn apply_with_idx<F>(&'a self, f: F) -> Self
    where
        F: Fn((usize, bool)) -> bool + Copy,
    {
        apply_enumerate!(self, f)
    }

    fn apply_with_idx_on_opt<F>(&'a self, f: F) -> Self
    where
        F: Fn((usize, Option<bool>)) -> Option<bool> + Copy,
    {
        self.into_iter().enumerate().map(f).collect_trusted()
    }

    fn apply_to_slice<F, T>(&'a self, f: F, slice: &mut [T])
    where
        F: Fn(Option<bool>, &T) -> T,
    {
        assert!(slice.len() >= self.len());

        let mut idx = 0;
        self.downcast_iter().for_each(|arr| {
            arr.into_iter().for_each(|opt_val| {
                // Safety:
                // length asserted above
                let item = unsafe { slice.get_unchecked_mut(idx) };
                *item = f(opt_val, item);
                idx += 1;
            })
        });
    }
}

impl<'a> ChunkApply<'a, &'a str, Cow<'a, str>> for Utf8Chunked {
    fn apply_cast_numeric<F, S>(&'a self, f: F) -> ChunkedArray<S>
    where
        F: Fn(&'a str) -> S::Native + Copy,
        S: PolarsNumericType,
    {
        let chunks = self
            .downcast_iter()
            .map(|array| {
                let values = array.values_iter().map(f);
                let values = Vec::<_>::from_trusted_len_iter(values);
                to_array::<S>(values, array.validity().cloned())
            })
            .collect();
        unsafe { ChunkedArray::from_chunks(self.name(), chunks) }
    }

    fn branch_apply_cast_numeric_no_null<F, S>(&'a self, f: F) -> ChunkedArray<S>
    where
        F: Fn(Option<&'a str>) -> S::Native + Copy,
        S: PolarsNumericType,
    {
        let chunks = self
            .downcast_iter()
            .map(|array| {
                let values = array.into_iter().map(f);
                let values = Vec::<_>::from_trusted_len_iter(values);
                to_array::<S>(values, array.validity().cloned())
            })
            .collect();
        unsafe { ChunkedArray::from_chunks(self.name(), chunks) }
    }

    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(&'a str) -> Cow<'a, str> + Copy,
    {
        apply!(self, f)
    }

    fn try_apply<F>(&'a self, f: F) -> PolarsResult<Self>
    where
        F: Fn(&'a str) -> PolarsResult<Cow<'a, str>> + Copy,
    {
        try_apply!(self, f)
    }

    fn apply_on_opt<F>(&'a self, f: F) -> Self
    where
        F: Fn(Option<&'a str>) -> Option<Cow<'a, str>> + Copy,
    {
        let mut ca: Self = self.into_iter().map(f).collect_trusted();
        ca.rename(self.name());
        ca
    }

    fn apply_with_idx<F>(&'a self, f: F) -> Self
    where
        F: Fn((usize, &'a str)) -> Cow<'a, str> + Copy,
    {
        apply_enumerate!(self, f)
    }

    fn apply_with_idx_on_opt<F>(&'a self, f: F) -> Self
    where
        F: Fn((usize, Option<&'a str>)) -> Option<Cow<'a, str>> + Copy,
    {
        let mut ca: Self = self.into_iter().enumerate().map(f).collect_trusted();
        ca.rename(self.name());
        ca
    }

    fn apply_to_slice<F, T>(&'a self, f: F, slice: &mut [T])
    where
        F: Fn(Option<&'a str>, &T) -> T,
    {
        assert!(slice.len() >= self.len());

        let mut idx = 0;
        self.downcast_iter().for_each(|arr| {
            arr.into_iter().for_each(|opt_val| {
                // Safety:
                // length asserted above
                let item = unsafe { slice.get_unchecked_mut(idx) };
                *item = f(opt_val, item);
                idx += 1;
            })
        });
    }
}

#[cfg(feature = "dtype-binary")]
impl<'a> ChunkApply<'a, &'a [u8], Cow<'a, [u8]>> for BinaryChunked {
    fn apply_cast_numeric<F, S>(&'a self, f: F) -> ChunkedArray<S>
    where
        F: Fn(&'a [u8]) -> S::Native + Copy,
        S: PolarsNumericType,
    {
        let chunks = self
            .downcast_iter()
            .map(|array| {
                let values = array.values_iter().map(f);
                let values = Vec::<_>::from_trusted_len_iter(values);
                to_array::<S>(values, array.validity().cloned())
            })
            .collect();
        unsafe { ChunkedArray::from_chunks(self.name(), chunks) }
    }

    fn branch_apply_cast_numeric_no_null<F, S>(&'a self, f: F) -> ChunkedArray<S>
    where
        F: Fn(Option<&'a [u8]>) -> S::Native + Copy,
        S: PolarsNumericType,
    {
        let chunks = self
            .downcast_iter()
            .map(|array| {
                let values = array.into_iter().map(f);
                let values = Vec::<_>::from_trusted_len_iter(values);
                to_array::<S>(values, array.validity().cloned())
            })
            .collect();
        unsafe { ChunkedArray::from_chunks(self.name(), chunks) }
    }

    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(&'a [u8]) -> Cow<'a, [u8]> + Copy,
    {
        apply!(self, f)
    }

    fn try_apply<F>(&'a self, f: F) -> PolarsResult<Self>
    where
        F: Fn(&'a [u8]) -> PolarsResult<Cow<'a, [u8]>> + Copy,
    {
        try_apply!(self, f)
    }

    fn apply_on_opt<F>(&'a self, f: F) -> Self
    where
        F: Fn(Option<&'a [u8]>) -> Option<Cow<'a, [u8]>> + Copy,
    {
        let mut ca: Self = self.into_iter().map(f).collect_trusted();
        ca.rename(self.name());
        ca
    }

    fn apply_with_idx<F>(&'a self, f: F) -> Self
    where
        F: Fn((usize, &'a [u8])) -> Cow<'a, [u8]> + Copy,
    {
        apply_enumerate!(self, f)
    }

    fn apply_with_idx_on_opt<F>(&'a self, f: F) -> Self
    where
        F: Fn((usize, Option<&'a [u8]>)) -> Option<Cow<'a, [u8]>> + Copy,
    {
        let mut ca: Self = self.into_iter().enumerate().map(f).collect_trusted();
        ca.rename(self.name());
        ca
    }

    fn apply_to_slice<F, T>(&'a self, f: F, slice: &mut [T])
    where
        F: Fn(Option<&'a [u8]>, &T) -> T,
    {
        assert!(slice.len() >= self.len());

        let mut idx = 0;
        self.downcast_iter().for_each(|arr| {
            arr.into_iter().for_each(|opt_val| {
                // Safety:
                // length asserted above
                let item = unsafe { slice.get_unchecked_mut(idx) };
                *item = f(opt_val, item);
                idx += 1;
            })
        });
    }
}

impl ChunkApplyKernel<BooleanArray> for BooleanChunked {
    fn apply_kernel(&self, f: &dyn Fn(&BooleanArray) -> ArrayRef) -> Self {
        let chunks = self.downcast_iter().map(f).collect();
        unsafe { Self::from_chunks(self.name(), chunks) }
    }

    fn apply_kernel_cast<S>(&self, f: &dyn Fn(&BooleanArray) -> ArrayRef) -> ChunkedArray<S>
    where
        S: PolarsDataType,
    {
        let chunks = self.downcast_iter().map(f).collect();
        unsafe { ChunkedArray::<S>::from_chunks(self.name(), chunks) }
    }
}

impl<T> ChunkApplyKernel<PrimitiveArray<T::Native>> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn apply_kernel(&self, f: &dyn Fn(&PrimitiveArray<T::Native>) -> ArrayRef) -> Self {
        self.apply_kernel_cast(&f)
    }
    fn apply_kernel_cast<S>(
        &self,
        f: &dyn Fn(&PrimitiveArray<T::Native>) -> ArrayRef,
    ) -> ChunkedArray<S>
    where
        S: PolarsDataType,
    {
        let chunks = self.downcast_iter().map(f).collect();
        unsafe { ChunkedArray::from_chunks(self.name(), chunks) }
    }
}

impl ChunkApplyKernel<LargeStringArray> for Utf8Chunked {
    fn apply_kernel(&self, f: &dyn Fn(&LargeStringArray) -> ArrayRef) -> Self {
        self.apply_kernel_cast(&f)
    }

    fn apply_kernel_cast<S>(&self, f: &dyn Fn(&LargeStringArray) -> ArrayRef) -> ChunkedArray<S>
    where
        S: PolarsDataType,
    {
        let chunks = self.downcast_iter().map(f).collect();
        unsafe { ChunkedArray::from_chunks(self.name(), chunks) }
    }
}

#[cfg(feature = "dtype-binary")]
impl ChunkApplyKernel<LargeBinaryArray> for BinaryChunked {
    fn apply_kernel(&self, f: &dyn Fn(&LargeBinaryArray) -> ArrayRef) -> Self {
        self.apply_kernel_cast(&f)
    }

    fn apply_kernel_cast<S>(&self, f: &dyn Fn(&LargeBinaryArray) -> ArrayRef) -> ChunkedArray<S>
    where
        S: PolarsDataType,
    {
        let chunks = self.downcast_iter().map(f).collect();
        unsafe { ChunkedArray::from_chunks(self.name(), chunks) }
    }
}

impl<'a> ChunkApply<'a, Series, Series> for ListChunked {
    fn apply_cast_numeric<F, S>(&self, f: F) -> ChunkedArray<S>
    where
        F: Fn(Series) -> S::Native + Copy,
        S: PolarsNumericType,
    {
        let dtype = self.inner_dtype();
        let chunks = self
            .downcast_iter()
            .map(|array| {
                unsafe {
                    let values = array
                        .values_iter()
                        .map(|array| {
                            // safety
                            // reported dtype is correct
                            let series =
                                Series::from_chunks_and_dtype_unchecked("", vec![array], &dtype);
                            f(series)
                        })
                        .trust_my_length(self.len())
                        .collect_trusted::<Vec<_>>();

                    to_array::<S>(values, array.validity().cloned())
                }
            })
            .collect();
        unsafe { ChunkedArray::from_chunks(self.name(), chunks) }
    }

    fn branch_apply_cast_numeric_no_null<F, S>(&self, f: F) -> ChunkedArray<S>
    where
        F: Fn(Option<Series>) -> S::Native + Copy,
        S: PolarsNumericType,
    {
        let dtype = self.inner_dtype();
        let chunks = self
            .downcast_iter()
            .map(|array| {
                let values = array.iter().map(|x| {
                    let x = x.map(|x| {
                        // safety
                        // reported dtype is correct
                        unsafe { Series::from_chunks_and_dtype_unchecked("", vec![x], &dtype) }
                    });
                    f(x)
                });
                let len = array.len();

                // we know the iterators len
                unsafe {
                    let values = Vec::<_>::from_trusted_len_iter(values.trust_my_length(len));
                    to_array::<S>(values, array.validity().cloned())
                }
            })
            .collect();
        unsafe { ChunkedArray::from_chunks(self.name(), chunks) }
    }

    /// Apply a closure `F` elementwise.
    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(Series) -> Series + Copy,
    {
        if self.is_empty() {
            return self.clone();
        }
        let mut fast_explode = true;
        let mut function = |s: Series| {
            let out = f(s);
            if out.is_empty() {
                fast_explode = false;
            }
            out
        };
        let mut ca: ListChunked = apply!(self, &mut function);
        if fast_explode {
            ca.set_fast_explode()
        }
        ca
    }

    fn try_apply<F>(&'a self, f: F) -> PolarsResult<Self>
    where
        F: Fn(Series) -> PolarsResult<Series> + Copy,
    {
        if self.is_empty() {
            return Ok(self.clone());
        }

        let mut fast_explode = true;
        let mut function = |s: Series| {
            let out = f(s);
            if let Ok(out) = &out {
                if out.is_empty() {
                    fast_explode = false;
                }
            }
            out
        };
        let ca: PolarsResult<ListChunked> = try_apply!(self, &mut function);
        let mut ca = ca?;
        if fast_explode {
            ca.set_fast_explode()
        }
        Ok(ca)
    }

    fn apply_on_opt<F>(&'a self, f: F) -> Self
    where
        F: Fn(Option<Series>) -> Option<Series> + Copy,
    {
        if self.is_empty() {
            return self.clone();
        }
        self.into_iter().map(f).collect_trusted()
    }

    /// Apply a closure elementwise. The closure gets the index of the element as first argument.
    fn apply_with_idx<F>(&'a self, f: F) -> Self
    where
        F: Fn((usize, Series)) -> Series + Copy,
    {
        if self.is_empty() {
            return self.clone();
        }
        let mut fast_explode = true;
        let mut function = |(idx, s)| {
            let out = f((idx, s));
            if out.is_empty() {
                fast_explode = false;
            }
            out
        };
        let mut ca: ListChunked = apply_enumerate!(self, function);
        if fast_explode {
            ca.set_fast_explode()
        }
        ca
    }

    /// Apply a closure elementwise. The closure gets the index of the element as first argument.
    fn apply_with_idx_on_opt<F>(&'a self, f: F) -> Self
    where
        F: Fn((usize, Option<Series>)) -> Option<Series> + Copy,
    {
        if self.is_empty() {
            return self.clone();
        }
        let mut fast_explode = true;
        let function = |(idx, s)| {
            let out = f((idx, s));
            if let Some(out) = &out {
                if out.is_empty() {
                    fast_explode = false;
                }
            }
            out
        };
        let mut ca: ListChunked = self.into_iter().enumerate().map(function).collect_trusted();
        if fast_explode {
            ca.set_fast_explode()
        }
        ca
    }

    fn apply_to_slice<F, T>(&'a self, f: F, slice: &mut [T])
    where
        F: Fn(Option<Series>, &T) -> T,
    {
        assert!(slice.len() >= self.len());

        let mut idx = 0;
        self.downcast_iter().for_each(|arr| {
            arr.iter().for_each(|opt_val| {
                let opt_val = opt_val.map(|arrayref| Series::try_from(("", arrayref)).unwrap());

                // Safety:
                // length asserted above
                let item = unsafe { slice.get_unchecked_mut(idx) };
                *item = f(opt_val, item);
                idx += 1;
            })
        });
    }
}

#[cfg(feature = "object")]
impl<'a, T> ChunkApply<'a, &'a T, T> for ObjectChunked<T>
where
    T: PolarsObject,
{
    fn apply_cast_numeric<F, S>(&'a self, _f: F) -> ChunkedArray<S>
    where
        F: Fn(&'a T) -> S::Native + Copy,
        S: PolarsNumericType,
    {
        todo!()
    }

    fn branch_apply_cast_numeric_no_null<F, S>(&'a self, _f: F) -> ChunkedArray<S>
    where
        F: Fn(Option<&'a T>) -> S::Native + Copy,
        S: PolarsNumericType,
    {
        todo!()
    }

    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(&'a T) -> T + Copy,
    {
        let mut ca: ObjectChunked<T> = self.into_iter().map(|opt_v| opt_v.map(f)).collect();
        ca.rename(self.name());
        ca
    }

    fn try_apply<F>(&'a self, _f: F) -> PolarsResult<Self>
    where
        F: Fn(&'a T) -> PolarsResult<T> + Copy,
    {
        todo!()
    }

    fn apply_on_opt<F>(&'a self, f: F) -> Self
    where
        F: Fn(Option<&'a T>) -> Option<T> + Copy,
    {
        let mut ca: ObjectChunked<T> = self.into_iter().map(f).collect();
        ca.rename(self.name());
        ca
    }

    fn apply_with_idx<F>(&'a self, _f: F) -> Self
    where
        F: Fn((usize, &'a T)) -> T + Copy,
    {
        todo!()
    }

    fn apply_with_idx_on_opt<F>(&'a self, _f: F) -> Self
    where
        F: Fn((usize, Option<&'a T>)) -> Option<T> + Copy,
    {
        todo!()
    }

    fn apply_to_slice<F, V>(&'a self, f: F, slice: &mut [V])
    where
        F: Fn(Option<&'a T>, &V) -> V,
    {
        assert!(slice.len() >= self.len());
        let mut idx = 0;
        self.downcast_iter().for_each(|arr| {
            arr.into_iter().for_each(|opt_val| {
                // Safety:
                // length asserted above
                let item = unsafe { slice.get_unchecked_mut(idx) };
                *item = f(opt_val, item);
                idx += 1;
            })
        });
    }
}
