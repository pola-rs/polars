//! Implementations of the ChunkApply Trait.
use crate::prelude::*;
use crate::utils::{CustomIterTools, NoNull};
use arrow::array::{Array, ArrayRef, BooleanArray, PrimitiveArray};
use std::borrow::Cow;
use std::convert::TryFrom;

macro_rules! apply {
    ($self:expr, $f:expr) => {{
        if $self.null_count() == 0 {
            $self.into_no_null_iter().map($f).collect()
        } else {
            $self.into_iter().map(|opt_v| opt_v.map($f)).collect()
        }
    }};
}

macro_rules! apply_enumerate {
    ($self:expr, $f:expr) => {{
        if $self.null_count() == 0 {
            $self.into_no_null_iter().enumerate().map($f).collect()
        } else {
            $self
                .into_iter()
                .enumerate()
                .map(|(idx, opt_v)| opt_v.map(|v| $f((idx, v))))
                .collect()
        }
    }};
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
        let mut chunks = self
            .data_views()
            .zip(self.null_bits())
            .map(|(slice, validity)| {
                let values = AlignedVec::<_>::from(slice);
                to_array::<T>(values, validity.clone())
            })
            .collect();
        ChunkedArray::<S>::new_from_chunks(self.name(), chunks)
    }

    fn branch_apply_cast_numeric_no_null<F, S>(&self, f: F) -> ChunkedArray<S>
    where
        F: Fn(Option<T::Native>) -> S::Native,
        S: PolarsNumericType,
    {
        let chunks = self
            .downcast_iter()
            .map(|array| {
                let values = if array.null_count() == 0 {
                    let values = array.values().iter().map(|&v| f(Some(v)));
                    AlignedVec::<_>::from_trusted_len_iter(values)
                } else {
                    let values = array.into_iter().map(|v| f(v.copied()));
                    AlignedVec::<_>::from_trusted_len_iter(values)
                };
                to_array::<S>(values, None)
            })
            .collect();
        ChunkedArray::<S>::new_from_chunks(self.name(), chunks)
    }

    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(T::Native) -> T::Native + Copy,
    {
        let chunks = self
            .data_views()
            .into_iter()
            .zip(self.null_bits())
            .map(|(slice, validity)| {
                let values = slice.iter().copied().map(f);
                let values = AlignedVec::<_>::from_trusted_len_iter(values);
                to_array::<T>(values, validity.clone())
            })
            .collect();
        ChunkedArray::<T>::new_from_chunks(self.name(), chunks)
    }

    fn apply_on_opt<F>(&'a self, f: F) -> Self
    where
        F: Fn(Option<T::Native>) -> Option<T::Native> + Copy,
    {
        self.downcast_iter()
            .flatten()
            .trust_my_length(self.len())
            .map(|v| f(v.copied()))
            .collect()
    }

    fn apply_with_idx<F>(&'a self, f: F) -> Self
    where
        F: Fn((usize, T::Native)) -> T::Native + Copy,
    {
        if self.null_count() == 0 {
            let ca: NoNull<_> = self.into_no_null_iter().enumerate().map(f).collect();
            ca.into_inner()
        } else {
            self.downcast_iter()
                .flatten()
                .trust_my_length(self.len())
                .enumerate()
                .map(|(idx, opt_v)| opt_v.map(|v| f((idx, *v))))
                .collect()
        }
    }

    fn apply_with_idx_on_opt<F>(&'a self, f: F) -> Self
    where
        F: Fn((usize, Option<T::Native>)) -> Option<T::Native> + Copy,
    {
        self.downcast_iter()
            .flatten()
            .trust_my_length(self.len())
            .enumerate()
            .map(|(idx, v)| f((idx, v.copied())))
            .collect()
    }
}

impl<'a> ChunkApply<'a, bool, bool> for BooleanChunked {
    fn apply_cast_numeric<F, S>(&self, f: F) -> ChunkedArray<S>
    where
        F: Fn(bool) -> S::Native + Copy,
        S: PolarsNumericType,
    {
        self.apply_kernel_cast(|array| {
            let values = array.values().iter().map(f);
            let values = AlignedVec::<_>::from_trusted_len_iter(values);
            let validity = array.validity().clone();
            to_array::<S>(values, validity)
        })
    }

    fn branch_apply_cast_numeric_no_null<F, S>(&self, f: F) -> ChunkedArray<S>
    where
        F: Fn(Option<bool>) -> S::Native + Copy,
        S: PolarsNumericType,
    {
        self.apply_kernel_cast(|array| {
            let values = AlignedVec::<_>::from_trusted_len_iter(array.into_iter().map(f));
            to_array::<S>(values, None)
        })
    }

    fn apply<F>(&self, f: F) -> Self
    where
        F: Fn(bool) -> bool + Copy,
    {
        apply!(self, f)
    }

    fn apply_on_opt<F>(&'a self, f: F) -> Self
    where
        F: Fn(Option<bool>) -> Option<bool> + Copy,
    {
        self.into_iter().map(f).collect()
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
        self.into_iter().enumerate().map(f).collect()
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
            .into_iter()
            .map(|array| {
                let values = array.values_iter().map(|x| f(x));
                let values = AlignedVec::<_>::from_trusted_len_iter(values);
                to_array::<S>(values, array.validity().clone())
            })
            .collect();
        ChunkedArray::new_from_chunks(self.name(), chunks)
    }

    fn branch_apply_cast_numeric_no_null<F, S>(&'a self, f: F) -> ChunkedArray<S>
    where
        F: Fn(Option<&'a str>) -> S::Native + Copy,
        S: PolarsNumericType,
    {
        let chunks = self
            .downcast_iter()
            .into_iter()
            .map(|array| {
                let values = array.into_iter().map(|x| f(x));
                let values = AlignedVec::<_>::from_trusted_len_iter(values);
                to_array::<S>(values, array.validity().clone())
            })
            .collect();
        ChunkedArray::new_from_chunks(self.name(), chunks)
    }

    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(&'a str) -> Cow<'a, str> + Copy,
    {
        apply!(self, f)
    }

    fn apply_on_opt<F>(&'a self, f: F) -> Self
    where
        F: Fn(Option<&'a str>) -> Option<Cow<'a, str>> + Copy,
    {
        self.into_iter().map(f).collect()
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
        self.into_iter().enumerate().map(f).collect()
    }
}

impl ChunkApplyKernel<BooleanArray> for BooleanChunked {
    fn apply_kernel<F>(&self, f: F) -> Self
    where
        F: Fn(&BooleanArray) -> ArrayRef,
    {
        let chunks = self
            .downcast_iter()
            .into_iter()
            .map(|array| f(array))
            .collect();
        Self::new_from_chunks(self.name(), chunks)
    }

    fn apply_kernel_cast<F, S>(&self, f: F) -> ChunkedArray<S>
    where
        F: Fn(&BooleanArray) -> ArrayRef,
        S: PolarsDataType,
    {
        let chunks = self
            .downcast_iter()
            .into_iter()
            .map(|array| f(array))
            .collect();
        ChunkedArray::<S>::new_from_chunks(self.name(), chunks)
    }
}

impl<T> ChunkApplyKernel<PrimitiveArray<T::Native>> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn apply_kernel<F>(&self, f: F) -> Self
    where
        F: Fn(&PrimitiveArray<T::Native>) -> ArrayRef,
    {
        self.apply_kernel_cast(f)
    }
    fn apply_kernel_cast<F, S>(&self, f: F) -> ChunkedArray<S>
    where
        F: Fn(&PrimitiveArray<T::Native>) -> ArrayRef,
        S: PolarsDataType,
    {
        let chunks = self.downcast_iter().into_iter().map(f).collect();
        ChunkedArray::new_from_chunks(self.name(), chunks)
    }
}

impl ChunkApplyKernel<LargeStringArray> for Utf8Chunked {
    fn apply_kernel<F>(&self, f: F) -> Self
    where
        F: Fn(&LargeStringArray) -> ArrayRef,
    {
        self.apply_kernel_cast(f)
    }

    fn apply_kernel_cast<F, S>(&self, f: F) -> ChunkedArray<S>
    where
        F: Fn(&LargeStringArray) -> ArrayRef,
        S: PolarsDataType,
    {
        let chunks = self.downcast_iter().into_iter().map(f).collect();
        ChunkedArray::new_from_chunks(self.name(), chunks)
    }
}

impl<'a> ChunkApply<'a, Series, Series> for ListChunked {
    fn apply_cast_numeric<F, S>(&self, f: F) -> ChunkedArray<S>
    where
        F: Fn(Series) -> S::Native + Copy,
        S: PolarsNumericType,
    {
        let chunks = self
            .downcast_iter()
            .into_iter()
            .map(|array| {
                let values: AlignedVec<_> = (0..array.len())
                    .map(|idx| {
                        let arrayref: ArrayRef = unsafe { array.value_unchecked(idx) }.into();
                        let series = Series::try_from(("", arrayref)).unwrap();
                        f(series)
                    })
                    .collect();
                to_array::<S>(values, array.validity().clone())
            })
            .collect();
        ChunkedArray::new_from_chunks(self.name(), chunks)
    }

    fn branch_apply_cast_numeric_no_null<F, S>(&self, f: F) -> ChunkedArray<S>
    where
        F: Fn(Option<Series>) -> S::Native + Copy,
        S: PolarsNumericType,
    {
        let chunks = self
            .downcast_iter()
            .into_iter()
            .map(|array| {
                let values = array.iter().map(|x| {
                    let x = x.map(|x| {
                        let x: ArrayRef = x.into();
                        Series::try_from(("", x)).unwrap()
                    });
                    f(x)
                });
                let values = AlignedVec::<_>::from_trusted_len_iter(values);
                to_array::<S>(values, array.validity().clone())
            })
            .collect();
        ChunkedArray::new_from_chunks(self.name(), chunks)
    }

    /// Apply a closure `F` elementwise.
    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(Series) -> Series + Copy,
    {
        apply!(self, f)
    }

    fn apply_on_opt<F>(&'a self, f: F) -> Self
    where
        F: Fn(Option<Series>) -> Option<Series> + Copy,
    {
        self.into_iter().map(f).collect()
    }

    /// Apply a closure elementwise. The closure gets the index of the element as first argument.
    fn apply_with_idx<F>(&'a self, f: F) -> Self
    where
        F: Fn((usize, Series)) -> Series + Copy,
    {
        apply_enumerate!(self, f)
    }

    /// Apply a closure elementwise. The closure gets the index of the element as first argument.
    fn apply_with_idx_on_opt<F>(&'a self, f: F) -> Self
    where
        F: Fn((usize, Option<Series>)) -> Option<Series> + Copy,
    {
        self.into_iter().enumerate().map(f).collect()
    }
}
