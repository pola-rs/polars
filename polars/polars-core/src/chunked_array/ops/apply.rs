//! Implementations of the ChunkApply Trait.
use crate::prelude::*;
use crate::utils::{CustomIterTools, NoNull};
use arrow::array::{Array, ArrayRef, BooleanArray, LargeStringArray, PrimitiveArray};
use std::borrow::Cow;
use std::convert::TryFrom;

macro_rules! apply {
    ($self:expr, $f:expr) => {{
        if $self.null_count() == 0 {
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
        if $self.null_count() == 0 {
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

impl<'a, T> ChunkApply<'a, T::Native, T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn apply_cast_numeric<F, S>(&self, f: F) -> ChunkedArray<S>
    where
        F: Fn(T::Native) -> S::Native + Copy,
        S: PolarsNumericType,
    {
        let mut ca: ChunkedArray<S> = self
            .data_views()
            .zip(self.null_bits())
            .map(|(slice, (_null_count, opt_buffer))| {
                let vec: AlignedVec<_> = slice.iter().copied().map(f).collect_trusted();
                (vec, opt_buffer.cloned())
            })
            .collect();
        ca.rename(self.name());
        ca
    }

    fn branch_apply_cast_numeric_no_null<F, S>(&self, f: F) -> ChunkedArray<S>
    where
        F: Fn(Option<T::Native>) -> S::Native + Copy,
        S: PolarsNumericType,
    {
        let chunks = self
            .downcast_iter()
            .map(|array| {
                let av: AlignedVec<_> = if array.null_count() == 0 {
                    array
                        .values()
                        .iter()
                        .map(|&v| f(Some(v)))
                        .trust_my_length(array.len())
                        .collect_trusted()
                } else {
                    array
                        .into_iter()
                        .map(f)
                        .trust_my_length(array.len())
                        .collect_trusted()
                };
                Arc::new(av.into_primitive_array::<S>(None)) as ArrayRef
            })
            .collect();
        ChunkedArray::<S>::new_from_chunks(self.name(), chunks)
    }

    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(T::Native) -> T::Native + Copy,
    {
        let mut ca: ChunkedArray<T> = self
            .data_views()
            .into_iter()
            .zip(self.null_bits())
            .map(|(slice, (_null_count, opt_buffer))| {
                let vec: AlignedVec<_> = slice.iter().copied().map(f).collect_trusted();
                (vec, opt_buffer.cloned())
            })
            .collect();
        ca.rename(self.name());
        ca
    }

    fn apply_on_opt<F>(&'a self, f: F) -> Self
    where
        F: Fn(Option<T::Native>) -> Option<T::Native> + Copy,
    {
        let chunks = self
            .downcast_iter()
            .map(|arr| {
                let iter = arr.into_iter().map(f);
                // Safety:
                // array's length is trusted
                let arr = unsafe { PrimitiveArray::<T>::from_trusted_len_iter(iter) };
                Arc::new(arr) as ArrayRef
            })
            .collect();
        Self::new_from_chunks(self.name(), chunks)
    }

    fn apply_with_idx<F>(&'a self, f: F) -> Self
    where
        F: Fn((usize, T::Native)) -> T::Native + Copy,
    {
        if self.null_count() == 0 {
            let ca: NoNull<_> = self
                .into_no_null_iter()
                .enumerate()
                .map(f)
                .collect_trusted();
            ca.into_inner()
        } else {
            self.downcast_iter()
                .flatten()
                .trust_my_length(self.len())
                .enumerate()
                .map(|(idx, opt_v)| opt_v.map(|v| f((idx, v))))
                .collect_trusted()
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
            .map(f)
            .collect_trusted()
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
                *item = f(opt_val, item);
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
        self.apply_kernel_cast(|array| {
            let av: AlignedVec<_> = (0..array.len())
                .map(|idx| unsafe { f(array.value_unchecked(idx)) })
                .collect_trusted();
            let null_bit_buffer = array.data_ref().null_buffer().cloned();
            Arc::new(av.into_primitive_array::<S>(null_bit_buffer)) as ArrayRef
        })
    }

    fn branch_apply_cast_numeric_no_null<F, S>(&self, f: F) -> ChunkedArray<S>
    where
        F: Fn(Option<bool>) -> S::Native + Copy,
        S: PolarsNumericType,
    {
        self.apply_kernel_cast(|array| {
            let av: AlignedVec<_> = array
                .into_iter()
                .map(f)
                .trust_my_length(array.len())
                .collect_trusted();
            Arc::new(av.into_primitive_array::<S>(None)) as ArrayRef
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
            .into_iter()
            .map(|array| {
                let av: AlignedVec<_> = (0..array.len())
                    .map(|idx| unsafe { f(array.value_unchecked(idx)) })
                    .collect_trusted();
                let null_bit_buffer = array.data_ref().null_buffer().cloned();
                Arc::new(av.into_primitive_array::<S>(null_bit_buffer)) as ArrayRef
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
                let av: AlignedVec<_> = array
                    .into_iter()
                    .map(f)
                    .trust_my_length(array.len())
                    .collect_trusted();
                let null_bit_buffer = array.data_ref().null_buffer().cloned();
                Arc::new(av.into_primitive_array::<S>(null_bit_buffer)) as ArrayRef
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

impl<T> ChunkApplyKernel<PrimitiveArray<T>> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn apply_kernel<F>(&self, f: F) -> Self
    where
        F: Fn(&PrimitiveArray<T>) -> ArrayRef,
    {
        self.apply_kernel_cast(f)
    }
    fn apply_kernel_cast<F, S>(&self, f: F) -> ChunkedArray<S>
    where
        F: Fn(&PrimitiveArray<T>) -> ArrayRef,
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
                let av: AlignedVec<_> = (0..array.len())
                    .map(|idx| {
                        let arrayref = unsafe { array.value_unchecked(idx) };
                        let series = Series::try_from(("", arrayref)).unwrap();
                        f(series)
                    })
                    .collect_trusted();
                let null_bit_buffer = array.data_ref().null_buffer().cloned();
                Arc::new(av.into_primitive_array::<S>(null_bit_buffer)) as ArrayRef
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
                let av: AlignedVec<_> = (0..array.len())
                    .map(|idx| {
                        let v = if array.is_valid(idx) {
                            let arrayref = unsafe { array.value_unchecked(idx) };
                            let series = Series::try_from(("", arrayref)).unwrap();
                            Some(series)
                        } else {
                            None
                        };

                        f(v)
                    })
                    .collect_trusted();
                let null_bit_buffer = array.data_ref().null_buffer().cloned();
                Arc::new(av.into_primitive_array::<S>(null_bit_buffer)) as ArrayRef
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
        self.into_iter().map(f).collect_trusted()
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
        self.into_iter().enumerate().map(f).collect_trusted()
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
