//! Implementations of the ChunkApply Trait.
use crate::prelude::*;
use crate::utils::Xob;
use arrow::array::{ArrayRef, LargeStringArray, PrimitiveArray};

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
    /// Chooses the fastest path for closure application.
    /// Null values remain null.
    ///
    /// # Example
    ///
    /// ```
    /// use polars_core::prelude::*;
    /// fn double(ca: &UInt32Chunked) -> UInt32Chunked {
    ///     ca.apply(|v| v * 2)
    /// }
    /// ```
    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(T::Native) -> T::Native + Copy,
    {
        if let Ok(slice) = self.cont_slice() {
            let new: Xob<ChunkedArray<T>> = slice.iter().copied().map(f).collect();
            new.into_inner()
        } else {
            let mut ca: ChunkedArray<T> = self
                .data_views()
                .iter()
                .copied()
                .zip(self.null_bits())
                .map(|(slice, (null_count, opt_buffer))| {
                    let vec: AlignedVec<_> = slice.iter().copied().map(f).collect();
                    (vec, (null_count, opt_buffer))
                })
                .collect();
            ca.rename(self.name());
            ca
        }
    }

    fn apply_with_idx<F>(&'a self, f: F) -> Self
    where
        F: Fn((usize, T::Native)) -> T::Native + Copy,
    {
        if self.null_count() == 0 {
            let ca: Xob<_> = self.into_no_null_iter().enumerate().map(f).collect();
            ca.into_inner()
        } else {
            self.into_iter()
                .enumerate()
                .map(|(idx, opt_v)| opt_v.map(|v| f((idx, v))))
                .collect()
        }
    }

    fn apply_with_idx_on_opt<F>(&'a self, f: F) -> Self
    where
        F: Fn((usize, Option<T::Native>)) -> Option<T::Native> + Copy,
    {
        self.into_iter().enumerate().map(f).collect()
    }
}

impl<'a> ChunkApply<'a, bool, bool> for BooleanChunked {
    fn apply<F>(&self, f: F) -> Self
    where
        F: Fn(bool) -> bool + Copy,
    {
        apply!(self, f)
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

impl<'a> ChunkApply<'a, &'a str, String> for Utf8Chunked {
    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(&'a str) -> String + Copy,
    {
        apply!(self, f)
    }
    fn apply_with_idx<F>(&'a self, f: F) -> Self
    where
        F: Fn((usize, &'a str)) -> String + Copy,
    {
        apply_enumerate!(self, f)
    }

    fn apply_with_idx_on_opt<F>(&'a self, f: F) -> Self
    where
        F: Fn((usize, Option<&'a str>)) -> Option<String> + Copy,
    {
        self.into_iter().enumerate().map(f).collect()
    }
}

impl<T> ChunkApplyKernel<PrimitiveArray<T>> for ChunkedArray<T>
where
    T: PolarsPrimitiveType,
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
        let chunks = self.downcast_chunks().into_iter().map(f).collect();
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
        let chunks = self.downcast_chunks().into_iter().map(f).collect();
        ChunkedArray::new_from_chunks(self.name(), chunks)
    }
}

impl<'a> ChunkApply<'a, Series, Series> for ListChunked {
    /// Apply a closure `F` elementwise.
    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(Series) -> Series + Copy,
    {
        apply!(self, f)
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
