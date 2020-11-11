//! Implementations of the ChunkApply Trait.
use crate::prelude::*;
use crate::utils::Xob;
use arrow::array::{ArrayRef, PrimitiveArray, StringArray};

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
    /// use polars::prelude::*;
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
}

impl<'a> ChunkApply<'a, bool, bool> for BooleanChunked {
    fn apply<F>(&self, f: F) -> Self
    where
        F: Fn(bool) -> bool + Copy,
    {
        if self.null_count() == 0 {
            self.into_no_null_iter().map(f).collect()
        } else {
            self.into_iter().map(|opt_v| opt_v.map(|v| f(v))).collect()
        }
    }
}

impl<'a> ChunkApply<'a, &'a str, String> for Utf8Chunked {
    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(&'a str) -> String,
    {
        if self.null_count() == 0 {
            self.into_no_null_iter().map(f).collect()
        } else {
            self.into_iter().map(|opt_v| opt_v.map(|v| f(v))).collect()
        }
    }
}

impl<T> ChunkApplyKernel<PrimitiveArray<T>> for ChunkedArray<T>
where
    T: ArrowPrimitiveType,
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

impl ChunkApplyKernel<StringArray> for Utf8Chunked {
    fn apply_kernel<F>(&self, f: F) -> Self
    where
        F: Fn(&StringArray) -> ArrayRef,
    {
        self.apply_kernel_cast(f)
    }

    fn apply_kernel_cast<F, S>(&self, f: F) -> ChunkedArray<S>
    where
        F: Fn(&StringArray) -> ArrayRef,
        S: PolarsDataType,
    {
        let chunks = self.downcast_chunks().into_iter().map(f).collect();
        ChunkedArray::new_from_chunks(self.name(), chunks)
    }
}
