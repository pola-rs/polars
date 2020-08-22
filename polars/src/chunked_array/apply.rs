//! Implementations of the ChunkApply Trait.
use crate::prelude::*;

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
            slice.iter().copied().map(f).map(Some).collect()
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
        self.into_iter().map(|opt_v| opt_v.map(|v| f(v))).collect()
    }
}

impl<'a> ChunkApply<'a, &'a str, String> for Utf8Chunked {
    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(&'a str) -> String,
    {
        self.into_iter().map(|opt_v| opt_v.map(|v| f(v))).collect()
    }
}
