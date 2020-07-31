use crate::prelude::*;

pub trait Apply<'a, A, B> {
    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(A) -> B + Copy;
}

impl<'a, T> Apply<'a, T::Native, T::Native> for ChunkedArray<T>
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
                    let vec: Vec<_> = slice.iter().copied().map(f).collect();
                    (vec, (null_count, opt_buffer))
                })
                .collect();
            ca.rename(self.name());
            ca
        }
    }
}

impl<'a> Apply<'a, bool, bool> for BooleanChunked {
    fn apply<F>(&self, f: F) -> Self
    where
        F: Fn(bool) -> bool,
    {
        self.into_iter()
            .map(|opt_v| {
                // Couldn't map due to movement of closure into map
                match opt_v {
                    None => None,
                    Some(v) => Some(f(v)),
                }
            })
            .collect()
    }
}

impl<'a> Apply<'a, &'a str, String> for Utf8Chunked {
    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(&'a str) -> String,
    {
        self.into_iter().map(f).collect()
    }
}
