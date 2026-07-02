use arrow::array::*;

use crate::prelude::*;

pub mod par;

impl<T> ChunkedArray<T>
where
    T: PolarsDataType,
{
    #[inline]
    pub fn iter(&self) -> impl PolarsIterator<Item = Option<T::Physical<'_>>> {
        // SAFETY: we set the correct length of the iterator.
        unsafe {
            self.downcast_iter()
                .flat_map(|arr| arr.iter())
                .trust_my_length(self.len())
        }
    }

    #[inline]
    pub fn no_null_iter(&self) -> impl PolarsIterator<Item = T::Physical<'_>> {
        // SAFETY: we set the correct length of the iterator.
        unsafe {
            self.downcast_iter()
                .flat_map(|arr| arr.values_iter())
                .trust_my_length(self.len())
        }
    }
}

/// A [`PolarsIterator`] is an iterator over a [`ChunkedArray`] which contains polars types. A [`PolarsIterator`]
/// must implement [`ExactSizeIterator`] and [`DoubleEndedIterator`].
pub trait PolarsIterator:
    ExactSizeIterator + DoubleEndedIterator + Send + Sync + TrustedLen
{
}

/// Implement [`PolarsIterator`] for every iterator that implements the needed traits.
impl<T: ?Sized> PolarsIterator for T where
    T: ExactSizeIterator + DoubleEndedIterator + Send + Sync + TrustedLen
{
}

impl ListChunked {
    pub fn series_iter(&self) -> impl PolarsIterator<Item = Option<Series>> {
        let dtype = self.inner_dtype();
        unsafe {
            self.downcast_iter()
                .flat_map(|arr| arr.iter())
                .trust_my_length(self.len())
                .map(move |arr| {
                    arr.map(|arr| {
                        Series::from_chunks_and_dtype_unchecked(PlSmallStr::EMPTY, vec![arr], dtype)
                    })
                })
        }
    }

    pub fn no_null_series_iter(&self) -> impl PolarsIterator<Item = Series> {
        let inner_type = self.inner_dtype();
        unsafe {
            self.downcast_iter()
                .flat_map(|arr| arr.values_iter())
                .map(move |arr| {
                    Series::from_chunks_and_dtype_unchecked(
                        PlSmallStr::EMPTY,
                        vec![arr],
                        inner_type,
                    )
                })
                .trust_my_length(self.len())
        }
    }
}

#[cfg(feature = "dtype-array")]
impl ArrayChunked {
    pub fn series_iter(&self) -> impl PolarsIterator<Item = Option<Series>> {
        let dtype = self.inner_dtype();
        unsafe {
            self.downcast_iter()
                .flat_map(|arr| arr.iter())
                .trust_my_length(self.len())
                .map(move |arr| {
                    arr.map(|arr| {
                        Series::from_chunks_and_dtype_unchecked(PlSmallStr::EMPTY, vec![arr], dtype)
                    })
                })
        }
    }
}
