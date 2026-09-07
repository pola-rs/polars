//! A [`ChunkedArray`] whose every chunk is flat, which is what handing out a *slice* needs.

use std::borrow::Cow;

use polars_array::{Flat, StaticArray};
use polars_buffer::Buffer;

use crate::prelude::*;

impl<T: PolarsDataType> ChunkedArray<T> {
    /// Whether every chunk of this array is [`flat`](polars_array::broadcast).
    pub fn is_flat(&self) -> bool {
        self.downcast_iter().all(StaticArray::is_flat)
    }

    /// Borrows this array as one whose every chunk is flat, or `None` if any chunk is scalar.
    pub fn as_flat(&self) -> Option<&Flat<Self>> {
        // SAFETY: `is_flat` is exactly the invariant of `Flat` for a `ChunkedArray`.
        self.is_flat().then(|| unsafe { Flat::new_ref(self) })
    }

    /// Returns this array with every chunk flat, writing out the chunks that are not.
    pub fn to_flat(&self) -> Cow<'_, Flat<Self>> {
        if self.is_flat() {
            // SAFETY: just checked.
            return Cow::Borrowed(unsafe { Flat::new_ref(self) });
        }

        let chunks = self
            .downcast_iter()
            .map(|arr| arr.to_flat().into_owned().into_array().into_boxed())
            .collect();

        // SAFETY: the chunks were just written out flat, and writing one out changes neither its
        // length nor which of its elements are null.
        let flat = unsafe {
            let mut out =
                Self::new_with_dims(self.field.clone(), chunks, self.length, self.null_count);
            out.set_flags(self.get_flags());
            Flat::new(out)
        };
        Cow::Owned(flat)
    }

    /// Writes out every scalar chunk of this array in place, leaving it flat.
    pub fn flatten_mut(&mut self) {
        if self.is_flat() {
            return;
        }

        let chunks: Vec<PlArrayRef> = self
            .downcast_iter()
            .map(|arr| arr.to_flat().into_owned().into_array().into_boxed())
            .collect();

        // SAFETY: writing a chunk out flat changes neither its length nor its null count, so the
        // dimensions this array carries stay correct.
        unsafe { *self.chunks_mut() = chunks };
    }
}

impl<T: PolarsNumericType> ChunkedArray<T> {
    /// The values of this array as one contiguous slice, writing out a scalar chunk only.
    pub fn to_data_views(&self) -> Vec<Cow<'_, Buffer<T::Native>>> {
        self.downcast_iter()
            .map(|arr| arr.to_flat_values())
            .collect()
    }

    pub fn to_cont_slice(&self) -> PolarsResult<Cow<'_, Buffer<T::Native>>> {
        polars_ensure!(
            self.chunks().len() == 1 && self.null_count() == 0,
            ComputeError: "chunked array is not contiguous"
        );

        Ok(self.downcast_as_array().to_flat_values())
    }
}

/// The chunks of a [`ChunkedArray`] that is known to be flat.
pub trait FlatChunkedArray<T: PolarsDataType> {
    /// The chunks, each as the flat array it is.
    fn flat_chunks(&self) -> impl DoubleEndedIterator<Item = &Flat<T::Array>>;

    /// The chunk at `idx`, or `None` if there are fewer chunks than that.
    fn flat_chunk(&self, idx: usize) -> Option<&Flat<T::Array>>;

    /// The single chunk of this array.
    fn flat_as_array(&self) -> &Flat<T::Array>;
}

impl<T: PolarsDataType> FlatChunkedArray<T> for Flat<ChunkedArray<T>> {
    #[inline]
    fn flat_chunks(&self) -> impl DoubleEndedIterator<Item = &Flat<T::Array>> {
        // SAFETY: this wrapper is the promise that every chunk is flat.
        self.as_array()
            .downcast_iter()
            .map(|arr| unsafe { Flat::new_ref(arr) })
    }

    #[inline]
    fn flat_chunk(&self, idx: usize) -> Option<&Flat<T::Array>> {
        // SAFETY: as above.
        self.as_array()
            .downcast_get(idx)
            .map(|arr| unsafe { Flat::new_ref(arr) })
    }

    #[inline]
    fn flat_as_array(&self) -> &Flat<T::Array> {
        // SAFETY: as above.
        unsafe { Flat::new_ref(self.as_array().downcast_as_array()) }
    }
}

/// The values of a numeric [`ChunkedArray`] that is known to be flat, as slices.
pub trait FlatNumericChunkedArray<T: PolarsNumericType> {
    /// The values of this array as one contiguous slice.
    fn cont_slice(&self) -> PolarsResult<&[T::Native]>;

    /// The values as one contiguous mutable slice, or `None` if there is no single run to hand out.
    fn cont_slice_mut(&mut self) -> Option<&mut [T::Native]>;

    /// The values of this array, one slice per chunk.
    fn data_views(&self) -> impl DoubleEndedIterator<Item = &[T::Native]>;
}

impl<T: PolarsNumericType> FlatNumericChunkedArray<T> for Flat<ChunkedArray<T>> {
    fn cont_slice(&self) -> PolarsResult<&[T::Native]> {
        let ca = self.as_array();
        polars_ensure!(
            ca.chunks().len() == 1 && ca.null_count() == 0,
            ComputeError: "chunked array is not contiguous"
        );
        Ok(self.flat_as_array().as_slice())
    }

    fn cont_slice_mut(&mut self) -> Option<&mut [T::Native]> {
        // SAFETY: writing over the values of a flat primitive array leaves it flat: neither its
        // length nor the number of slots its buffers hold is touched.
        let ca = unsafe { self.as_array_mut() };
        if ca.chunks().len() != 1 || ca.null_count() != 0 {
            return None;
        }

        // SAFETY: the values are only written over, so the length, the null count and the flags
        // this array carries all stay correct.
        let arr = unsafe { ca.downcast_iter_mut().next().unwrap() };
        // `flat_values_mut` is no use here: an array of a *single* element reads as scalar
        // whichever way it was built, since its values buffer holds one slot either way, and it
        // answers `None` for one. The buffer is the same in both representations, and this array
        // is flat, so it holds one slot per element whichever one it reads as.
        arr.flat_or_scalar_values_mut().get_mut_slice()
    }

    fn data_views(&self) -> impl DoubleEndedIterator<Item = &[T::Native]> {
        self.flat_chunks().map(|arr| arr.as_slice())
    }
}
