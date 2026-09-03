//! A [`ChunkedArray`] whose every chunk is flat, which is what handing out a *slice* of the values
//! needs: a [`scalar`](polars_array::broadcast) chunk has one value where a slice needs `len`.

use std::borrow::Cow;

use polars_array::{Flat, StaticArray};

use crate::prelude::*;

impl<T: PolarsDataType> ChunkedArray<T> {
    /// Whether every chunk of this array is [`flat`](polars_array::broadcast). This is what
    /// [`ChunkedArray::as_flat`] answers with a borrow rather than with a `bool`.
    pub fn is_flat(&self) -> bool {
        self.downcast_iter().all(StaticArray::is_flat)
    }

    /// Borrows this array as one whose every chunk is flat, or `None` if any chunk is scalar. This
    /// is the `O(n_chunks)` half of [`ChunkedArray::to_flat`]: it writes nothing out.
    pub fn as_flat(&self) -> Option<&Flat<Self>> {
        // SAFETY: `is_flat` is exactly the invariant of `Flat` for a `ChunkedArray`.
        self.is_flat().then(|| unsafe { Flat::new_ref(self) })
    }

    /// Returns this array with every chunk in the flat representation: `O(n_chunks)` for an array
    /// that is laid out flat, `O(len)` for the chunks of one that is not, which are written out.
    pub fn to_flat(&self) -> Cow<'_, Flat<Self>> {
        if self.is_flat() {
            // SAFETY: just checked.
            return Cow::Borrowed(unsafe { Flat::new_ref(self) });
        }

        let chunks = self
            .downcast_iter()
            .map(|arr| arr.to_flat().into_array().into_boxed())
            .collect();

        // SAFETY: the chunks were just written out flat, and writing one out changes neither its
        // length nor which of its elements are null.
        let flat = unsafe {
            let mut out = Self::new_with_dims(
                self.field.clone(),
                chunks,
                self.length,
                self.null_count,
            );
            out.set_flags(self.get_flags());
            Flat::new(out)
        };
        Cow::Owned(flat)
    }

    /// Writes out every scalar chunk of this array in place, leaving it flat. This is
    /// [`ChunkedArray::to_flat`] for a caller that needs the array *itself* to be flat.
    pub fn flatten_mut(&mut self) {
        if self.is_flat() {
            return;
        }

        let chunks: Vec<PlArrayRef> = self
            .downcast_iter()
            .map(|arr| arr.to_flat().into_array().into_boxed())
            .collect();

        // SAFETY: writing a chunk out flat changes neither its length nor its null count, so the
        // dimensions this array carries stay correct.
        unsafe { *self.chunks_mut() = chunks };
    }
}

/// The chunks of a [`ChunkedArray`] that is known to be flat. An extension trait because [`Flat`]
/// belongs to `polars-array`, which is what keeps the array it wraps out of reach.
pub trait FlatChunkedArray<T: PolarsDataType> {
    /// The chunks, each as the flat array it is.
    fn flat_chunks(&self) -> impl DoubleEndedIterator<Item = &Flat<T::Array>>;

    /// The chunk at `idx`, or `None` if there are fewer chunks than that.
    fn flat_chunk(&self, idx: usize) -> Option<&Flat<T::Array>>;

    /// The single chunk of this array. Panics if this array does not have exactly one chunk.
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

/// The values of a numeric [`ChunkedArray`] that is known to be flat, as slices. This is what the
/// flatness is for: a flat chunk holds one slot per element, so its values are a `&[T::Native]`.
pub trait FlatNumericChunkedArray<T: PolarsNumericType> {
    /// The values of this array as one contiguous slice. Errors if this array has more than one
    /// chunk, or any null: neither leaves one run of values to hand out.
    fn cont_slice(&self) -> PolarsResult<&[T::Native]>;

    /// The values of this array as one contiguous mutable slice, or `None` if there is no single
    /// run of them to hand out, or the buffer holding them is shared with another array.
    fn cont_slice_mut(&mut self) -> Option<&mut [T::Native]>;

    /// The values of this array, one slice per chunk. NOTE: null values should be taken into
    /// account by the user of these slices, as they are handled separately.
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
        arr.flat_values_mut()
            .expect("a chunk of a flat ChunkedArray is flat")
            .get_mut_slice()
    }

    fn data_views(&self) -> impl DoubleEndedIterator<Item = &[T::Native]> {
        self.flat_chunks().map(|arr| arr.as_slice())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn a_scalar_chunk_is_written_out_by_to_flat() {
        // `full` repeats one value in `O(1)`, which is the scalar representation.
        let ca = Int32Chunked::full(PlSmallStr::EMPTY, 7, 3);
        assert!(!ca.is_flat());
        assert!(ca.as_flat().is_none());

        let flat = ca.to_flat();
        assert!(matches!(flat, Cow::Owned(_)));
        assert_eq!(flat.cont_slice().unwrap(), [7, 7, 7]);

        // The array itself is untouched: `to_flat` writes out a copy.
        assert!(!ca.is_flat());
    }

    #[test]
    fn a_flat_chunk_is_borrowed_rather_than_copied() {
        let ca = Int32Chunked::new(PlSmallStr::EMPTY, &[1, 2, 3]);
        assert!(ca.is_flat());

        let flat = ca.to_flat();
        assert!(matches!(flat, Cow::Borrowed(_)));
        assert_eq!(flat.cont_slice().unwrap(), [1, 2, 3]);
        assert_eq!(flat.data_views().next().unwrap(), [1, 2, 3]);
    }

    #[test]
    fn flatten_mut_leaves_the_array_itself_flat() {
        let mut ca = Int32Chunked::full(PlSmallStr::EMPTY, 7, 3);
        ca.flatten_mut();

        assert!(ca.is_flat());
        assert_eq!(ca.as_flat().unwrap().cont_slice().unwrap(), [7, 7, 7]);
        assert_eq!(ca.len(), 3);
        assert_eq!(ca.null_count(), 0);
    }

    #[test]
    fn cont_slice_needs_one_chunk_and_no_nulls() {
        let mut ca = Int32Chunked::new(PlSmallStr::EMPTY, &[1, 2]);
        ca.append(&Int32Chunked::new(PlSmallStr::EMPTY, &[3])).unwrap();
        assert!(ca.to_flat().cont_slice().is_err());

        let ca = Int32Chunked::new(PlSmallStr::EMPTY, &[Some(1), None]);
        assert!(ca.to_flat().cont_slice().is_err());
    }
}
