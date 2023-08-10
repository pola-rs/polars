use arrow::bitmap::Bitmap;

use super::*;

impl<T: PolarsDataType> ChunkedArray<T> {
    /// Get a mask of the null values.
    pub fn is_null(&self) -> BooleanChunked {
        if !self.has_validity() {
            return BooleanChunked::full(self.name(), false, self.len());
        }
        // dispatch to non-generic function
        is_null(self.name(), &self.chunks)
    }

    /// Get a mask of the valid values.
    pub fn is_not_null(&self) -> BooleanChunked {
        if self.null_count() == 0 {
            return BooleanChunked::full(self.name(), true, self.len());
        }
        // dispatch to non-generic function
        is_not_null(self.name(), &self.chunks)
    }

    pub(crate) fn coalesce_nulls(&self, other: &[ArrayRef]) -> Self {
        let chunks = coalesce_nulls(&self.chunks, other);
        unsafe { self.copy_with_chunks(chunks, true, false) }
    }
}

pub fn is_not_null(name: &str, chunks: &[ArrayRef]) -> BooleanChunked {
    let chunks = chunks.iter().map(|arr| {
        let bitmap = arr
            .validity()
            .cloned()
            .unwrap_or_else(|| !(&Bitmap::new_zeroed(arr.len())));
        BooleanArray::from_data_default(bitmap, None)
    });
    BooleanChunked::from_chunk_iter(name, chunks)
}

pub fn is_null(name: &str, chunks: &[ArrayRef]) -> BooleanChunked {
    let chunks = chunks.iter().map(|arr| {
        let bitmap = arr
            .validity()
            .map(|bitmap| !bitmap)
            .unwrap_or_else(|| Bitmap::new_zeroed(arr.len()));
        BooleanArray::from_data_default(bitmap, None)
    });
    BooleanChunked::from_chunk_iter(name, chunks)
}

pub(crate) fn coalesce_nulls(chunks: &[ArrayRef], other: &[ArrayRef]) -> Vec<ArrayRef> {
    assert_eq!(chunks.len(), other.len());
    chunks
        .iter()
        .zip(other)
        .map(|(a, b)| {
            assert_eq!(a.len(), b.len());
            let validity = match (a.validity(), b.validity()) {
                (None, Some(b)) => Some(b.clone()),
                (Some(a), Some(b)) => Some(a & b),
                (Some(a), None) => Some(a.clone()),
                (None, None) => None,
            };

            a.with_validity(validity)
        })
        .collect()
}
