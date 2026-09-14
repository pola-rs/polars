use arrow::bitmap::Bitmap;

use super::*;

impl<T: PolarsDataType> ChunkedArray<T> {
    /// Get a mask of the null values.
    pub fn is_null(&self) -> BooleanChunked {
        if !self.has_nulls() {
            return BooleanChunked::full(self.name().clone(), false, self.len());
        }
        // dispatch to non-generic function
        is_null(self.name().clone(), &self.chunks)
    }

    /// Get a mask of the valid values.
    pub fn is_not_null(&self) -> BooleanChunked {
        if self.null_count() == 0 {
            return BooleanChunked::full(self.name().clone(), true, self.len());
        }
        // dispatch to non-generic function
        is_not_null(self.name().clone(), &self.chunks)
    }
}

pub fn is_not_null(name: PlSmallStr, chunks: &[ArrayRef]) -> BooleanChunked {
    let chunks = chunks.iter().map(|arr| {
        let bitmap = arr
            .validity()
            .cloned()
            .unwrap_or_else(|| !(&Bitmap::new_zeroed(arr.len())));
        BooleanArray::from_data_default(bitmap, None)
    });
    BooleanChunked::from_chunk_iter(name, chunks)
}

pub fn is_null(name: PlSmallStr, chunks: &[ArrayRef]) -> BooleanChunked {
    let chunks = chunks.iter().map(|arr| {
        let bitmap = arr
            .validity()
            .map(|bitmap| !bitmap)
            .unwrap_or_else(|| Bitmap::new_zeroed(arr.len()));
        BooleanArray::from_data_default(bitmap, None)
    });
    BooleanChunked::from_chunk_iter(name, chunks)
}

pub fn replace_non_null(name: PlSmallStr, chunks: &[ArrayRef], default: bool) -> BooleanChunked {
    BooleanChunked::from_chunk_iter(
        name,
        chunks.iter().map(|el| {
            BooleanArray::from_data_default(
                Bitmap::new_with_value(default, el.len()),
                el.validity().cloned(),
            )
        }),
    )
}
