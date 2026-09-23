use polars_array::bitmap::invert;

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

/// The mask of a chunk, as the boolean array of which elements are not null.
pub fn is_not_null(name: PlSmallStr, chunks: &[PlArrayRef]) -> BooleanChunked {
    let chunks = chunks.iter().map(|arr| match arr.validity() {
        Some(validity) => PlBooleanArray::from_pl_bitmap(PlBitmap::from(validity)),
        None => PlBooleanArray::new_scalar(true, arr.len()),
    });
    BooleanChunked::from_chunk_iter(name, chunks)
}

/// The mask of a chunk, as the boolean array of which elements are null — see [`is_not_null`].
pub fn is_null(name: PlSmallStr, chunks: &[PlArrayRef]) -> BooleanChunked {
    let chunks = chunks.iter().map(|arr| match arr.validity() {
        Some(validity) => PlBooleanArray::from_pl_bitmap(invert(validity)),
        None => PlBooleanArray::new_scalar(false, arr.len()),
    });
    BooleanChunked::from_chunk_iter(name, chunks)
}

pub fn replace_non_null(name: PlSmallStr, chunks: &[PlArrayRef], default: bool) -> BooleanChunked {
    BooleanChunked::from_chunk_iter(
        name,
        chunks.iter().map(|el| {
            PlBooleanArray::new_scalar(default, el.len())
                .with_validity(el.validity().map(PlBitmap::from))
        }),
    )
}
