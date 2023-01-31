use crate::prelude::*;
use crate::series::IsSorted;

pub(crate) fn new_chunks(chunks: &mut Vec<ArrayRef>, other: &[ArrayRef], len: usize) {
    // replace an empty array
    if chunks.len() == 1 && len == 0 {
        *chunks = other.to_owned();
    } else {
        chunks.extend_from_slice(other);
    }
}

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    /// Append in place. This is done by adding the chunks of `other` to this [`ChunkedArray`].
    ///
    /// See also [`extend`](Self::extend) for appends to the underlying memory
    pub fn append(&mut self, other: &Self) {
        let len = self.len();
        self.length += other.length;
        new_chunks(&mut self.chunks, &other.chunks, len);
        self.set_sorted_flag(IsSorted::Not);
    }
}

#[doc(hidden)]
impl BooleanChunked {
    pub fn append(&mut self, other: &Self) {
        let len = self.len();
        self.length += other.length;
        new_chunks(&mut self.chunks, &other.chunks, len);
        self.set_sorted_flag(IsSorted::Not);
    }
}
#[doc(hidden)]
impl Utf8Chunked {
    pub fn append(&mut self, other: &Self) {
        let len = self.len();
        self.length += other.length;
        new_chunks(&mut self.chunks, &other.chunks, len);
        self.set_sorted_flag(IsSorted::Not);
    }
}

#[cfg(feature = "dtype-binary")]
#[doc(hidden)]
impl BinaryChunked {
    pub fn append(&mut self, other: &Self) {
        let len = self.len();
        self.length += other.length;
        new_chunks(&mut self.chunks, &other.chunks, len);
        self.set_sorted_flag(IsSorted::Not);
    }
}

#[doc(hidden)]
impl ListChunked {
    pub fn append(&mut self, other: &Self) -> PolarsResult<()> {
        let dtype = merge_dtypes(self.dtype(), other.dtype())?;
        self.field = Arc::new(Field::new(self.name(), dtype));

        let len = self.len();
        self.length += other.length;
        new_chunks(&mut self.chunks, &other.chunks, len);
        self.set_sorted_flag(IsSorted::Not);
        if !other._can_fast_explode() {
            self.unset_fast_explode()
        }
        Ok(())
    }
}
#[cfg(feature = "object")]
#[doc(hidden)]
impl<T: PolarsObject> ObjectChunked<T> {
    pub fn append(&mut self, other: &Self) {
        let len = self.len();
        self.length += other.length;
        self.set_sorted_flag(IsSorted::Not);
        new_chunks(&mut self.chunks, &other.chunks, len);
    }
}
