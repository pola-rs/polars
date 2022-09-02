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
        self.set_sorted2(IsSorted::Not);
    }
}

#[doc(hidden)]
impl BooleanChunked {
    pub fn append(&mut self, other: &Self) {
        let len = self.len();
        self.length += other.length;
        new_chunks(&mut self.chunks, &other.chunks, len);
        self.set_sorted2(IsSorted::Not);
    }
}
#[doc(hidden)]
impl Utf8Chunked {
    pub fn append(&mut self, other: &Self) {
        let len = self.len();
        self.length += other.length;
        new_chunks(&mut self.chunks, &other.chunks, len);
        self.set_sorted2(IsSorted::Not);
    }
}

#[doc(hidden)]
impl ListChunked {
    pub fn append(&mut self, other: &Self) -> Result<()> {
        // todo! there should be a merge-dtype function, that goes all the way down.

        #[cfg(feature = "dtype-categorical")]
        use DataType::*;
        #[cfg(feature = "dtype-categorical")]
        if let (Categorical(Some(rev_map_l)), Categorical(Some(rev_map_r))) =
            (self.inner_dtype(), other.inner_dtype())
        {
            if !rev_map_l.same_src(rev_map_r.as_ref()) {
                let rev_map = merge_categorical_map(&rev_map_l, &rev_map_r)?;
                self.field = Arc::new(Field::new(
                    self.name(),
                    DataType::List(Box::new(DataType::Categorical(Some(rev_map)))),
                ));
            }
        }

        let len = self.len();
        self.length += other.length;
        new_chunks(&mut self.chunks, &other.chunks, len);
        Ok(())
    }
}
#[cfg(feature = "object")]
#[doc(hidden)]
impl<T: PolarsObject> ObjectChunked<T> {
    pub fn append(&mut self, other: &Self) {
        let len = self.len();
        self.length += other.length;
        new_chunks(&mut self.chunks, &other.chunks, len);
    }
}
