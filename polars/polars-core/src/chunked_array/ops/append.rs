use crate::prelude::*;

fn new_chunks(chunks: &mut Vec<ArrayRef>, other: &[ArrayRef], len: usize) {
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
    /// Append in place.
    pub fn append(&mut self, other: &Self) {
        let len = self.len();
        new_chunks(&mut self.chunks, &other.chunks, len);
    }
}

impl BooleanChunked {
    pub fn append(&mut self, other: &Self) {
        let len = self.len();
        new_chunks(&mut self.chunks, &other.chunks, len);
    }
}
impl Utf8Chunked {
    pub fn append(&mut self, other: &Self) {
        let len = self.len();
        new_chunks(&mut self.chunks, &other.chunks, len);
    }
}

impl ListChunked {
    pub fn append(&mut self, other: &Self) {
        let len = self.len();
        new_chunks(&mut self.chunks, &other.chunks, len);
    }
}
#[cfg(feature = "object")]
impl<T: PolarsObject> ObjectChunked<T> {
    pub fn append(&mut self, other: &Self) {
        let len = self.len();
        new_chunks(&mut self.chunks, &other.chunks, len);
    }
}
#[cfg(feature = "dtype-categorical")]
impl CategoricalChunked {
    pub fn append(&mut self, other: &Self) {
        if let (Some(rev_map_l), Some(rev_map_r)) = (
            self.categorical_map.as_ref(),
            other.categorical_map.as_ref(),
        ) {
            // first assertion checks if the global string cache is equal,
            // the second checks if we append a slice from this array to self
            if !rev_map_l.same_src(rev_map_r) && !Arc::ptr_eq(rev_map_l, rev_map_r) {
                panic!("Appending categorical data can only be done if they are made under the same global string cache. \
                Consider using a global string cache.")
            }

            let new_rev_map = self.merge_categorical_map(other);
            self.categorical_map = Some(new_rev_map);
        }

        let len = self.len();
        new_chunks(&mut self.chunks, &other.chunks, len);
    }
}
