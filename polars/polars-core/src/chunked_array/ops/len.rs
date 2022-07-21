use arrow::compute::concatenate;
use crate::utils::slice_offsets;
use super::*;

#[inline]
fn slice(
    chunks: &[ArrayRef],
    offset: i64,
    slice_length: usize,
    own_length: usize,
) -> (Vec<ArrayRef>, usize) {
    let mut new_chunks = Vec::with_capacity(1);
    let (raw_offset, slice_len) = slice_offsets(offset, slice_length, own_length);

    let mut remaining_length = slice_len;
    let mut remaining_offset = raw_offset;
    let mut new_len = 0;

    for chunk in chunks {
        let chunk_len = chunk.len();
        if remaining_offset > 0 && remaining_offset >= chunk_len {
            remaining_offset -= chunk_len;
            continue;
        }
        let take_len = if remaining_length + remaining_offset > chunk_len {
            chunk_len - remaining_offset
        } else {
            remaining_length
        };
        new_len += take_len;

        debug_assert!(remaining_offset + take_len <= chunk.len());
        unsafe {
            // Safety:
            // this function ensures the slices are in bounds
            new_chunks.push(chunk.slice_unchecked(remaining_offset, take_len));
        }
        remaining_length -= take_len;
        remaining_offset = 0;
        if remaining_length == 0 {
            break;
        }
    }
    if new_chunks.is_empty() {
        new_chunks.push(chunks[0].slice(0, 0));
    }
    (new_chunks, new_len)
}

impl<T: PolarsDataType> ChunkedArray<T> {


    /// Get the length of the ChunkedArray
    pub fn len(&self) -> usize {
        self.length as usize
    }

    /// Check if ChunkedArray is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Compute the length
    pub(crate) fn compute_len(&mut self) {
        fn inner(chunks: &[ArrayRef]) -> usize {
            match chunks.len() {
                // fast path
                1 => chunks[0].len(),
                _ => chunks.iter().fold(0, |acc, arr| acc + arr.len()),
            }
        }
        self.length = inner(&self.chunks) as IdxSize
    }

    pub fn rechunk(&self) -> Self {
        if self.chunks().len() == 1 {
            self.clone()
        } else {
            let chunks = vec![concatenate::concatenate(
                self.chunks
                    .iter()
                    .map(|a| &**a)
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
                .unwrap()];
            ChunkedArray::from_chunks(self.name(), chunks)
        }
    }

    /// Slice the array. The chunks are reallocated the underlying data slices are zero copy.
    ///
    /// When offset is negative it will be counted from the end of the array.
    /// This method will never error,
    /// and will slice the best match when offset, or length is out of bounds
    #[inline]
    pub fn slice(&self, offset: i64, length: usize) -> Self {
        let (chunks, len) = slice(&self.chunks, offset, length, self.len());
        let mut out = self.copy_with_chunks(chunks, true);
        out.length = len as IdxSize;
        out
    }

    // #[must_use]
    // pub fn rechunk(&self) -> Self
    //     where
    //         Self: std::marker::Sized;

    // /// Slice the array. The chunks are reallocated the underlying data slices are zero copy.
    // ///
    // /// When offset is negative it will be counted from the end of the array.
    // /// This method will never error,
    // /// and will slice the best match when offset, or length is out of bounds
    // #[must_use]
    // pub fn slice(&self, offset: i64, length: usize) -> Self
    //     where
    //         Self: std::marker::Sized;

    /// Take a view of top n elements
    #[must_use]
    pub fn limit(&self, num_elements: usize) -> Self
        where
            Self: Sized,
    {
        self.slice(0, num_elements)
    }

    /// Get the head of the ChunkedArray
    #[must_use]
    pub fn head(&self, length: Option<usize>) -> Self
        where
            Self: Sized,
    {
        match length {
            Some(len) => self.slice(0, std::cmp::min(len, self.len())),
            None => self.slice(0, std::cmp::min(10, self.len())),
        }
    }

    /// Get the tail of the ChunkedArray
    #[must_use]
    pub fn tail(&self, length: Option<usize>) -> Self
        where
            Self: Sized,
    {
        let len = match length {
            Some(len) => std::cmp::min(len, self.len()),
            None => std::cmp::min(10, self.len()),
        };
        self.slice(-(len as i64), len)
    }

}
