#[cfg(feature = "object")]
use arrow::array::Array;
use arrow::compute::concatenate;

use super::*;
#[cfg(feature = "object")]
use crate::chunked_array::object::builder::ObjectChunkedBuilder;
use crate::utils::slice_offsets;

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
            new_chunks.push(chunk.sliced_unchecked(remaining_offset, take_len));
        }
        remaining_length -= take_len;
        remaining_offset = 0;
        if remaining_length == 0 {
            break;
        }
    }
    if new_chunks.is_empty() {
        new_chunks.push(chunks[0].sliced(0, 0));
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
        self.length = inner(&self.chunks) as IdxSize;
        #[cfg(feature = "python")]
        assert!(
            self.length < IdxSize::MAX,
            "Polars' maximum length reached. Consider installing 'polars-u64-idx'."
        );
        #[cfg(not(feature = "python"))]
        assert!(
            self.length < IdxSize::MAX,
            "Polars' maximum length reached. Consider compiling with 'bigidx' feature."
        );
    }

    pub fn rechunk(&self) -> Self {
        match self.dtype() {
            #[cfg(feature = "object")]
            DataType::Object(_) => {
                panic!("implementation error")
            }
            _ => {
                fn inner_rechunk(chunks: &[ArrayRef]) -> Vec<ArrayRef> {
                    vec![concatenate::concatenate(
                        chunks.iter().map(|a| &**a).collect::<Vec<_>>().as_slice(),
                    )
                    .unwrap()]
                }

                if self.chunks.len() == 1 {
                    self.clone()
                } else {
                    let chunks = inner_rechunk(&self.chunks);
                    self.copy_with_chunks(chunks, true)
                }
            }
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

#[cfg(feature = "object")]
impl<T: PolarsObject> ObjectChunked<T> {
    pub(crate) fn rechunk_object(&self) -> Self {
        if self.chunks.len() == 1 {
            self.clone()
        } else {
            let mut builder = ObjectChunkedBuilder::new(self.name(), self.len());
            let chunks = self.downcast_iter();

            // todo! use iterators once implemented
            // no_null path
            if !self.has_validity() {
                for arr in chunks {
                    for idx in 0..arr.len() {
                        builder.append_value(arr.value(idx).clone())
                    }
                }
            } else {
                for arr in chunks {
                    for idx in 0..arr.len() {
                        if arr.is_valid(idx) {
                            builder.append_value(arr.value(idx).clone())
                        } else {
                            builder.append_null()
                        }
                    }
                }
            }
            builder.finish()
        }
    }
}

#[cfg(test)]
mod test {
    #[cfg(feature = "dtype-categorical")]
    use crate::prelude::*;

    #[test]
    #[cfg(feature = "dtype-categorical")]
    fn test_categorical_map_after_rechunk() {
        let s = Series::new("", &["foo", "bar", "spam"]);
        let mut a = s.cast(&DataType::Categorical(None)).unwrap();

        a.append(&a.slice(0, 2)).unwrap();
        let a = a.rechunk();
        assert!(a.categorical().unwrap().get_rev_map().len() > 0);
    }
}
