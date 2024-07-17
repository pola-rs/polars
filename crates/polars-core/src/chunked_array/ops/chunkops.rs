use arrow::legacy::kernels::concatenate::concatenate_owned_unchecked;
use polars_error::constants::LENGTH_LIMIT_MSG;

use super::*;
use crate::chunked_array::metadata::MetadataProperties;
#[cfg(feature = "object")]
use crate::chunked_array::object::builder::ObjectChunkedBuilder;
use crate::utils::slice_offsets;

pub(crate) fn split_at(
    chunks: &[ArrayRef],
    offset: i64,
    own_length: usize,
) -> (Vec<ArrayRef>, Vec<ArrayRef>) {
    let mut new_chunks_left = Vec::with_capacity(1);
    let mut new_chunks_right = Vec::with_capacity(1);
    let (raw_offset, _) = slice_offsets(offset, 0, own_length);

    let mut remaining_offset = raw_offset;
    let mut iter = chunks.iter();

    for chunk in &mut iter {
        let chunk_len = chunk.len();
        if remaining_offset > 0 && remaining_offset >= chunk_len {
            remaining_offset -= chunk_len;
            new_chunks_left.push(chunk.clone());
            continue;
        }

        let (l, r) = chunk.split_at_boxed(remaining_offset);
        new_chunks_left.push(l);
        new_chunks_right.push(r);
        break;
    }

    for chunk in iter {
        new_chunks_right.push(chunk.clone())
    }
    if new_chunks_left.is_empty() {
        new_chunks_left.push(chunks[0].sliced(0, 0));
    }
    if new_chunks_right.is_empty() {
        new_chunks_right.push(chunks[0].sliced(0, 0));
    }
    (new_chunks_left, new_chunks_right)
}

pub(crate) fn slice(
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
            // SAFETY:
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
    #[inline]
    pub fn len(&self) -> usize {
        self.length as usize
    }

    /// Return the number of null values in the ChunkedArray.
    #[inline]
    pub fn null_count(&self) -> usize {
        self.null_count as usize
    }

    /// Set the null count directly.
    ///
    /// This can be useful after mutably adjusting the validity of the
    /// underlying arrays.
    ///
    /// # Safety
    /// The new null count must match the total null count of the underlying
    /// arrays.
    pub unsafe fn set_null_count(&mut self, null_count: IdxSize) {
        self.null_count = null_count;
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
        let len = inner(&self.chunks);
        // Length limit is `IdxSize::MAX - 1`. We use `IdxSize::MAX` to indicate `NULL` in indexing.
        assert!(len < IdxSize::MAX as usize, "{}", LENGTH_LIMIT_MSG);
        self.length = len as IdxSize;
        self.null_count = self
            .chunks
            .iter()
            .map(|arr| arr.null_count())
            .sum::<usize>() as IdxSize;
    }

    pub fn rechunk(&self) -> Self {
        match self.dtype() {
            #[cfg(feature = "object")]
            DataType::Object(_, _) => {
                panic!("implementation error")
            },
            _ => {
                fn inner_rechunk(chunks: &[ArrayRef]) -> Vec<ArrayRef> {
                    vec![concatenate_owned_unchecked(chunks).unwrap()]
                }

                if self.chunks.len() == 1 {
                    self.clone()
                } else {
                    let chunks = inner_rechunk(&self.chunks);

                    let mut ca = unsafe { self.copy_with_chunks(chunks) };

                    use MetadataProperties as P;
                    ca.copy_metadata(
                        self,
                        P::SORTED
                            | P::FAST_EXPLODE_LIST
                            | P::MIN_VALUE
                            | P::MAX_VALUE
                            | P::DISTINCT_COUNT,
                    );

                    ca
                }
            },
        }
    }

    /// Split the array. The chunks are reallocated the underlying data slices are zero copy.
    ///
    /// When offset is negative it will be counted from the end of the array.
    /// This method will never error,
    /// and will slice the best match when offset, or length is out of bounds
    pub fn split_at(&self, offset: i64) -> (Self, Self) {
        // A normal slice, slice the buffers and thus keep the whole memory allocated.
        let (l, r) = split_at(&self.chunks, offset, self.len());
        let mut out_l = unsafe { self.copy_with_chunks(l) };
        let mut out_r = unsafe { self.copy_with_chunks(r) };

        use MetadataProperties as P;
        let mut properties_l = P::SORTED | P::FAST_EXPLODE_LIST;
        let mut properties_r = P::SORTED | P::FAST_EXPLODE_LIST;

        let is_ascending = self.is_sorted_ascending_flag();
        let is_descending = self.is_sorted_descending_flag();

        if is_ascending || is_descending {
            let has_nulls_at_start = self.null_count() != 0
                && self
                    .chunks()
                    .first()
                    .unwrap()
                    .as_ref()
                    .validity()
                    .map_or(false, |bm| bm.get(0).unwrap());

            if !has_nulls_at_start {
                let can_copy_min_value = !has_nulls_at_start && is_ascending;
                let can_copy_max_value = !has_nulls_at_start && is_descending;

                properties_l.set(P::MIN_VALUE, can_copy_min_value);
                properties_l.set(P::MAX_VALUE, can_copy_max_value);
            }

            let has_nulls_at_end = self.null_count() != 0
                && self
                    .chunks()
                    .last()
                    .unwrap()
                    .as_ref()
                    .validity()
                    .map_or(false, |bm| bm.get(bm.len() - 1).unwrap());

            if !has_nulls_at_end {
                let can_copy_min_value = !has_nulls_at_end && is_descending;
                let can_copy_max_value = !has_nulls_at_end && is_ascending;
                properties_r.set(P::MIN_VALUE, can_copy_min_value);
                properties_r.set(P::MAX_VALUE, can_copy_max_value);
            }
        }
        out_l.copy_metadata(self, properties_l);
        out_r.copy_metadata(self, properties_r);

        (out_l, out_r)
    }

    /// Slice the array. The chunks are reallocated the underlying data slices are zero copy.
    ///
    /// When offset is negative it will be counted from the end of the array.
    /// This method will never error,
    /// and will slice the best match when offset, or length is out of bounds
    pub fn slice(&self, offset: i64, length: usize) -> Self {
        // The len: 0 special cases ensure we release memory.
        // A normal slice, slice the buffers and thus keep the whole memory allocated.
        let exec = || {
            let (chunks, len) = slice(&self.chunks, offset, length, self.len());
            let mut out = unsafe { self.copy_with_chunks(chunks) };

            use MetadataProperties as P;
            let mut properties = P::SORTED | P::FAST_EXPLODE_LIST;

            let is_ascending = self.is_sorted_ascending_flag();
            let is_descending = self.is_sorted_descending_flag();

            if length != 0 && (is_ascending || is_descending) {
                let (raw_offset, slice_len) = slice_offsets(offset, length, self.len());

                let mut can_copy_min_value = false;
                let mut can_copy_max_value = false;

                let is_at_start = raw_offset == 0;
                if is_at_start {
                    let has_nulls_at_start = self.null_count() != 0
                        && self
                            .chunks()
                            .first()
                            .unwrap()
                            .as_ref()
                            .validity()
                            .map_or(false, |bm| bm.get(0).unwrap());

                    can_copy_min_value |= !has_nulls_at_start && is_ascending;
                    can_copy_max_value |= !has_nulls_at_start && is_descending;
                }

                let is_until_end = raw_offset + slice_len == self.len();
                if is_until_end {
                    let has_nulls_at_end = self.null_count() != 0
                        && self
                            .chunks()
                            .last()
                            .unwrap()
                            .as_ref()
                            .validity()
                            .map_or(false, |bm| bm.get(bm.len() - 1).unwrap());

                    can_copy_min_value |= !has_nulls_at_end && is_descending;
                    can_copy_max_value |= !has_nulls_at_end && is_ascending;
                }

                properties.set(P::MIN_VALUE, can_copy_min_value);
                properties.set(P::MAX_VALUE, can_copy_max_value);
            }

            out.copy_metadata(self, properties);
            out.length = len as IdxSize;

            out
        };

        match length {
            0 => match self.dtype() {
                #[cfg(feature = "object")]
                DataType::Object(_, _) => exec(),
                _ => self.clear(),
            },
            _ => exec(),
        }
    }

    /// Take a view of top n elements
    #[must_use]
    pub fn limit(&self, num_elements: usize) -> Self
    where
        Self: Sized,
    {
        self.slice(0, num_elements)
    }

    /// Get the head of the [`ChunkedArray`]
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

    /// Get the tail of the [`ChunkedArray`]
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

    /// Remove empty chunks.
    pub fn prune_empty_chunks(&mut self) {
        let mut count = 0u32;
        unsafe {
            self.chunks_mut().retain(|arr| {
                count += 1;
                // Always keep at least one chunk
                if count == 1 {
                    true
                } else {
                    // Remove the empty chunks
                    arr.len() > 0
                }
            })
        }
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
            if !self.has_nulls() {
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
        let mut a = s
            .cast(&DataType::Categorical(None, Default::default()))
            .unwrap();

        a.append(&a.slice(0, 2)).unwrap();
        let a = a.rechunk();
        assert!(a.categorical().unwrap().get_rev_map().len() > 0);
    }
}
