use crate::prelude::*;
use arrow::{array::*, bitmap::MutableBitmap, buffer::Buffer};
use std::convert::TryFrom;

/// Convert Arrow array offsets to indexes of the original list
pub(crate) fn offsets_to_indexes(offsets: &[i64], capacity: usize) -> AlignedVec<u32> {
    let mut idx = AlignedVec::with_capacity(capacity);

    let mut count = 0;
    let mut last_idx = 0;
    for &offset in offsets.iter().skip(1) {
        while count < offset {
            count += 1;
            idx.push(last_idx)
        }
        last_idx += 1;
    }
    for _ in 0..(capacity - count as usize) {
        idx.push(last_idx);
    }
    idx
}

impl ChunkExplode for ListChunked {
    fn explode_and_offsets(&self) -> Result<(Series, Buffer<i64>)> {
        // A list array's memory layout is actually already 'exploded', so we can just take the values array
        // of the list. And we also return a slice of the offsets. This slice can be used to find the old
        // list layout or indexes to expand the DataFrame in the same manner as the 'explode' operation
        let ca = self.rechunk();
        let listarr: &LargeListArray = ca
            .downcast_iter()
            .next()
            .ok_or_else(|| PolarsError::NoData("cannot explode empty list".into()))?;
        let offsets = listarr.offsets();
        let values = listarr.values().clone();

        let s = Series::try_from((self.name(), values)).unwrap();
        Ok((s, offsets.clone()))
    }
}

impl ChunkExplode for Utf8Chunked {
    fn explode_and_offsets(&self) -> Result<(Series, Buffer<i64>)> {
        // A list array's memory layout is actually already 'exploded', so we can just take the values array
        // of the list. And we also return a slice of the offsets. This slice can be used to find the old
        // list layout or indexes to expand the DataFrame in the same manner as the 'explode' operation
        let ca = self.rechunk();
        let array: &Utf8Array<i64> = ca
            .downcast_iter()
            .next()
            .ok_or_else(|| PolarsError::NoData("cannot explode empty str".into()))?;
        let values = array.values();
        let old_offsets = array.offsets().clone();

        // Because the strings are u8 stored but really are utf8 data we need to traverse the utf8 to
        // get the chars indexes
        // Utf8Array guarantees that this holds.
        let str_data = unsafe { std::str::from_utf8_unchecked(values) };

        // iterator over index and chars, we take only the index
        let chars = str_data
            .char_indices()
            .map(|t| t.0 as i64)
            .chain(std::iter::once(str_data.len() as i64));

        // char_indices is TrustedLen
        let offsets = unsafe { Buffer::from_trusted_len_iter_unchecked(chars) };

        // the old bitmap doesn't fit on the exploded array, so we need to create a new one.
        let validity = if let Some(validity) = array.validity() {
            let capacity = offsets.len();
            let mut bitmap = MutableBitmap::with_capacity(offsets.len() - 1);

            let mut count = 0;
            let mut last_idx = 0;
            let mut last_valid = validity.get_bit(last_idx);
            for &offset in offsets.iter().skip(1) {
                while count < offset {
                    count += 1;
                    bitmap.push(last_valid);
                }
                last_idx += 1;
                last_valid = validity.get_bit(last_idx);
            }
            for _ in 0..(capacity - count as usize) {
                bitmap.push(last_valid);
            }
            bitmap.into()
        } else {
            None
        };
        let array =
            unsafe { Utf8Array::<i64>::from_data_unchecked(offsets, values.clone(), validity) };

        let new_arr = Arc::new(array) as ArrayRef;

        let s = Series::try_from((self.name(), new_arr)).unwrap();
        Ok((s, old_offsets))
    }
}
