use crate::prelude::*;
use arrow::array::{ArrayRef, BooleanBufferBuilder};
use arrow::datatypes::ToByteSlice;
use arrow::{
    array::{Array, ArrayData, LargeListArray, LargeStringArray},
    buffer::Buffer,
};
use itertools::Itertools;
use std::convert::TryFrom;

/// Convert Arrow array offsets to indexes of the original list
pub(crate) fn offsets_to_indexes(offsets: &[i64], capacity: usize) -> AlignedVec<u32> {
    let mut idx = AlignedVec::with_capacity_aligned(capacity);

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
    fn explode_and_offsets(&self) -> Result<(Series, &[i64])> {
        // A list array's memory layout is actually already 'exploded', so we can just take the values array
        // of the list. And we also return a slice of the offsets. This slice can be used to find the old
        // list layout or indexes to expand the DataFrame in the same manner as the 'explode' operation
        let ca = self.rechunk();
        let listarr: &LargeListArray = ca
            .downcast_iter()
            .next()
            .ok_or_else(|| PolarsError::NoData("cannot explode empty list".into()))?;
        let list_data = listarr.data();
        let values = listarr.values();
        let offset_ptr = list_data.buffers()[0].as_ptr() as *const i64;
        // offsets in the list array. These indicate where a new list starts
        let offsets = unsafe { std::slice::from_raw_parts(offset_ptr, self.len()) };

        let s = Series::try_from((self.name(), values)).unwrap();
        Ok((s, offsets))
    }
}

impl ChunkExplode for Utf8Chunked {
    fn explode_and_offsets(&self) -> Result<(Series, &[i64])> {
        // A list array's memory layout is actually already 'exploded', so we can just take the values array
        // of the list. And we also return a slice of the offsets. This slice can be used to find the old
        // list layout or indexes to expand the DataFrame in the same manner as the 'explode' operation
        let ca = self.rechunk();
        let stringarr: &LargeStringArray = ca
            .downcast_iter()
            .next()
            .ok_or_else(|| PolarsError::NoData("cannot explode empty str".into()))?;
        let list_data = stringarr.data();
        let str_values_buf = stringarr.value_data();

        // We get the offsets of the strings in the original array
        let offset_ptr = list_data.buffers()[0].as_ptr() as *const i64;
        // offsets in the list array. These indicate where a new list starts
        let offsets = unsafe { std::slice::from_raw_parts(offset_ptr, self.len()) };

        // Because the strings are u8 stored but really are utf8 data we need to traverse the utf8 to
        // get the chars indexes
        let str_data = unsafe { std::str::from_utf8_unchecked(str_values_buf.as_slice()) };
        // iterator over index and chars, we take only the index
        // todo! directly create a buffer from an aligned vec or a mutable buffer
        let mut new_offsets = str_data.char_indices().map(|t| t.0 as i64).collect_vec();
        // somehow I don't get the last value if we don't add this one.
        new_offsets.push(str_data.len() as i64);

        // first buffer are the offsets. We now have only a single offset
        // second buffer is the actual values buffer
        let mut builder = ArrayData::builder(ArrowDataType::LargeUtf8)
            .len(new_offsets.len() - 1)
            .add_buffer(Buffer::from(new_offsets.to_byte_slice()))
            .add_buffer(str_values_buf);

        // the old bitmap doesn't fit on the exploded array, so we need to create a new one.
        if self.null_count() > 0 {
            let capacity = new_offsets.len();
            let mut bitmap_builder = BooleanBufferBuilder::new(new_offsets.len());

            let mut count = 0;
            let mut last_idx = 0;
            let mut last_valid = stringarr.is_valid(last_idx);
            for &offset in offsets.iter().skip(1) {
                while count < offset {
                    count += 1;
                    bitmap_builder.append(last_valid);
                }
                last_idx += 1;
                last_valid = stringarr.is_valid(last_idx);
            }
            for _ in 0..(capacity - count as usize) {
                bitmap_builder.append(last_valid);
            }
            builder = builder.null_bit_buffer(bitmap_builder.finish());
        }
        let arr_data = builder.build();

        let new_arr = Arc::new(LargeStringArray::from(arr_data)) as ArrayRef;

        let s = Series::try_from((self.name(), new_arr)).unwrap();
        Ok((s, offsets))
    }
}
