use super::utils::count_set_bits_offset;
use crate::chunked_array::builder::set_null_bits;
use crate::chunked_array::kernels::utils::{get_bitmasks, BitMaskU64Prep};
use crate::datatypes::PolarsNumericType;
use arrow::array::{Array, ArrayData, BooleanArray, PrimitiveArray};
use arrow::buffer::MutableBuffer;
use arrow::datatypes::ToByteSlice;
use std::mem;

/// Is very fast when large parts of the mask are false, or true. The mask should have no offset.
/// Not that the nulls of the mask are ignored.
pub fn set_with_value<T>(
    mask: &BooleanArray,
    left: &PrimitiveArray<T>,
    value: T::Native,
) -> PrimitiveArray<T>
where
    T: PolarsNumericType,
{
    let value_size = mem::size_of::<T::Native>();
    // Create a slice of 64 elements of T::Native.
    // this slice will be copied in when a whole batch is valid
    // this slice will be created as bytes
    let mut value_slice = Vec::with_capacity(value_size * 64);
    for _ in 0..64 {
        value_slice.extend(value.to_byte_slice().iter());
    }

    // create a byte slice representing a single value T::Native
    let single_value_slice = &value_slice[0..value_size];
    let bitmasks = get_bitmasks();
    let mut bitmask_u64_helper = BitMaskU64Prep::new(mask);
    let mask_u64 = bitmask_u64_helper.get_mask_as_u64();

    let left_bytes = left.data_ref().buffers()[0].as_slice();
    let left_offset = left.offset();

    let mut target_buffer = MutableBuffer::new(left.len() * value_size);
    target_buffer.resize(left.len() * value_size, 0);
    let target_bytes = target_buffer.as_slice_mut();

    let all_ones = u64::MAX;
    let bytes_length_64_values = 64 * value_size;
    let mut target_bytes_idx = 0;

    for (i, &mask_batch) in mask_u64.iter().enumerate() {
        if mask_batch == 0 {
            //  take 64 values from left

            // offset in the array
            let data_idx = (i * 64) + left_offset;

            // left could have more zeros due to padding
            if data_idx + 64 < left.len() {
                // offset of the bytes in the array
                let data_bytes_idx = data_idx * value_size;
                // write the bytes
                target_bytes[target_bytes_idx..(target_bytes_idx + bytes_length_64_values)]
                    .copy_from_slice(
                        &left_bytes[data_bytes_idx..(data_bytes_idx + bytes_length_64_values)],
                    );
                target_bytes_idx += bytes_length_64_values;
                continue;
            }
        } else if mask_batch == all_ones {
            //  fill 64 values at once

            // write the bytes
            target_bytes[target_bytes_idx..(target_bytes_idx + bytes_length_64_values)]
                .copy_from_slice(&value_slice);
            target_bytes_idx += bytes_length_64_values;
            continue;
        }

        for (j, &bitmask) in bitmasks.iter().enumerate() {
            // for each bit in batch.
            // the and operation nullifies the other unset bits
            if (mask_batch & bitmask) == 0 {
                // left path
                let mut data_idx = (i * 64) + j;

                // there could be more zero bits due to padding
                if data_idx == left.len() {
                    break;
                }
                data_idx += left_offset;

                let data_bytes_idx = data_idx * value_size;
                target_bytes[target_bytes_idx..(target_bytes_idx + value_size)]
                    .copy_from_slice(&left_bytes[data_bytes_idx..(data_bytes_idx + value_size)]);
            } else {
                target_bytes[target_bytes_idx..(target_bytes_idx + value_size)]
                    .copy_from_slice(&single_value_slice);
            }
            target_bytes_idx += value_size;
        }
    }
    let builder = ArrayData::builder(T::DATA_TYPE)
        .len(left.len())
        .add_buffer(target_buffer.into());
    let null_bit_buffer = left
        .data()
        .null_bitmap()
        .as_ref()
        .map(|bm| bm.clone().into_buffer());
    let null_count = null_bit_buffer
        .as_ref()
        .map(|buf| left.len() - count_set_bits_offset(buf.as_slice(), 0, left.len()));
    let builder = set_null_bits(builder, null_bit_buffer, null_count);
    let data = builder.build();
    PrimitiveArray::<T>::from(data)
}

#[cfg(test)]
mod test {
    use super::*;
    use arrow::array::UInt32Array;

    #[test]
    fn test_set_with() {
        let mask = BooleanArray::from((0..86).map(|v| v > 68 && v != 85).collect::<Vec<bool>>());
        let val = UInt32Array::from((0..86).collect::<Vec<_>>());
        let a = set_with_value(&mask, &val, 100);
        let slice = a.values();
        assert_eq!(slice[a.len() - 1], 85);
        assert_eq!(slice[a.len() - 2], 100);
        assert_eq!(slice[67], 67);
        assert_eq!(slice[68], 68);
        assert_eq!(slice[1], 1);
        assert_eq!(slice[0], 0);
    }
}
