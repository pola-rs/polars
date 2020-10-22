use crate::chunked_array::builder::{aligned_vec_to_primitive_array, set_null_bits};
use crate::prelude::*;
use arrow::array::{Array, ArrayData, BooleanArray, PrimitiveArray, PrimitiveArrayOps};
use arrow::bitmap::Bitmap;
use arrow::buffer::{Buffer, MutableBuffer};
use arrow::datatypes::ArrowNumericType;
use arrow::error::Result as ArrowResult;
use arrow::util::bit_util::count_set_bits_offset;
use std::ops::BitOr;

pub(super) fn apply_bin_op_to_option_bitmap<F>(
    left: &Option<Bitmap>,
    right: &Option<Bitmap>,
    op: F,
) -> Result<Option<Bitmap>>
where
    F: Fn(&Bitmap, &Bitmap) -> ArrowResult<Bitmap>,
{
    match *left {
        None => match *right {
            None => Ok(None),
            Some(ref r) => Ok(Some(r.clone())),
        },
        Some(ref l) => match *right {
            None => Ok(Some(l.clone())),
            Some(ref r) => Ok(Some(op(&l, &r)?)),
        },
    }
}

fn get_new_null_bit_buffer(mask: &impl Array, a: &impl Array, b: &impl Array) -> Option<Buffer> {
    // get data buffers
    let data_a = a.data();
    let data_b = b.data();
    let data_mask = mask.data();

    // get null bitmasks
    let mask_bitmap = data_mask.null_bitmap();
    let a_bitmap = data_a.null_bitmap();
    let b_bitmap = data_b.null_bitmap();

    // Compute final null values by bitor ops
    let bitmap = apply_bin_op_to_option_bitmap(mask_bitmap, a_bitmap, |a, b| a.bitor(b)).unwrap();
    let bitmap = apply_bin_op_to_option_bitmap(&bitmap, b_bitmap, |a, b| a.bitor(b)).unwrap();
    let null_bit_buffer = bitmap.map(|bitmap| bitmap.into_buffer());
    null_bit_buffer
}

/// Is very fast when large parts of the mask are false, or true. The mask should have no offset.
fn zip_impl<T>(
    mask: &BooleanArray,
    left: &PrimitiveArray<T>,
    right: &PrimitiveArray<T>,
) -> PrimitiveArray<T>
where
    T: ArrowNumericType,
{
    let value_size = std::mem::size_of::<T::Native>();
    // 2^0, 2^1, ... 2^64.
    // in binary this is
    //  1
    //  10
    //  100
    // etc.
    let mut bitmasks = [0; 64];
    for i in 0..64 {
        bitmasks[i] = 1u64 << i
    }
    let mask_bytes = mask.data_ref().buffers()[0].data();

    let mut u64_buffer = MutableBuffer::new(mask_bytes.len());
    // add to the resulting len so is a multiple of u64
    // note: the last % 8 is such that when the first part results to 8 -> output = 0
    let pad_additional_len = (8 - mask_bytes.len() % 8) % 8;
    u64_buffer
        .write_bytes(mask_bytes, pad_additional_len)
        .unwrap();
    let mask_u64 = u64_buffer.typed_data_mut::<u64>();

    // mask any bits outside of the given len
    if mask.len() % 64 != 0 {
        let last_idx = mask_u64.len() - 1;
        // max is all bits set
        // right shift the number of bits that should be unset
        let bitmask = u64::MAX >> (64 - mask.len() % 64);
        // binary and will nullify all that isn't in the mask.
        mask_u64[last_idx] &= bitmask
    }

    let left_bytes = left.data_ref().buffers()[0].data();
    let left_offset = left.offset();
    let right_bytes = right.data_ref().buffers()[0].data();
    let right_offset = right.offset();

    let mut target_buffer = MutableBuffer::new(left.len() * value_size);
    target_buffer.resize(left.len() * value_size).unwrap();
    let target_bytes = target_buffer.data_mut();

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
            //  take 64 values from right

            // offset in the array
            let data_idx = (i * 64) + right_offset;
            // offset of the bytes in the array
            let data_bytes_idx = data_idx * value_size;
            // write the bytes
            target_bytes[target_bytes_idx..(target_bytes_idx + bytes_length_64_values)]
                .copy_from_slice(
                    &right_bytes[data_bytes_idx..(data_bytes_idx + bytes_length_64_values)],
                );
            target_bytes_idx += bytes_length_64_values;
            continue;
        }

        for (j, &bitmask) in bitmasks.iter().enumerate() {
            // for each bit in batch.
            // the and operation nullifies the other unset bits
            if (mask_batch & bitmask) != 0 {
                // left path
                let data_idx = (i * 64) + j + left_offset;
                let data_bytes_idx = data_idx * value_size;
                target_bytes[target_bytes_idx..(target_bytes_idx + value_size)]
                    .copy_from_slice(&left_bytes[data_bytes_idx..(data_bytes_idx + value_size)]);
            } else {
                // right path
                let data_idx = (i * 64) + j + right_offset;

                // there could be more zero bits due to padding
                if data_idx == left.len() {
                    break;
                }
                let data_bytes_idx = data_idx * value_size;
                target_bytes[target_bytes_idx..(target_bytes_idx + value_size)]
                    .copy_from_slice(&right_bytes[data_bytes_idx..(data_bytes_idx + value_size)]);
            }
            target_bytes_idx += value_size;
        }
    }
    let builder = ArrayData::builder(T::get_data_type())
        .len(left.len())
        .add_buffer(target_buffer.freeze());
    let null_bit_buffer = get_new_null_bit_buffer(mask, left, right);
    let null_count = null_bit_buffer
        .as_ref()
        .map(|buf| count_set_bits_offset(buf.data(), 0, left.len()));

    let builder = set_null_bits(builder, null_bit_buffer, null_count, left.len());
    let data = builder.build();
    PrimitiveArray::<T>::from(data)
}

pub fn zip<T>(
    mask: &BooleanArray,
    a: &PrimitiveArray<T>,
    b: &PrimitiveArray<T>,
) -> Result<PrimitiveArray<T>>
where
    T: ArrowNumericType,
{
    if mask.offset() == 0 {
        return Ok(zip_impl(mask, a, b));
    }

    let null_bit_buffer = get_new_null_bit_buffer(mask, a, b);

    // Create an aligned vector.
    let mut values = AlignedVec::with_capacity_aligned(mask.len());

    // Get a slice to the values in the arrow arrays with the right offset
    let vals_a = a.value_slice(a.offset(), a.len());
    let vals_b = a.value_slice(b.offset(), b.len());

    // fill the aligned vector
    for i in 0..mask.len() {
        let take_a = mask.value(i);
        if take_a {
            unsafe {
                values.push(*vals_a.get_unchecked(i));
            }
        } else {
            unsafe {
                values.push(*vals_b.get_unchecked(i));
            }
        }
    }

    Ok(aligned_vec_to_primitive_array::<T>(
        values,
        null_bit_buffer,
        None,
    ))
}

#[cfg(test)]
mod test {
    use super::*;
    use arrow::array::UInt8Array;

    #[test]
    fn test_zip_with() {
        let mask = BooleanArray::from(vec![true, true, false, true]);
        let val = UInt8Array::from(vec![1, 1, 1, 1]);
        let val_2 = UInt8Array::from(vec![4, 4, 4, 4]);
        let a = zip_impl(&mask, &val, &val_2);
        assert_eq!(a.value_slice(0, a.len()), &[1, 1, 4, 1]);
    }
}
