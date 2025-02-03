//! Row Encoding for Enum's and Categorical's
//!
//! This is a fixed-size encoding that takes a number of maximum bits that each value can take and
//! compresses such that a minimum amount of bytes are used for each value.

use std::mem::MaybeUninit;

use arrow::array::{Array, PrimitiveArray};
use arrow::bitmap::BitmapBuilder;
use arrow::datatypes::ArrowDataType;
use polars_utils::slice::Slice2Uninit;

use crate::row::RowEncodingOptions;

macro_rules! with_constant_num_bytes {
    ($num_bytes:ident, $block:block) => {
        with_arms!(
            $num_bytes,
            $block,
            (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        )
    };
}

pub fn len_from_precision(precision: usize) -> usize {
    len_from_num_bits(num_bits_from_precision(precision))
}

fn num_bits_from_precision(precision: usize) -> usize {
    assert!(precision <= 38);
    // This may seem random. But this is ceil(s * log2(10)) which is a reduction of
    // ceil(log2(10**s))
    ((precision as f32) * 10.0f32.log2()).ceil() as usize
}

fn len_from_num_bits(num_bits: usize) -> usize {
    // 1 bit is used to indicate the nullability
    // 1 bit is used to indicate the signedness
    (num_bits + 2).div_ceil(8)
}

pub unsafe fn encode(
    buffer: &mut [MaybeUninit<u8>],
    input: &PrimitiveArray<i128>,
    opt: RowEncodingOptions,
    offsets: &mut [usize],
    precision: usize,
) {
    if input.null_count() == 0 {
        unsafe { encode_slice(buffer, input.values(), opt, offsets, precision) }
    } else {
        unsafe {
            encode_iter(
                buffer,
                input.iter().map(|v| v.copied()),
                opt,
                offsets,
                precision,
            )
        }
    }
}

pub unsafe fn encode_slice(
    buffer: &mut [MaybeUninit<u8>],
    input: &[i128],
    opt: RowEncodingOptions,
    offsets: &mut [usize],
    precision: usize,
) {
    let num_bits = num_bits_from_precision(precision);

    // If the output will not fit in less bytes, just use the normal i128 encoding kernel.
    if num_bits >= 127 {
        super::numeric::encode_slice(buffer, input, opt, offsets);
        return;
    }

    let num_bytes = len_from_num_bits(num_bits);
    let mask = (1 << (num_bits + 1)) - 1;
    let valid_mask = ((!opt.null_sentinel() & 0x80) as i128) << ((num_bytes - 1) * 8);
    let sign_mask = 1 << num_bits;
    let invert_mask = if opt.contains(RowEncodingOptions::DESCENDING) {
        mask
    } else {
        0
    };

    with_constant_num_bytes!(num_bytes, {
        for (offset, &v) in offsets.iter_mut().zip(input) {
            let mut v = v;

            v &= mask; // Mask out higher sign extension bits
            v ^= sign_mask; // Flip sign-bit to maintain order
            v ^= invert_mask; // Invert for descending
            v |= valid_mask; // Add valid indicator

            unsafe { buffer.get_unchecked_mut(*offset..*offset + num_bytes) }
                .copy_from_slice(v.to_be_bytes()[16 - num_bytes..].as_uninit());
            *offset += num_bytes;
        }
    });
}

pub unsafe fn encode_iter(
    buffer: &mut [MaybeUninit<u8>],
    input: impl Iterator<Item = Option<i128>>,
    opt: RowEncodingOptions,
    offsets: &mut [usize],
    precision: usize,
) {
    let num_bits = num_bits_from_precision(precision);
    // If the output will not fit in less bytes, just use the normal i128 encoding kernel.
    if num_bits >= 127 {
        super::numeric::encode_iter(buffer, input, opt, offsets);
        return;
    }

    let num_bytes = len_from_num_bits(num_bits);
    let null_value = (opt.null_sentinel() as i128) << ((num_bytes - 1) * 8);
    let mask = (1 << (num_bits + 1)) - 1;
    let valid_mask = ((!opt.null_sentinel() & 0x80) as i128) << ((num_bytes - 1) * 8);
    let sign_mask = 1 << num_bits;
    let invert_mask = if opt.contains(RowEncodingOptions::DESCENDING) {
        mask
    } else {
        0
    };

    with_constant_num_bytes!(num_bytes, {
        for (offset, v) in offsets.iter_mut().zip(input) {
            match v {
                None => {
                    unsafe { buffer.get_unchecked_mut(*offset..*offset + num_bytes) }
                        .copy_from_slice(null_value.to_be_bytes()[16 - num_bytes..].as_uninit());
                },
                Some(mut v) => {
                    v &= mask; // Mask out higher sign extension bits
                    v ^= sign_mask; // Flip sign-bit to maintain order
                    v ^= invert_mask; // Invert for descending
                    v |= valid_mask; // Add valid indicator

                    unsafe { buffer.get_unchecked_mut(*offset..*offset + num_bytes) }
                        .copy_from_slice(v.to_be_bytes()[16 - num_bytes..].as_uninit());
                },
            }

            *offset += num_bytes;
        }
    });
}

pub unsafe fn decode(
    rows: &mut [&[u8]],
    opt: RowEncodingOptions,
    precision: usize,
) -> PrimitiveArray<i128> {
    let num_bits = num_bits_from_precision(precision);
    // If the output will not fit in less bytes, just use the normal i128 decoding kernel.
    if num_bits >= 127 {
        let (_, values, validity) = super::numeric::decode_primitive(rows, opt).into_inner();
        return PrimitiveArray::new(ArrowDataType::Int128, values, validity);
    }

    let mut values = Vec::with_capacity(rows.len());
    let null_sentinel = opt.null_sentinel();

    let num_bytes = len_from_num_bits(num_bits);
    let mask = (1 << (num_bits + 1)) - 1;
    let sign_mask = 1 << num_bits;
    let invert_mask = if opt.contains(RowEncodingOptions::DESCENDING) {
        mask
    } else {
        0
    };

    with_constant_num_bytes!(num_bytes, {
        values.extend(
            rows.iter_mut()
                .take_while(|row| *unsafe { row.get_unchecked(0) } != null_sentinel)
                .map(|row| {
                    let mut value = 0i128;
                    let value_ref: &mut [u8; 16] = bytemuck::cast_mut(&mut value);
                    value_ref[16 - num_bytes..].copy_from_slice(row.get_unchecked(..num_bytes));
                    *row = &row[num_bytes..];

                    if cfg!(target_endian = "little") {
                        // Big-Endian -> Little-Endian
                        value = value.swap_bytes();
                    }

                    value ^= invert_mask; // Invert for descending
                    value ^= sign_mask; // Flip sign bit to maintain order

                    // Sign extend. This also masks out the valid bit.
                    value <<= i128::BITS - num_bits as u32 - 1;
                    value >>= i128::BITS - num_bits as u32 - 1;

                    value
                }),
        );
    });

    if values.len() == rows.len() {
        return PrimitiveArray::new(ArrowDataType::Int128, values.into(), None);
    }

    let mut validity = BitmapBuilder::with_capacity(rows.len());
    validity.extend_constant(values.len(), true);

    let start_len = values.len();

    with_constant_num_bytes!(num_bytes, {
        values.extend(rows[start_len..].iter_mut().map(|row| {
            validity.push(*unsafe { row.get_unchecked(0) } != null_sentinel);

            let mut value = 0i128;
            let value_ref: &mut [u8; 16] = bytemuck::cast_mut(&mut value);
            value_ref[16 - num_bytes..].copy_from_slice(row.get_unchecked(..num_bytes));
            *row = &row[num_bytes..];

            if cfg!(target_endian = "little") {
                // Big-Endian -> Little-Endian
                value = value.swap_bytes();
            }

            value ^= invert_mask; // Invert for descending
            value ^= sign_mask; // Flip sign bit to maintain order

            // Sign extend. This also masks out the valid bit.
            value <<= i128::BITS - num_bits as u32 - 1;
            value >>= i128::BITS - num_bits as u32 - 1;

            value
        }));
    });

    PrimitiveArray::new(
        ArrowDataType::Int128,
        values.into(),
        validity.into_opt_validity(),
    )
}
