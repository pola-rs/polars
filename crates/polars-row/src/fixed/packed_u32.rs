//! Row Encoding for Enum's
//!
//! This is a fixed-size encoding that takes a number of maximum bits that each value can take and
//! compresses such that a minimum amount of bytes are used for each value.

use std::mem::MaybeUninit;

use arrow::array::{Array, PrimitiveArray};
use arrow::bitmap::BitmapBuilder;
use arrow::datatypes::ArrowDataType;
use polars_utils::slice::Slice2Uninit;

use crate::row::RowEncodingOptions;

pub fn len_from_num_bits(mut num_bits: usize) -> usize {
    // 1 bit is used to indicate the nullability
    num_bits += 1;
    num_bits.div_ceil(8)
}

macro_rules! with_constant_num_bytes {
    ($num_bytes:ident, $block:block) => {
        with_arms!($num_bytes, $block, (1, 2, 3, 4))
    };
}

pub unsafe fn encode(
    buffer: &mut [MaybeUninit<u8>],
    input: &PrimitiveArray<u32>,
    opt: RowEncodingOptions,
    offsets: &mut [usize],

    num_bits: usize,
) {
    if input.null_count() == 0 {
        unsafe { encode_slice(buffer, input.values(), opt, offsets, num_bits) }
    } else {
        unsafe {
            encode_iter(
                buffer,
                input.iter().map(|v| v.copied()),
                opt,
                offsets,
                num_bits,
            )
        }
    }
}

fn get_invert_mask(opt: RowEncodingOptions, num_bits: usize) -> u32 {
    if !opt.contains(RowEncodingOptions::DESCENDING) {
        return 0;
    }

    (1 << num_bits) - 1
}

pub unsafe fn encode_slice(
    buffer: &mut [MaybeUninit<u8>],
    input: &[u32],
    opt: RowEncodingOptions,
    offsets: &mut [usize],

    num_bits: usize,
) {
    if num_bits == 32 {
        super::numeric::encode_slice(buffer, input, opt, offsets);
        return;
    }

    let num_bytes = len_from_num_bits(num_bits);
    let valid_mask = ((!opt.null_sentinel() & 0x80) as u32) << ((num_bytes - 1) * 8);
    let invert_mask = get_invert_mask(opt, num_bits);

    with_constant_num_bytes!(num_bytes, {
        for (offset, &v) in offsets.iter_mut().zip(input) {
            let v = (v ^ invert_mask) | valid_mask;
            unsafe { buffer.get_unchecked_mut(*offset..*offset + num_bytes) }
                .copy_from_slice(v.to_be_bytes()[4 - num_bytes..].as_uninit());
            *offset += num_bytes;
        }
    });
}

pub unsafe fn encode_iter(
    buffer: &mut [MaybeUninit<u8>],
    input: impl Iterator<Item = Option<u32>>,
    opt: RowEncodingOptions,
    offsets: &mut [usize],

    num_bits: usize,
) {
    if num_bits == 32 {
        super::numeric::encode_iter(buffer, input, opt, offsets);
        return;
    }

    let num_bytes = len_from_num_bits(num_bits);
    let null_value = (opt.null_sentinel() as u32) << ((num_bytes - 1) * 8);
    let valid_mask = ((!opt.null_sentinel() & 0x80) as u32) << ((num_bytes - 1) * 8);
    let invert_mask = get_invert_mask(opt, num_bits);

    with_constant_num_bytes!(num_bytes, {
        for (offset, v) in offsets.iter_mut().zip(input) {
            match v {
                None => {
                    unsafe { buffer.get_unchecked_mut(*offset..*offset + num_bytes) }
                        .copy_from_slice(null_value.to_be_bytes()[4 - num_bytes..].as_uninit());
                },
                Some(v) => {
                    let v = (v ^ invert_mask) | valid_mask;
                    unsafe { buffer.get_unchecked_mut(*offset..*offset + num_bytes) }
                        .copy_from_slice(v.to_be_bytes()[4 - num_bytes..].as_uninit());
                },
            }

            *offset += num_bytes;
        }
    });
}

pub unsafe fn decode(
    rows: &mut [&[u8]],
    opt: RowEncodingOptions,
    num_bits: usize,
) -> PrimitiveArray<u32> {
    if num_bits == 32 {
        return super::numeric::decode_primitive(rows, opt);
    }

    let mut values = Vec::with_capacity(rows.len());
    let null_sentinel = opt.null_sentinel();

    let num_bytes = len_from_num_bits(num_bits);
    let mask = (1 << num_bits) - 1;
    let invert_mask = get_invert_mask(opt, num_bits);

    with_constant_num_bytes!(num_bytes, {
        values.extend(
            rows.iter_mut()
                .take_while(|row| *unsafe { row.get_unchecked(0) } != null_sentinel)
                .map(|row| {
                    let mut value = 0u32;
                    let value_ref: &mut [u8; 4] = bytemuck::cast_mut(&mut value);
                    value_ref[4 - num_bytes..].copy_from_slice(row.get_unchecked(..num_bytes));

                    *row = &row[num_bytes..];
                    ((value.swap_bytes()) & mask) ^ invert_mask
                }),
        );
    });

    if values.len() == rows.len() {
        return PrimitiveArray::new(ArrowDataType::UInt32, values.into(), None);
    }

    let mut validity = BitmapBuilder::with_capacity(rows.len());
    validity.extend_constant(values.len(), true);

    let start_len = values.len();

    with_constant_num_bytes!(num_bytes, {
        values.extend(rows[start_len..].iter_mut().map(|row| {
            validity.push(*unsafe { row.get_unchecked(0) } != null_sentinel);

            let mut value = 0u32;
            let value_ref: &mut [u8; 4] = bytemuck::cast_mut(&mut value);
            value_ref[4 - num_bytes..].copy_from_slice(row.get_unchecked(..num_bytes));

            *row = &row[num_bytes..];
            ((value.swap_bytes()) & mask) ^ invert_mask
        }));
    });

    PrimitiveArray::new(
        ArrowDataType::UInt32,
        values.into(),
        validity.into_opt_validity(),
    )
}
