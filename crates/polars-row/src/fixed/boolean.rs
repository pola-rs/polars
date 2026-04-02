#![allow(unsafe_op_in_unsafe_fn)]
//! Row encoding for Booleans
//!
//! Each Boolean value is encoded by one byte:
//!
//! | Value | Encoding        |
//! |-------|-----------------|
//! | None  | `0x00` / `0xFF` |
//! | False | `0x02` / `0xFD` |
//! | True  | `0x03` / `0xFC` |

use std::mem::MaybeUninit;

use arrow::array::BooleanArray;
use arrow::bitmap::Bitmap;
use arrow::datatypes::ArrowDataType;

use crate::row::RowEncodingOptions;

pub(crate) unsafe fn encode_bool<I: Iterator<Item = Option<bool>>>(
    buffer: &mut [MaybeUninit<u8>],
    input: I,
    opt: RowEncodingOptions,
    offsets: &mut [usize],
) {
    let null_sentinel = opt.null_sentinel();
    let true_sentinel = opt.bool_true_sentinel();
    let false_sentinel = opt.bool_false_sentinel();

    for (offset, opt_value) in offsets.iter_mut().zip(input) {
        let b = match opt_value {
            None => null_sentinel,
            Some(false) => false_sentinel,
            Some(true) => true_sentinel,
        };

        *buffer.get_unchecked_mut(*offset) = MaybeUninit::new(b);
        *offset += 1;
    }
}

pub(crate) unsafe fn decode_bool(rows: &mut [&[u8]], opt: RowEncodingOptions) -> BooleanArray {
    let mut has_nulls = false;
    let null_sentinel = opt.null_sentinel();
    let true_sentinel = opt.bool_true_sentinel();

    let values = Bitmap::from_trusted_len_iter_unchecked(rows.iter().map(|row| {
        let b = *row.get_unchecked(0);
        has_nulls |= b == null_sentinel;
        b == true_sentinel
    }));

    if !has_nulls {
        rows.iter_mut()
            .for_each(|row| *row = row.get_unchecked(1..));
        return BooleanArray::new(ArrowDataType::Boolean, values, None);
    }

    let validity = Bitmap::from_trusted_len_iter_unchecked(rows.iter_mut().map(|row| {
        let v = *row.get_unchecked(0) != null_sentinel;
        *row = row.get_unchecked(1..);
        v
    }));
    BooleanArray::new(ArrowDataType::Boolean, values, Some(validity))
}
