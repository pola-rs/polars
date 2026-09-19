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

use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::{Bitmap, BitmapBuilder};
use polars_arrow::datatypes::ArrowDataType;

use crate::row::RowEncodingOptions;

#[inline(always)]
fn sentinel(opt_value: Option<bool>, opt: RowEncodingOptions) -> u8 {
    match opt_value {
        None => opt.null_sentinel(),
        Some(false) => opt.bool_false_sentinel(),
        Some(true) => opt.bool_true_sentinel(),
    }
}

pub(crate) unsafe fn encode_bool<I: Iterator<Item = Option<bool>>>(
    buffer: &mut [MaybeUninit<u8>],
    input: I,
    opt: RowEncodingOptions,
    offsets: &mut [usize],
) {
    for (offset, opt_value) in offsets.iter_mut().zip(input) {
        *buffer.get_unchecked_mut(*offset) = MaybeUninit::new(sentinel(opt_value, opt));
        *offset += 1;
    }
}

/// Encode values `stride` bytes apart starting at `out`.
pub(crate) unsafe fn encode_bool_strided(
    out: *mut MaybeUninit<u8>,
    stride: usize,
    arr: &BooleanArray,
    opt: RowEncodingOptions,
) {
    for (i, opt_value) in arr.iter().enumerate() {
        *out.add(i * stride) = MaybeUninit::new(sentinel(opt_value, opt));
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

/// Collects decoded booleans and their validity. Both are pushed 64 rows at a time as one
/// word.
pub(crate) struct BooleanCollector {
    values: BitmapBuilder,
    validity: BitmapBuilder,
    has_nulls: bool,
}

impl BooleanCollector {
    pub fn with_capacity(num_rows: usize) -> Self {
        Self {
            values: BitmapBuilder::with_capacity(num_rows),
            validity: BitmapBuilder::with_capacity(num_rows),
            has_nulls: false,
        }
    }

    /// Decode `num_rows` values that are `stride` bytes apart starting at `ptr`.
    pub unsafe fn decode_strided(
        &mut self,
        ptr: *const u8,
        stride: usize,
        num_rows: usize,
        opt: RowEncodingOptions,
    ) {
        let null_sentinel = opt.null_sentinel();
        let true_sentinel = opt.bool_true_sentinel();
        self.values.reserve(num_rows);
        self.validity.reserve(num_rows);

        let mut row = 0;
        while row < num_rows {
            let len = (num_rows - row).min(64);
            let mut values = 0u64;
            let mut valid = 0u64;
            for i in 0..len {
                let b = *ptr.add((row + i) * stride);
                values |= ((b == true_sentinel) as u64) << i;
                valid |= ((b != null_sentinel) as u64) << i;
            }
            self.has_nulls |= valid != (u64::MAX >> (64 - len));
            self.values.push_word_with_len_unchecked(values, len);
            self.validity.push_word_with_len_unchecked(valid, len);
            row += len;
        }
    }

    pub fn finish(self) -> BooleanArray {
        let validity = if self.has_nulls {
            self.validity.into_opt_validity()
        } else {
            None
        };
        BooleanArray::new(ArrowDataType::Boolean, self.values.freeze(), validity)
    }
}
