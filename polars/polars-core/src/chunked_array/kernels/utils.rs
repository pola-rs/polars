use crate::prelude::*;
use arrow::array::{Array, BooleanArray};
use arrow::bitmap::Bitmap;
use arrow::buffer::{Buffer, MutableBuffer};
use arrow::error::Result as ArrowResult;
use std::ops::BitOr;

use arrow::util::bit_util::{get_bit, round_upto_power_of_2};

static POPCOUNT_TABLE: [u8; 256] = [
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,
];

/// Returns the number of 1-bits in `data`, starting from `offset` with `length` bits
/// inspected. Note that both `offset` and `length` are measured in bits.
#[inline]
pub fn count_set_bits_offset(data: &[u8], offset: usize, length: usize) -> usize {
    let bit_end = offset + length;
    assert!(bit_end <= (data.len() << 3));

    let byte_start = std::cmp::min(round_upto_power_of_2(offset, 8), bit_end);
    let num_bytes = (bit_end - byte_start) >> 3;

    let mut result = 0;

    for i in offset..byte_start {
        if get_bit(data, i) {
            result += 1;
        }
    }
    for i in 0..num_bytes {
        // todo! does bounds checking
        result += POPCOUNT_TABLE[data[(byte_start >> 3) + i] as usize] as usize;
    }
    for i in (byte_start + (num_bytes << 3))..bit_end {
        if get_bit(data, i) {
            result += 1;
        }
    }

    result
}

pub(crate) fn apply_bin_op_to_option_bitmap<F>(
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

/// Combine two null bitmaps.
pub(crate) fn combine_bitmaps_or(left: &impl Array, right: &impl Array) -> Option<Buffer> {
    // get data buffers
    let data_array = right.data();
    let data_mask = left.data();

    // get null bitmasks
    let mask_bitmap = data_mask.null_bitmap();
    let array_bitmap = data_array.null_bitmap();

    // Compute final null values by bitor ops
    let bitmap =
        apply_bin_op_to_option_bitmap(mask_bitmap, array_bitmap, |a, b| a.bitor(b)).unwrap();
    bitmap.map(|bitmap| bitmap.into_buffer())
}

#[inline]
/// Bit masks to use with bitand operation to check if a bit is set.
/// 2^0, 2^1, ... 2^64.
/// in binary this is
///  1
///  10
///  100
/// etc.
pub(crate) fn get_bitmasks() -> [u64; 64] {
    let mut bitmasks = [0; 64];
    for (i, item) in bitmasks.iter_mut().enumerate() {
        *item = 1u64 << i
    }
    bitmasks
}

pub(crate) struct BitMaskU64Prep {
    buf: MutableBuffer,
}

impl BitMaskU64Prep {
    pub(crate) fn new(mask: &BooleanArray) -> Self {
        let mask_bytes = mask.data_ref().buffers()[0].as_slice();
        let mut u64_buffer = MutableBuffer::new(mask_bytes.len());

        // add to the resulting len so is a multiple of u64
        // note: the last % 8 is such that when the first part results to 8 -> output = 0
        let pad_additional_len = (8 - mask_bytes.len() % 8) % 8;
        u64_buffer.extend_from_slice(mask_bytes);
        u64_buffer.extend((0..pad_additional_len).map(|_| 0u8));
        // u64_buffer
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
        Self { buf: u64_buffer }
    }

    pub(crate) fn get_mask_as_u64(&mut self) -> &mut [u64] {
        self.buf.typed_data_mut::<u64>()
    }
}
