use crate::prelude::*;
use arrow::array::{Array, BooleanArray};
use arrow::bitmap::Bitmap;
use arrow::buffer::MutableBuffer;
use arrow::error::Result as ArrowResult;

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
    for i in 0..64 {
        bitmasks[i] = 1u64 << i
    }
    bitmasks
}

pub(crate) struct BitMaskU64Prep {
    buf: MutableBuffer,
}

impl BitMaskU64Prep {
    pub(crate) fn new(mask: &BooleanArray) -> Self {
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
        Self { buf: u64_buffer }
    }

    pub(crate) fn get_mask_as_u64(&mut self) -> &mut [u64] {
        self.buf.typed_data_mut::<u64>()
    }
}
