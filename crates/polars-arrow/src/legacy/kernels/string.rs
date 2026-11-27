use polars_buffer::Buffer;

use crate::array::{Array, ArrayRef, UInt32Array, Utf8ViewArray, View};
use crate::datatypes::ArrowDataType;
use crate::legacy::trusted_len::TrustedLenPush;

pub fn utf8view_len_bytes(array: &Utf8ViewArray) -> ArrayRef {
    let values = array.len_iter().collect::<Vec<_>>();
    let values: Buffer<_> = values.into();
    let array = UInt32Array::new(ArrowDataType::UInt32, values, array.validity().cloned());
    Box::new(array)
}

const CONTINUATION_BITS: u64 = 0x8080_8080_8080_8080;

/// Number of UTF-8 continuation bytes (`0b10xx_xxxx`) in the bytes of `word`.
#[inline(always)]
fn num_continuation_bytes_in_word(word: u64) -> u32 {
    (word & !(word << 1) & CONTINUATION_BITS).count_ones()
}

/// Number of UTF-8 continuation bytes (`0b10xx_xxxx`) in `bytes`.
#[inline]
pub fn num_continuation_bytes(bytes: &[u8]) -> u32 {
    // Sum in u8 lanes per block so that this vectorizes well; a block has fewer than 256 bytes.
    let blocks = bytes.chunks_exact(128);
    let rest = blocks.remainder();
    let mut count = blocks
        .map(|block| {
            block
                .iter()
                .map(|&b| ((b as i8) < -0x40) as u8)
                .fold(0u8, |acc, x| acc.wrapping_add(x)) as u32
        })
        .sum::<u32>();

    let words = rest.chunks_exact(8);
    let tail = words.remainder();
    let num_tail_bytes = tail.len();
    for word in words {
        count += num_continuation_bytes_in_word(u64::from_le_bytes(word.try_into().unwrap()));
    }
    if num_tail_bytes > 0 {
        let last = if bytes.len() >= 8 {
            // Read the last 8 bytes and drop the ones that were already counted.
            u64::from_le_bytes(bytes[bytes.len() - 8..].try_into().unwrap())
                >> (8 * (8 - num_tail_bytes))
        } else {
            let mut buf = [0u8; 8];
            buf[..num_tail_bytes].copy_from_slice(tail);
            u64::from_le_bytes(buf)
        };
        count += num_continuation_bytes_in_word(last);
    }
    count
}

pub fn string_len_chars(array: &Utf8ViewArray) -> ArrayRef {
    const INLINE_CONTINUATION_BITS: u128 = 0x8080_8080_8080_8080_8080_8080_0000_0000;

    let buffers = array.data_buffers();
    let values = array.views().iter().map(|view| {
        if view.is_inline() {
            let data = view.as_u128().to_le();
            // Padding of null views need not be zero, so only keep the first `length` bytes.
            let used = u128::MAX >> (8 * (View::MAX_INLINE_SIZE - view.length));
            let continuation = data & !(data << 1) & INLINE_CONTINUATION_BITS & used;
            view.length - continuation.count_ones()
        } else {
            // SAFETY: views of a valid array point into its buffers.
            let bytes = unsafe { view.get_external_slice_unchecked(buffers) };
            view.length - num_continuation_bytes(bytes)
        }
    });
    let values: Buffer<_> = Vec::from_trusted_len_iter(values).into();
    let array = UInt32Array::new(ArrowDataType::UInt32, values, array.validity().cloned());
    Box::new(array)
}
