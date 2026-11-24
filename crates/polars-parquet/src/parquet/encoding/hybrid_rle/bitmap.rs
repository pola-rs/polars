use std::io::Write;

use polars_arrow::bitmap::Bitmap;
use polars_arrow::bitmap::bitmask::BitMask;
use polars_arrow::bitmap::utils::{BitChunks, leading_ones, leading_zeros};

use super::Encoder;
use super::encoder::MAX_VALUES_PER_LITERAL_RUN;
use crate::parquet::encoding::uleb128;

const BIT_MASK: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// Sets bit at position `i` in `byte`
#[inline]
pub fn set(byte: u8, i: usize) -> u8 {
    byte | BIT_MASK[i]
}

/// An [`Iterator`] of bool that decodes a bitmap.
/// This is a specialization of [`super::super::bitpacked::Decoder`] for `num_bits == 1`.
#[derive(Debug)]
pub struct BitmapIter<'a> {
    iter: std::slice::Iter<'a, u8>,
    current_byte: &'a u8,
    remaining: usize,
    mask: u8,
}

impl<'a> BitmapIter<'a> {
    /// Returns a new [`BitmapIter`].
    /// # Panics
    /// This function panics iff `offset / 8 > slice.len()`
    #[inline]
    pub fn new(slice: &'a [u8], offset: usize, len: usize) -> Self {
        let bytes = &slice[offset / 8..];

        let mut iter = bytes.iter();

        let current_byte = iter.next().unwrap_or(&0);

        Self {
            iter,
            mask: 1u8.rotate_left(offset as u32),
            remaining: len,
            current_byte,
        }
    }
}

impl Iterator for BitmapIter<'_> {
    type Item = bool;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // easily predictable in branching
        if self.remaining == 0 {
            return None;
        } else {
            self.remaining -= 1;
        }
        let value = self.current_byte & self.mask != 0;
        self.mask = self.mask.rotate_left(1);
        if self.mask == 1 {
            // reached a new byte => try to fetch it from the iterator
            if let Some(v) = self.iter.next() {
                self.current_byte = v
            }
        }
        Some(value)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

/// Writes an iterator of bools into writer, with LSB first.
pub fn encode_bool<W: Write, I: Iterator<Item = bool>>(
    writer: &mut W,
    mut iterator: I,
) -> std::io::Result<()> {
    // the length of the iterator.
    let length = iterator.size_hint().1.unwrap();

    let chunks = length / 8;
    let remainder = length % 8;

    (0..chunks).try_for_each(|_| {
        let mut byte = 0u8;
        (0..8).for_each(|i| {
            if iterator.next().unwrap() {
                byte = set(byte, i)
            }
        });
        writer.write_all(&[byte])
    })?;

    if remainder != 0 {
        let mut last = 0u8;
        iterator.enumerate().for_each(|(i, value)| {
            if value {
                last = set(last, i)
            }
        });
        writer.write_all(&[last])
    } else {
        Ok(())
    }
}

pub(crate) fn append_packed_bits(writer: &mut Vec<u8>, bytes: &[u8], offset: usize, len: usize) {
    writer.reserve(len.div_ceil(8));
    if offset.is_multiple_of(8) {
        let start = offset / 8;
        writer.extend_from_slice(&bytes[start..start + len / 8]);
        if !len.is_multiple_of(8) {
            writer.push(bytes[start + len / 8] & ((1 << (len % 8)) - 1));
        }
    } else {
        let chunks = BitChunks::<u8>::new(bytes, offset, len);
        let remainder = chunks.remainder();
        writer.extend(chunks);
        if !len.is_multiple_of(8) {
            writer.push(remainder);
        }
    }
}

pub(crate) fn encode_bitmap(writer: &mut Vec<u8>, validity: &Bitmap) -> std::io::Result<()> {
    let (bytes, offset, len) = validity.as_slice();
    let mask = BitMask::from_bitmap(validity);
    let mut literal_start = 0;
    let write_literals = |writer: &mut Vec<u8>, start: usize, end: usize| {
        for pos in (start..end).step_by(MAX_VALUES_PER_LITERAL_RUN) {
            let length = (end - pos).min(MAX_VALUES_PER_LITERAL_RUN);
            let mut header = [0; 10];
            let n = uleb128::encode(((length.div_ceil(8) as u64) << 1) | 1, &mut header);
            writer.extend_from_slice(&header[..n]);
            append_packed_bits(writer, bytes, offset + pos, length);
        }
    };

    // A non-final literal must contain a whole number of eight-value groups.
    let mut i = 0;
    while i + 9 <= len {
        let word = mask.get_u32(i) & 0x1ff;
        if word != 0 && word != 0x1ff {
            i += 8;
            continue;
        }
        write_literals(writer, literal_start, i);
        let run = if word == 0 {
            leading_zeros(bytes, offset + i, len - i)
        } else {
            leading_ones(bytes, offset + i, len - i)
        };
        <bool as Encoder<bool>>::run_length_encode(writer, run, word != 0, 1)?;
        i += run;
        literal_start = i;
    }
    write_literals(writer, literal_start, len);
    Ok(())
}
