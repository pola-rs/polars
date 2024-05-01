use std::io::Write;

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

impl<'a> Iterator for BitmapIter<'a> {
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
    let reminder = length % 8;

    (0..chunks).try_for_each(|_| {
        let mut byte = 0u8;
        (0..8).for_each(|i| {
            if iterator.next().unwrap() {
                byte = set(byte, i)
            }
        });
        writer.write_all(&[byte])
    })?;

    if reminder != 0 {
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
