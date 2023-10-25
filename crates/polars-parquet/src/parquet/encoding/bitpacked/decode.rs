use crate::error::Error;

use super::{Packed, Unpackable, Unpacked};

/// An [`Iterator`] of [`Unpackable`] unpacked from a bitpacked slice of bytes.
/// # Implementation
/// This iterator unpacks bytes in chunks and does not allocate.
#[derive(Debug, Clone)]
pub struct Decoder<'a, T: Unpackable> {
    packed: std::slice::Chunks<'a, u8>,
    num_bits: usize,
    remaining: usize,          // in number of items
    current_pack_index: usize, // invariant: < T::PACK_LENGTH
    unpacked: T::Unpacked,     // has the current unpacked values.
}

#[inline]
fn decode_pack<T: Unpackable>(packed: &[u8], num_bits: usize, unpacked: &mut T::Unpacked) {
    if packed.len() < T::Unpacked::LENGTH * num_bits / 8 {
        let mut buf = T::Packed::zero();
        buf.as_mut()[..packed.len()].copy_from_slice(packed);
        T::unpack(buf.as_ref(), num_bits, unpacked)
    } else {
        T::unpack(packed, num_bits, unpacked)
    }
}

impl<'a, T: Unpackable> Decoder<'a, T> {
    /// Returns a [`Decoder`] with `T` encoded in `packed` with `num_bits`.
    pub fn try_new(packed: &'a [u8], num_bits: usize, mut length: usize) -> Result<Self, Error> {
        let block_size = std::mem::size_of::<T>() * num_bits;

        if num_bits == 0 {
            return Err(Error::oos("Bitpacking requires num_bits > 0"));
        }

        if packed.len() * 8 < length * num_bits {
            return Err(Error::oos(format!(
                "Unpacking {length} items with a number of bits {num_bits} requires at least {} bytes.",
                length * num_bits / 8
            )));
        }

        let mut packed = packed.chunks(block_size);
        let mut unpacked = T::Unpacked::zero();
        if let Some(chunk) = packed.next() {
            decode_pack::<T>(chunk, num_bits, &mut unpacked);
        } else {
            length = 0
        };

        Ok(Self {
            remaining: length,
            packed,
            num_bits,
            unpacked,
            current_pack_index: 0,
        })
    }
}

impl<'a, T: Unpackable> Iterator for Decoder<'a, T> {
    type Item = T;

    #[inline] // -71% improvement in bench
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let result = self.unpacked[self.current_pack_index];
        self.current_pack_index += 1;
        self.remaining -= 1;
        if self.current_pack_index == T::Unpacked::LENGTH {
            if let Some(packed) = self.packed.next() {
                decode_pack::<T>(packed, self.num_bits, &mut self.unpacked);
                self.current_pack_index = 0;
            }
        }
        Some(result)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::case1;
    use super::*;

    #[test]
    fn test_decode_rle() {
        // Test data: 0-7 with bit width 3
        // 0: 000
        // 1: 001
        // 2: 010
        // 3: 011
        // 4: 100
        // 5: 101
        // 6: 110
        // 7: 111
        let num_bits = 3;
        let length = 8;
        // encoded: 0b10001000u8, 0b11000110, 0b11111010
        let data = vec![0b10001000u8, 0b11000110, 0b11111010];

        let decoded = Decoder::<u32>::try_new(&data, num_bits, length)
            .unwrap()
            .collect::<Vec<_>>();
        assert_eq!(decoded, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn decode_large() {
        let (num_bits, expected, data) = case1();

        let decoded = Decoder::<u32>::try_new(&data, num_bits, expected.len())
            .unwrap()
            .collect::<Vec<_>>();
        assert_eq!(decoded, expected);
    }

    #[test]
    fn test_decode_bool() {
        let num_bits = 1;
        let length = 8;
        let data = vec![0b10101010];

        let decoded = Decoder::<u32>::try_new(&data, num_bits, length)
            .unwrap()
            .collect::<Vec<_>>();
        assert_eq!(decoded, vec![0, 1, 0, 1, 0, 1, 0, 1]);
    }

    #[test]
    fn test_decode_u64() {
        let num_bits = 1;
        let length = 8;
        let data = vec![0b10101010];

        let decoded = Decoder::<u64>::try_new(&data, num_bits, length)
            .unwrap()
            .collect::<Vec<_>>();
        assert_eq!(decoded, vec![0, 1, 0, 1, 0, 1, 0, 1]);
    }

    #[test]
    fn even_case() {
        // [0, 1, 2, 3, 4, 5, 6, 0]x99
        let data = &[0b10001000u8, 0b11000110, 0b00011010];
        let num_bits = 3;
        let copies = 99; // 8 * 99 % 32 != 0
        let expected = std::iter::repeat(&[0u32, 1, 2, 3, 4, 5, 6, 0])
            .take(copies)
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let data = std::iter::repeat(data)
            .take(copies)
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let length = expected.len();

        let decoded = Decoder::<u32>::try_new(&data, num_bits, length)
            .unwrap()
            .collect::<Vec<_>>();
        assert_eq!(decoded, expected);
    }

    #[test]
    fn odd_case() {
        // [0, 1, 2, 3, 4, 5, 6, 0]x4 + [2]
        let data = &[0b10001000u8, 0b11000110, 0b00011010];
        let num_bits = 3;
        let copies = 4;
        let expected = std::iter::repeat(&[0u32, 1, 2, 3, 4, 5, 6, 0])
            .take(copies)
            .flatten()
            .copied()
            .chain(std::iter::once(2))
            .collect::<Vec<_>>();
        let data = std::iter::repeat(data)
            .take(copies)
            .flatten()
            .copied()
            .chain(std::iter::once(0b00000010u8))
            .collect::<Vec<_>>();
        let length = expected.len();

        let decoded = Decoder::<u32>::try_new(&data, num_bits, length)
            .unwrap()
            .collect::<Vec<_>>();
        assert_eq!(decoded, expected);
    }

    #[test]
    fn test_errors() {
        // zero length
        assert!(Decoder::<u64>::try_new(&[], 1, 0).is_ok());
        // no bytes
        assert!(Decoder::<u64>::try_new(&[], 1, 1).is_err());
        // too few bytes
        assert!(Decoder::<u64>::try_new(&[1], 1, 8).is_ok());
        assert!(Decoder::<u64>::try_new(&[1, 1], 2, 8).is_ok());
        assert!(Decoder::<u64>::try_new(&[1], 1, 9).is_err());
        // zero num_bits
        assert!(Decoder::<u64>::try_new(&[1], 0, 1).is_err());
    }
}
