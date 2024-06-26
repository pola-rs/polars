use super::{Packed, Unpackable, Unpacked};
use crate::parquet::error::ParquetError;

/// An [`Iterator`] of [`Unpackable`] unpacked from a bitpacked slice of bytes.
/// # Implementation
/// This iterator unpacks bytes in chunks and does not allocate.
#[derive(Debug, Clone)]
pub struct Decoder<'a, T: Unpackable> {
    packed: std::slice::Chunks<'a, u8>,
    num_bits: usize,
    /// number of items
    length: usize,
    _pd: std::marker::PhantomData<T>,
}

#[derive(Debug)]
pub struct DecoderIter<T: Unpackable> {
    buffer: Vec<T>,
    idx: usize,
}

impl<T: Unpackable> Iterator for DecoderIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.buffer.len() {
            return None;
        }

        let value = self.buffer[self.idx];
        self.idx += 1;

        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.buffer.len() - self.idx;

        (len, Some(len))
    }
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
    pub fn new(packed: &'a [u8], num_bits: usize, length: usize) -> Self {
        Self::try_new(packed, num_bits, length).unwrap()
    }

    pub fn collect_into_iter(self) -> DecoderIter<T> {
        let mut buffer = Vec::new();
        self.collect_into(&mut buffer);
        DecoderIter { buffer, idx: 0 }
    }

    /// Returns a [`Decoder`] with `T` encoded in `packed` with `num_bits`.
    pub fn try_new(packed: &'a [u8], num_bits: usize, length: usize) -> Result<Self, ParquetError> {
        let block_size = std::mem::size_of::<T>() * num_bits;

        if num_bits == 0 {
            return Err(ParquetError::oos("Bitpacking requires num_bits > 0"));
        }

        if packed.len() * 8 < length * num_bits {
            return Err(ParquetError::oos(format!(
                "Unpacking {length} items with a number of bits {num_bits} requires at least {} bytes.",
                length * num_bits / 8
            )));
        }

        let packed = packed.chunks(block_size);

        Ok(Self {
            length,
            packed,
            num_bits,
            _pd: Default::default(),
        })
    }
}

impl<'a, T: Unpackable> Decoder<'a, T> {
    pub fn collect_into(mut self, vec: &mut Vec<T>) {
        // @NOTE:
        // When microbenchmarking changing this from a element-wise iterator to a collect into
        // improves the speed by around 4x.
        //
        // The unsafe code here allows us to not have to do a double memcopy. This saves us 20% in
        // our microbenchmark.
        //
        // GB: I did some profiling on this function using the Yellow NYC Taxi dataset. There, the
        // average self.length is ~52.8 and the average num_packs is ~2.2. Let this guide your
        // decisions surrounding the optimization of this function.

        // @NOTE:
        // Since T::Unpacked::LENGTH is always a power of two and known at compile time. Division,
        // modulo and multiplication are just trivial operators.
        let num_packs = (self.length / T::Unpacked::LENGTH)
            + usize::from(self.length % T::Unpacked::LENGTH != 0);

        // We reserve enough space here for self.length rounded up to the next multiple of
        // T::Unpacked::LENGTH so that we can safely just write into that memory. Otherwise, we
        // would have to make a special path where we memcopy twice which is less than ideal.
        vec.reserve(num_packs * T::Unpacked::LENGTH);

        // IMPORTANT: This pointer calculation has to appear after the reserve since that reserve
        // might move the buffer.
        let mut unpacked_ptr = vec.as_mut_ptr().wrapping_add(vec.len());

        for _ in 0..num_packs {
            // This unwrap should never fail since the packed length is checked on initialized of
            // the `Decoder`.
            let packed = self.packed.next().unwrap();

            // SAFETY:
            // Since we did a `vec::reserve` before with the total length, we know that the memory
            // necessary for a `T::Unpacked` is available.
            //
            // - The elements in this buffer are properly aligned, so elements in a slice will also
            // be properly aligned.
            // - It is deferencable because it is (i) not null, (ii) in one allocated object, (iii)
            // not pointing to deallocated memory, (iv) we do not rely on atomicity and (v) we do
            // not read or write beyond the lifetime of `vec`.
            // - All data is initialized before reading it. This is not perfect but should not lead
            // to any UB.
            // - We don't alias the same data from anywhere else at the same time, because we have
            // the mutable reference to `vec`.
            let unpacked_ref = unsafe { (unpacked_ptr as *mut T::Unpacked).as_mut() }.unwrap();

            decode_pack::<T>(packed, self.num_bits, unpacked_ref);

            unpacked_ptr = unpacked_ptr.wrapping_add(T::Unpacked::LENGTH);
        }

        // SAFETY:
        // We have written these elements before so we know that these are available now.
        //
        // - The capacity is larger since we reserved enough spaced with the opening
        // `vec::reserve`.
        // - All elements are initialized by the `decode_pack` into the `unpacked_ref`.
        unsafe { vec.set_len(vec.len() + self.length) }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::case1;
    use super::*;

    impl<'a, T: Unpackable> Decoder<'a, T> {
        pub fn collect(self) -> Vec<T> {
            let mut vec = Vec::new();
            self.collect_into(&mut vec);
            vec
        }
    }

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
            .collect();
        assert_eq!(decoded, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn decode_large() {
        let (num_bits, expected, data) = case1();

        let decoded = Decoder::<u32>::try_new(&data, num_bits, expected.len())
            .unwrap()
            .collect();
        assert_eq!(decoded, expected);
    }

    #[test]
    fn test_decode_bool() {
        let num_bits = 1;
        let length = 8;
        let data = vec![0b10101010];

        let decoded = Decoder::<u32>::try_new(&data, num_bits, length)
            .unwrap()
            .collect();
        assert_eq!(decoded, vec![0, 1, 0, 1, 0, 1, 0, 1]);
    }

    #[test]
    fn test_decode_u64() {
        let num_bits = 1;
        let length = 8;
        let data = vec![0b10101010];

        let decoded = Decoder::<u64>::try_new(&data, num_bits, length)
            .unwrap()
            .collect();
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
            .collect();
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
            .collect();
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
