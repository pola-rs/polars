use std::fmt::Binary;
use std::ops::{BitAndAssign, Not, Shl, ShlAssign, ShrAssign};

use num_traits::PrimInt;

use super::NativeType;

/// A chunk of bits. This is used to create masks of a given length
/// whose width is `1` bit. In `portable_simd` notation, this corresponds to `m1xY`.
///
/// This (sealed) trait is implemented for [`u8`], [`u16`], [`u32`] and [`u64`].
pub trait BitChunk:
    super::private::Sealed
    + PrimInt
    + NativeType
    + Binary
    + ShlAssign
    + Not<Output = Self>
    + ShrAssign<usize>
    + ShlAssign<usize>
    + Shl<usize, Output = Self>
    + BitAndAssign
{
    /// convert itself into bytes.
    fn to_ne_bytes(self) -> Self::Bytes;
    /// convert itself from bytes.
    fn from_ne_bytes(v: Self::Bytes) -> Self;
}

macro_rules! bit_chunk {
    ($ty:ty) => {
        impl BitChunk for $ty {
            #[inline(always)]
            fn to_ne_bytes(self) -> Self::Bytes {
                self.to_ne_bytes()
            }

            #[inline(always)]
            fn from_ne_bytes(v: Self::Bytes) -> Self {
                Self::from_ne_bytes(v)
            }
        }
    };
}

bit_chunk!(u8);
bit_chunk!(u16);
bit_chunk!(u32);
bit_chunk!(u64);

/// An [`Iterator<Item=bool>`] over a [`BitChunk`].
///
/// This iterator is often compiled to SIMD.
///
/// The [LSB](https://en.wikipedia.org/wiki/Bit_numbering#Least_significant_bit) corresponds
/// to the first slot, as defined by the arrow specification.
/// # Example
/// ```
/// use polars_arrow::types::BitChunkIter;
/// let a = 0b00010000u8;
/// let iter = BitChunkIter::new(a, 7);
/// let r = iter.collect::<Vec<_>>();
/// assert_eq!(r, vec![false, false, false, false, true, false, false]);
/// ```
pub struct BitChunkIter<T: BitChunk> {
    value: T,
    mask: T,
    remaining: usize,
}

impl<T: BitChunk> BitChunkIter<T> {
    /// Creates a new [`BitChunkIter`] with `len` bits.
    #[inline]
    pub fn new(value: T, len: usize) -> Self {
        assert!(len <= std::mem::size_of::<T>() * 8);
        Self {
            value,
            remaining: len,
            mask: T::one(),
        }
    }
}

impl<T: BitChunk> Iterator for BitChunkIter<T> {
    type Item = bool;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        };
        let result = Some(self.value & self.mask != T::zero());
        self.remaining -= 1;
        self.mask <<= 1;
        result
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

// # Safety
// a mathematical invariant of this iterator
unsafe impl<T: BitChunk> crate::trusted_len::TrustedLen for BitChunkIter<T> {}

/// An [`Iterator<Item=usize>`] over a [`BitChunk`] returning the index of each bit set in the chunk
/// See <https://lemire.me/blog/2018/03/08/iterating-over-set-bits-quickly-simd-edition/> for details
/// # Example
/// ```
/// use polars_arrow::types::BitChunkOnes;
/// let a = 0b00010000u8;
/// let iter = BitChunkOnes::new(a);
/// let r = iter.collect::<Vec<_>>();
/// assert_eq!(r, vec![4]);
/// ```
pub struct BitChunkOnes<T: BitChunk> {
    value: T,
    remaining: usize,
}

impl<T: BitChunk> BitChunkOnes<T> {
    /// Creates a new [`BitChunkOnes`] with `len` bits.
    #[inline]
    pub fn new(value: T) -> Self {
        Self {
            value,
            remaining: value.count_ones() as usize,
        }
    }

    #[inline]
    pub fn from_known_count(value: T, remaining: usize) -> Self {
        Self { value, remaining }
    }
}

impl<T: BitChunk> Iterator for BitChunkOnes<T> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let v = self.value.trailing_zeros() as usize;
        self.value &= self.value - T::one();

        self.remaining -= 1;
        Some(v)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

// # Safety
// a mathematical invariant of this iterator
unsafe impl<T: BitChunk> crate::trusted_len::TrustedLen for BitChunkOnes<T> {}
