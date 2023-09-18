use crate::bitmap::{Bitmap, MutableBitmap};

use super::utils::{BitChunk, BitChunkIterExact, BitChunksExact};

/// Applies a function to every bit of this [`MutableBitmap`] in chunks
///
/// This function can be for operations like `!` to a [`MutableBitmap`].
pub fn unary_assign<T: BitChunk, F: Fn(T) -> T>(bitmap: &mut MutableBitmap, op: F) {
    let mut chunks = bitmap.bitchunks_exact_mut::<T>();

    chunks.by_ref().for_each(|chunk| {
        let new_chunk: T = match (chunk as &[u8]).try_into() {
            Ok(a) => T::from_ne_bytes(a),
            Err(_) => unreachable!(),
        };
        let new_chunk = op(new_chunk);
        chunk.copy_from_slice(new_chunk.to_ne_bytes().as_ref());
    });

    if chunks.remainder().is_empty() {
        return;
    }
    let mut new_remainder = T::zero().to_ne_bytes();
    chunks
        .remainder()
        .iter()
        .enumerate()
        .for_each(|(index, b)| new_remainder[index] = *b);
    new_remainder = op(T::from_ne_bytes(new_remainder)).to_ne_bytes();

    let len = chunks.remainder().len();
    chunks
        .remainder()
        .copy_from_slice(&new_remainder.as_ref()[..len]);
}

impl std::ops::Not for MutableBitmap {
    type Output = Self;

    #[inline]
    fn not(mut self) -> Self {
        unary_assign(&mut self, |a: u64| !a);
        self
    }
}

fn binary_assign_impl<I, T, F>(lhs: &mut MutableBitmap, mut rhs: I, op: F)
where
    I: BitChunkIterExact<T>,
    T: BitChunk,
    F: Fn(T, T) -> T,
{
    let mut lhs_chunks = lhs.bitchunks_exact_mut::<T>();

    lhs_chunks
        .by_ref()
        .zip(rhs.by_ref())
        .for_each(|(lhs, rhs)| {
            let new_chunk: T = match (lhs as &[u8]).try_into() {
                Ok(a) => T::from_ne_bytes(a),
                Err(_) => unreachable!(),
            };
            let new_chunk = op(new_chunk, rhs);
            lhs.copy_from_slice(new_chunk.to_ne_bytes().as_ref());
        });

    let rem_lhs = lhs_chunks.remainder();
    let rem_rhs = rhs.remainder();
    if rem_lhs.is_empty() {
        return;
    }
    let mut new_remainder = T::zero().to_ne_bytes();
    lhs_chunks
        .remainder()
        .iter()
        .enumerate()
        .for_each(|(index, b)| new_remainder[index] = *b);
    new_remainder = op(T::from_ne_bytes(new_remainder), rem_rhs).to_ne_bytes();

    let len = lhs_chunks.remainder().len();
    lhs_chunks
        .remainder()
        .copy_from_slice(&new_remainder.as_ref()[..len]);
}

/// Apply a bitwise binary operation to a [`MutableBitmap`].
///
/// This function can be used for operations like `&=` to a [`MutableBitmap`].
/// # Panics
/// This function panics iff `lhs.len() != `rhs.len()`
pub fn binary_assign<T: BitChunk, F>(lhs: &mut MutableBitmap, rhs: &Bitmap, op: F)
where
    F: Fn(T, T) -> T,
{
    assert_eq!(lhs.len(), rhs.len());

    let (slice, offset, length) = rhs.as_slice();
    if offset == 0 {
        let iter = BitChunksExact::<T>::new(slice, length);
        binary_assign_impl(lhs, iter, op)
    } else {
        let rhs_chunks = rhs.chunks::<T>();
        binary_assign_impl(lhs, rhs_chunks, op)
    }
}

#[inline]
/// Compute bitwise OR operation in-place
fn or_assign<T: BitChunk>(lhs: &mut MutableBitmap, rhs: &Bitmap) {
    if rhs.unset_bits() == 0 {
        assert_eq!(lhs.len(), rhs.len());
        lhs.clear();
        lhs.extend_constant(rhs.len(), true);
    } else if rhs.unset_bits() == rhs.len() {
        // bitmap remains
    } else {
        binary_assign(lhs, rhs, |x: T, y| x | y)
    }
}

impl<'a> std::ops::BitOrAssign<&'a Bitmap> for &mut MutableBitmap {
    #[inline]
    fn bitor_assign(&mut self, rhs: &'a Bitmap) {
        or_assign::<u64>(self, rhs)
    }
}

impl<'a> std::ops::BitOr<&'a Bitmap> for MutableBitmap {
    type Output = Self;

    #[inline]
    fn bitor(mut self, rhs: &'a Bitmap) -> Self {
        or_assign::<u64>(&mut self, rhs);
        self
    }
}

#[inline]
/// Compute bitwise `&` between `lhs` and `rhs`, assigning it to `lhs`
fn and_assign<T: BitChunk>(lhs: &mut MutableBitmap, rhs: &Bitmap) {
    if rhs.unset_bits() == 0 {
        // bitmap remains
    }
    if rhs.unset_bits() == rhs.len() {
        assert_eq!(lhs.len(), rhs.len());
        lhs.clear();
        lhs.extend_constant(rhs.len(), false);
    } else {
        binary_assign(lhs, rhs, |x: T, y| x & y)
    }
}

impl<'a> std::ops::BitAndAssign<&'a Bitmap> for &mut MutableBitmap {
    #[inline]
    fn bitand_assign(&mut self, rhs: &'a Bitmap) {
        and_assign::<u64>(self, rhs)
    }
}

impl<'a> std::ops::BitAnd<&'a Bitmap> for MutableBitmap {
    type Output = Self;

    #[inline]
    fn bitand(mut self, rhs: &'a Bitmap) -> Self {
        and_assign::<u64>(&mut self, rhs);
        self
    }
}

#[inline]
/// Compute bitwise XOR operation
fn xor_assign<T: BitChunk>(lhs: &mut MutableBitmap, rhs: &Bitmap) {
    binary_assign(lhs, rhs, |x: T, y| x ^ y)
}

impl<'a> std::ops::BitXorAssign<&'a Bitmap> for &mut MutableBitmap {
    #[inline]
    fn bitxor_assign(&mut self, rhs: &'a Bitmap) {
        xor_assign::<u64>(self, rhs)
    }
}

impl<'a> std::ops::BitXor<&'a Bitmap> for MutableBitmap {
    type Output = Self;

    #[inline]
    fn bitxor(mut self, rhs: &'a Bitmap) -> Self {
        xor_assign::<u64>(&mut self, rhs);
        self
    }
}
