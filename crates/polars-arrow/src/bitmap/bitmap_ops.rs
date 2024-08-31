use std::ops::{BitAnd, BitOr, BitXor, Not};

use super::utils::{BitChunk, BitChunkIterExact, BitChunksExact};
use super::Bitmap;
use crate::bitmap::MutableBitmap;
use crate::trusted_len::TrustedLen;

#[inline(always)]
pub(crate) fn push_bitchunk<T: BitChunk>(buffer: &mut Vec<u8>, value: T) {
    buffer.extend(value.to_ne_bytes())
}

/// Creates a [`Vec<u8>`] from a [`TrustedLen`] of [`BitChunk`].
pub fn chunk_iter_to_vec<T: BitChunk, I: TrustedLen<Item = T>>(iter: I) -> Vec<u8> {
    let cap = iter.size_hint().0 * std::mem::size_of::<T>();
    let mut buffer = Vec::with_capacity(cap);
    for v in iter {
        push_bitchunk(&mut buffer, v)
    }
    buffer
}

fn chunk_iter_to_vec_and_remainder<T: BitChunk, I: TrustedLen<Item = T>>(
    iter: I,
    remainder: T,
) -> Vec<u8> {
    let cap = (iter.size_hint().0 + 1) * std::mem::size_of::<T>();
    let mut buffer = Vec::with_capacity(cap);
    for v in iter {
        push_bitchunk(&mut buffer, v)
    }
    push_bitchunk(&mut buffer, remainder);
    debug_assert_eq!(buffer.len(), cap);
    buffer
}

/// Apply a bitwise operation `op` to four inputs and return the result as a [`Bitmap`].
pub fn quaternary<F>(a1: &Bitmap, a2: &Bitmap, a3: &Bitmap, a4: &Bitmap, op: F) -> Bitmap
where
    F: Fn(u64, u64, u64, u64) -> u64,
{
    assert_eq!(a1.len(), a2.len());
    assert_eq!(a1.len(), a3.len());
    assert_eq!(a1.len(), a4.len());
    let a1_chunks = a1.chunks();
    let a2_chunks = a2.chunks();
    let a3_chunks = a3.chunks();
    let a4_chunks = a4.chunks();

    let rem_a1 = a1_chunks.remainder();
    let rem_a2 = a2_chunks.remainder();
    let rem_a3 = a3_chunks.remainder();
    let rem_a4 = a4_chunks.remainder();

    let chunks = a1_chunks
        .zip(a2_chunks)
        .zip(a3_chunks)
        .zip(a4_chunks)
        .map(|(((a1, a2), a3), a4)| op(a1, a2, a3, a4));

    let buffer = chunk_iter_to_vec_and_remainder(chunks, op(rem_a1, rem_a2, rem_a3, rem_a4));
    let length = a1.len();

    Bitmap::from_u8_vec(buffer, length)
}

/// Apply a bitwise operation `op` to three inputs and return the result as a [`Bitmap`].
pub fn ternary<F>(a1: &Bitmap, a2: &Bitmap, a3: &Bitmap, op: F) -> Bitmap
where
    F: Fn(u64, u64, u64) -> u64,
{
    assert_eq!(a1.len(), a2.len());
    assert_eq!(a1.len(), a3.len());
    let a1_chunks = a1.chunks();
    let a2_chunks = a2.chunks();
    let a3_chunks = a3.chunks();

    let rem_a1 = a1_chunks.remainder();
    let rem_a2 = a2_chunks.remainder();
    let rem_a3 = a3_chunks.remainder();

    let chunks = a1_chunks
        .zip(a2_chunks)
        .zip(a3_chunks)
        .map(|((a1, a2), a3)| op(a1, a2, a3));

    let buffer = chunk_iter_to_vec_and_remainder(chunks, op(rem_a1, rem_a2, rem_a3));
    let length = a1.len();

    Bitmap::from_u8_vec(buffer, length)
}

/// Apply a bitwise operation `op` to two inputs and return the result as a [`Bitmap`].
pub fn binary<F>(lhs: &Bitmap, rhs: &Bitmap, op: F) -> Bitmap
where
    F: Fn(u64, u64) -> u64,
{
    assert_eq!(lhs.len(), rhs.len());
    let lhs_chunks = lhs.chunks();
    let rhs_chunks = rhs.chunks();
    let rem_lhs = lhs_chunks.remainder();
    let rem_rhs = rhs_chunks.remainder();

    let chunks = lhs_chunks
        .zip(rhs_chunks)
        .map(|(left, right)| op(left, right));

    let buffer = chunk_iter_to_vec_and_remainder(chunks, op(rem_lhs, rem_rhs));
    let length = lhs.len();

    Bitmap::from_u8_vec(buffer, length)
}

/// Apply a bitwise operation `op` to two inputs and fold the result.
pub fn binary_fold<B, F, R>(lhs: &Bitmap, rhs: &Bitmap, op: F, init: B, fold: R) -> B
where
    F: Fn(u64, u64) -> B,
    R: Fn(B, B) -> B,
{
    assert_eq!(lhs.len(), rhs.len());
    let lhs_chunks = lhs.chunks();
    let rhs_chunks = rhs.chunks();
    let rem_lhs = lhs_chunks.remainder();
    let rem_rhs = rhs_chunks.remainder();

    let result = lhs_chunks
        .zip(rhs_chunks)
        .fold(init, |prev, (left, right)| fold(prev, op(left, right)));

    fold(result, op(rem_lhs, rem_rhs))
}

/// Apply a bitwise operation `op` to two inputs and fold the result.
pub fn binary_fold_mut<B, F, R>(
    lhs: &MutableBitmap,
    rhs: &MutableBitmap,
    op: F,
    init: B,
    fold: R,
) -> B
where
    F: Fn(u64, u64) -> B,
    R: Fn(B, B) -> B,
{
    assert_eq!(lhs.len(), rhs.len());
    let lhs_chunks = lhs.chunks();
    let rhs_chunks = rhs.chunks();
    let rem_lhs = lhs_chunks.remainder();
    let rem_rhs = rhs_chunks.remainder();

    let result = lhs_chunks
        .zip(rhs_chunks)
        .fold(init, |prev, (left, right)| fold(prev, op(left, right)));

    fold(result, op(rem_lhs, rem_rhs))
}

fn unary_impl<F, I>(iter: I, op: F, length: usize) -> Bitmap
where
    I: BitChunkIterExact<u64>,
    F: Fn(u64) -> u64,
{
    let rem = op(iter.remainder());
    let buffer = chunk_iter_to_vec_and_remainder(iter.map(op), rem);

    Bitmap::from_u8_vec(buffer, length)
}

/// Apply a bitwise operation `op` to one input and return the result as a [`Bitmap`].
pub fn unary<F>(lhs: &Bitmap, op: F) -> Bitmap
where
    F: Fn(u64) -> u64,
{
    let (slice, offset, length) = lhs.as_slice();
    if offset == 0 {
        let iter = BitChunksExact::<u64>::new(slice, length);
        unary_impl(iter, op, lhs.len())
    } else {
        let iter = lhs.chunks::<u64>();
        unary_impl(iter, op, lhs.len())
    }
}

// create a new [`Bitmap`] semantically equal to ``bitmap`` but with an offset equal to ``offset``
pub(crate) fn align(bitmap: &Bitmap, new_offset: usize) -> Bitmap {
    let length = bitmap.len();

    let bitmap: Bitmap = std::iter::repeat(false)
        .take(new_offset)
        .chain(bitmap.iter())
        .collect();

    bitmap.sliced(new_offset, length)
}

/// Compute bitwise A AND B operation.
pub fn and(lhs: &Bitmap, rhs: &Bitmap) -> Bitmap {
    if lhs.unset_bits() == lhs.len() || rhs.unset_bits() == rhs.len() {
        assert_eq!(lhs.len(), rhs.len());
        Bitmap::new_zeroed(lhs.len())
    } else {
        binary(lhs, rhs, |x, y| x & y)
    }
}

/// Compute bitwise A AND NOT B operation.
pub fn and_not(lhs: &Bitmap, rhs: &Bitmap) -> Bitmap {
    binary(lhs, rhs, |x, y| x & !y)
}

/// Compute bitwise A OR B operation.
pub fn or(lhs: &Bitmap, rhs: &Bitmap) -> Bitmap {
    if lhs.unset_bits() == 0 || rhs.unset_bits() == 0 {
        assert_eq!(lhs.len(), rhs.len());
        let mut mutable = MutableBitmap::with_capacity(lhs.len());
        mutable.extend_constant(lhs.len(), true);
        mutable.into()
    } else {
        binary(lhs, rhs, |x, y| x | y)
    }
}

/// Compute bitwise A OR NOT B operation.
pub fn or_not(lhs: &Bitmap, rhs: &Bitmap) -> Bitmap {
    binary(lhs, rhs, |x, y| x | !y)
}

/// Compute bitwise XOR operation.
pub fn xor(lhs: &Bitmap, rhs: &Bitmap) -> Bitmap {
    let lhs_nulls = lhs.unset_bits();
    let rhs_nulls = rhs.unset_bits();

    // all false or all true
    if lhs_nulls == rhs_nulls && rhs_nulls == rhs.len() || lhs_nulls == 0 && rhs_nulls == 0 {
        assert_eq!(lhs.len(), rhs.len());
        Bitmap::new_zeroed(rhs.len())
    }
    // all false and all true or vice versa
    else if (lhs_nulls == 0 && rhs_nulls == rhs.len())
        || (lhs_nulls == lhs.len() && rhs_nulls == 0)
    {
        assert_eq!(lhs.len(), rhs.len());
        let mut mutable = MutableBitmap::with_capacity(lhs.len());
        mutable.extend_constant(lhs.len(), true);
        mutable.into()
    } else {
        binary(lhs, rhs, |x, y| x ^ y)
    }
}

/// Compute bitwise equality (not XOR) operation.
fn eq(lhs: &Bitmap, rhs: &Bitmap) -> bool {
    if lhs.len() != rhs.len() {
        return false;
    }

    let mut lhs_chunks = lhs.chunks::<u64>();
    let mut rhs_chunks = rhs.chunks::<u64>();

    let equal_chunks = lhs_chunks
        .by_ref()
        .zip(rhs_chunks.by_ref())
        .all(|(left, right)| left == right);

    if !equal_chunks {
        return false;
    }
    let lhs_remainder = lhs_chunks.remainder_iter();
    let rhs_remainder = rhs_chunks.remainder_iter();
    lhs_remainder.zip(rhs_remainder).all(|(x, y)| x == y)
}

pub fn num_intersections_with(lhs: &Bitmap, rhs: &Bitmap) -> usize {
    binary_fold(
        lhs,
        rhs,
        |lhs, rhs| (lhs & rhs).count_ones() as usize,
        0,
        |lhs, rhs| lhs + rhs,
    )
}

pub fn intersects_with(lhs: &Bitmap, rhs: &Bitmap) -> bool {
    binary_fold(
        lhs,
        rhs,
        |lhs, rhs| lhs & rhs != 0,
        false,
        |lhs, rhs| lhs || rhs,
    )
}

pub fn intersects_with_mut(lhs: &MutableBitmap, rhs: &MutableBitmap) -> bool {
    binary_fold_mut(
        lhs,
        rhs,
        |lhs, rhs| lhs & rhs != 0,
        false,
        |lhs, rhs| lhs || rhs,
    )
}

pub fn num_edges(lhs: &Bitmap) -> usize {
    if lhs.is_empty() {
        return 0;
    }

    // @TODO: If is probably quite inefficient to do it like this because now either one is not
    // aligned. Maybe, we can implement a smarter way to do this.
    binary_fold(
        &unsafe { lhs.clone().sliced_unchecked(0, lhs.len() - 1) },
        &unsafe { lhs.clone().sliced_unchecked(1, lhs.len() - 1) },
        |l, r| (l ^ r).count_ones() as usize,
        0,
        |acc, v| acc + v,
    )
}

/// Compute `out[i] = if selector[i] { truthy[i] } else { falsy }`.
pub fn select_constant(selector: &Bitmap, truthy: &Bitmap, falsy: bool) -> Bitmap {
    let falsy_mask: u64 = if falsy {
        0xFFFF_FFFF_FFFF_FFFF
    } else {
        0x0000_0000_0000_0000
    };

    binary(selector, truthy, |s, t| (s & t) | (!s & falsy_mask))
}

/// Compute `out[i] = if selector[i] { truthy[i] } else { falsy[i] }`.
pub fn select(selector: &Bitmap, truthy: &Bitmap, falsy: &Bitmap) -> Bitmap {
    ternary(selector, truthy, falsy, |s, t, f| (s & t) | (!s & f))
}

impl PartialEq for Bitmap {
    fn eq(&self, other: &Self) -> bool {
        eq(self, other)
    }
}

impl<'a, 'b> BitOr<&'b Bitmap> for &'a Bitmap {
    type Output = Bitmap;

    fn bitor(self, rhs: &'b Bitmap) -> Bitmap {
        or(self, rhs)
    }
}

impl<'a, 'b> BitAnd<&'b Bitmap> for &'a Bitmap {
    type Output = Bitmap;

    fn bitand(self, rhs: &'b Bitmap) -> Bitmap {
        and(self, rhs)
    }
}

impl<'a, 'b> BitXor<&'b Bitmap> for &'a Bitmap {
    type Output = Bitmap;

    fn bitxor(self, rhs: &'b Bitmap) -> Bitmap {
        xor(self, rhs)
    }
}

impl Not for &Bitmap {
    type Output = Bitmap;

    fn not(self) -> Bitmap {
        unary(self, |a| !a)
    }
}
