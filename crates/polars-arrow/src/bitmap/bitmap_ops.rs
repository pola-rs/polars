use std::ops::{BitAnd, BitOr, BitXor, Not};

use super::Bitmap;
use super::bitmask::BitMask;
use super::utils::{BitChunk, BitChunkIterExact, BitChunksExact};
use crate::bitmap::MutableBitmap;
use crate::trusted_len::TrustedLen;

#[inline(always)]
pub(crate) fn push_bitchunk<T: BitChunk>(buffer: &mut Vec<u8>, value: T) {
    buffer.extend(value.to_ne_bytes())
}

/// Creates a [`Vec<u8>`] from a [`TrustedLen`] of [`BitChunk`].
pub fn chunk_iter_to_vec<T: BitChunk, I: TrustedLen<Item = T>>(iter: I) -> Vec<u8> {
    let cap = iter.size_hint().0 * size_of::<T>();
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
    let cap = (iter.size_hint().0 + 1) * size_of::<T>();
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
pub fn binary_mask_fold<B, F, R>(lhs: BitMask<'_>, rhs: BitMask<'_>, op: F, init: B, fold: R) -> B
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

    let bitmap: Bitmap = std::iter::repeat_n(false, new_offset)
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

/// Dilate set bits to the following `w.saturating_sub(1)` positions.
///
/// `out[i] = OR_{j=max(0, i-w+1)..=i} in[j]`, i.e. every set bit is smeared to cover a
/// window of `w` positions ending at itself. Computed by repeated doubling: each round ORs
/// the words with themselves shifted left by `min(covered, w - covered)`, so after a round
/// `covered` positions are covered, turning an O(w) smear into O(log w) rounds.
pub fn dilate(bitmap: &Bitmap, w: usize, out_len: usize) -> Bitmap {
    assert!(out_len >= bitmap.len());
    let num_words = out_len.div_ceil(64);
    let mut chunks = bitmap.fast_iter_u64();
    let mut words = Vec::with_capacity(num_words);
    words.extend(&mut chunks);
    let (remainder, remainder_len) = chunks.remainder();
    words.extend_from_slice(&remainder[..remainder_len.div_ceil(64)]);
    words.resize(num_words, 0);

    // Double the covered range each round without reading writes from that round:
    // words |= words << min(covered, w - covered), then covered += shift,
    // so `covered` doubles (clamped to w) each round.
    let mut covered = 1usize;
    let mut previous = vec![0u64; words.len()];
    while covered < w {
        let shift = covered.min(w - covered);
        covered += shift;
        let (word_shift, bit_shift) = (shift / 64, (shift % 64) as u32);
        if word_shift == 0 {
            let inverse_shift = 64 - bit_shift;
            let mut previous_word = 0u64;
            for word in &mut words {
                let current_word = *word;
                *word |= previous_word >> inverse_shift | current_word << bit_shift;
                previous_word = current_word;
            }
        } else {
            previous.copy_from_slice(&words);
            if bit_shift == 0 {
                for (dst, &src) in words.iter_mut().skip(word_shift).zip(&previous) {
                    *dst |= src;
                }
            } else {
                let inverse_shift = 64 - bit_shift;
                for (dst, src) in words
                    .iter_mut()
                    .skip(word_shift + 1)
                    .zip(previous.array_windows::<2>())
                {
                    *dst |= src[0] >> inverse_shift | src[1] << bit_shift;
                }
                if word_shift < words.len() {
                    words[word_shift] |= words[0] << bit_shift;
                }
            }
        }
    }

    // No-op on little-endian targets.
    for word in &mut words {
        *word = word.to_le();
    }
    Bitmap::from_u8_vec(bytemuck::cast_slice(&words).to_vec(), out_len)
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

pub fn num_intersections_with(lhs: BitMask<'_>, rhs: BitMask<'_>) -> usize {
    binary_mask_fold(
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

impl<'b> BitOr<&'b Bitmap> for &Bitmap {
    type Output = Bitmap;

    fn bitor(self, rhs: &'b Bitmap) -> Bitmap {
        or(self, rhs)
    }
}

impl<'b> BitAnd<&'b Bitmap> for &Bitmap {
    type Output = Bitmap;

    fn bitand(self, rhs: &'b Bitmap) -> Bitmap {
        and(self, rhs)
    }
}

impl<'b> BitXor<&'b Bitmap> for &Bitmap {
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

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::bitmap::proptest::bitmap;

    fn two_equal_length_bitmaps() -> impl Strategy<Value = (Bitmap, Bitmap)> {
        (1..=250usize).prop_flat_map(|length| {
            (
                bitmap(length..300),
                bitmap(length..300),
                0..length,
                0..length,
            )
                .prop_flat_map(move |(lhs, rhs, lhs_offset, rhs_offset)| {
                    (0..usize::min(length - lhs_offset, length - rhs_offset)).prop_map(
                        move |slice_length| {
                            (
                                lhs.clone().sliced(lhs_offset, slice_length),
                                rhs.clone().sliced(rhs_offset, slice_length),
                            )
                        },
                    )
                })
        })
    }

    fn sliced_bitmap() -> impl Strategy<Value = Bitmap> {
        bitmap(1..300).prop_flat_map(|b| {
            (0..b.len(), 1..=b.len()).prop_map(move |(offset, len)| {
                let len = len.min(b.len() - offset);
                b.clone().sliced(offset, len)
            })
        })
    }

    proptest! {
        #[test]
        fn test_dilate(
            bitmap in sliced_bitmap(),
            w in 0..300usize,
            extra in 0..300usize,
        ) {
            let out_len = bitmap.len() + extra;
            let out = dilate(&bitmap, w, out_len);
            let expected: Bitmap = (0..out_len)
                .map(|i| {
                    let start = i.saturating_sub(w.saturating_sub(1));
                    (start..=i).any(|j| j < bitmap.len() && bitmap.get_bit(j))
                })
                .collect();
            prop_assert_eq!(out, expected);
        }
    }

    proptest! {
        #[test]
        fn test_num_intersections_with(
            (lhs, rhs) in two_equal_length_bitmaps()
        ) {
            let kernel_out = num_intersections_with(BitMask::from_bitmap(&lhs), BitMask::from_bitmap(&rhs));
            let mut reference_out = 0;
            for (l, r) in lhs.iter().zip(rhs.iter()) {
                reference_out += usize::from(l & r);
            }

            prop_assert_eq!(kernel_out, reference_out);
        }
    }
}
