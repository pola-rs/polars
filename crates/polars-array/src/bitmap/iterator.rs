use std::ops::Range;

use arrow::bitmap::utils::get_bit_unchecked;
use arrow::trusted_len::TrustedLen;

use crate::bitmap::PlBitmapRef;

/// Iterator over the bits of a [`PlBitmap`](super::PlBitmap) or a [`PlBitmapRef`].
///
/// A scalar mask is iterated without being materialized, so this is `O(1)` in memory regardless
/// of the mask's length.
///
/// Which of the two representations the mask is in is settled once, when the iterator is created,
/// rather than at every bit: a flat mask is walked by index, so that the bits it yields do not
/// depend on one another and a loop over them runs several at a time, and a scalar one is a
/// counter over the single bit every element shares.
#[derive(Clone)]
pub struct PlBitmapIter<'a> {
    repr: Repr<'a>,
}

/// The representation the mask turned out to be in, resolved once.
#[derive(Clone)]
enum Repr<'a> {
    /// One bit per element, at the positions of `bytes` that `range` covers.
    Flat {
        bytes: &'a [u8],
        range: Range<usize>,
    },
    /// The single bit every element shares, and how many are left to yield.
    Scalar { bit: bool, remaining: usize },
}

impl<'a> PlBitmapIter<'a> {
    #[inline]
    pub(crate) fn new(mask: PlBitmapRef<'a>) -> Self {
        match mask.flat_bitmap() {
            Some(bitmap) => {
                let (bytes, offset, length) = bitmap.as_slice();
                Self::flat(bytes, offset..offset + length)
            },
            // A mask that is not flat is scalar: every element reads the one bit it is backed by,
            // which an empty mask has no position left to read.
            None => Self {
                repr: Repr::Scalar {
                    bit: mask.scalar_value().unwrap_or(false),
                    remaining: mask.len(),
                },
            },
        }
    }

    /// The bits of `bytes` that `range` covers.
    ///
    /// # Panics
    /// Panics unless `range` is in bounds of the bits `bytes` holds.
    #[inline]
    pub(crate) fn flat(bytes: &'a [u8], range: Range<usize>) -> Self {
        assert!(range.end <= bytes.len() * 8);
        Self {
            repr: Repr::Flat { bytes, range },
        }
    }
}

/// The bit at `i` of `bytes`, which is in bounds of them.
#[inline(always)]
fn bit(bytes: &[u8], i: usize) -> bool {
    debug_assert!(i < bytes.len() * 8);
    // SAFETY: the positions a mask has left to yield are in bounds of the bytes it is backed by,
    // which is what its constructors check.
    unsafe { get_bit_unchecked(bytes, i) }
}

impl Iterator for PlBitmapIter<'_> {
    type Item = bool;

    #[inline]
    fn next(&mut self) -> Option<bool> {
        match &mut self.repr {
            Repr::Flat { bytes, range } => range.next().map(|i| bit(bytes, i)),
            Repr::Scalar { bit, remaining } => {
                *remaining = remaining.checked_sub(1)?;
                Some(*bit)
            },
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<bool> {
        match &mut self.repr {
            Repr::Flat { bytes, range } => range.nth(n).map(|i| bit(bytes, i)),
            Repr::Scalar { bit, remaining } => {
                let Some(left) = remaining.checked_sub(n + 1) else {
                    *remaining = 0;
                    return None;
                };
                *remaining = left;
                Some(*bit)
            },
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.len();
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn last(mut self) -> Option<bool> {
        // Walking to the last bit is what the default would do; the mask is double ended.
        self.next_back()
    }

    /// Hoists the representation out of the loop: a flat mask folds over the positions it covers,
    /// which are independent of one another, and a scalar one folds over the single bit it shares.
    /// `for_each`, `collect` and friends route through here.
    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, bool) -> B,
    {
        match self.repr {
            Repr::Flat { bytes, range } => range.fold(init, |acc, i| f(acc, bit(bytes, i))),
            Repr::Scalar {
                bit: value,
                remaining,
            } => {
                let mut acc = init;
                for _ in 0..remaining {
                    acc = f(acc, value);
                }
                acc
            },
        }
    }
}

impl DoubleEndedIterator for PlBitmapIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<bool> {
        match &mut self.repr {
            Repr::Flat { bytes, range } => range.next_back().map(|i| bit(bytes, i)),
            Repr::Scalar { bit, remaining } => {
                *remaining = remaining.checked_sub(1)?;
                Some(*bit)
            },
        }
    }

    /// Hoists the representation out of the loop, the way [`Iterator::fold`] does.
    #[inline]
    fn rfold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, bool) -> B,
    {
        match self.repr {
            Repr::Flat { bytes, range } => range.rfold(init, |acc, i| f(acc, bit(bytes, i))),
            Repr::Scalar {
                bit: value,
                remaining,
            } => {
                let mut acc = init;
                for _ in 0..remaining {
                    acc = f(acc, value);
                }
                acc
            },
        }
    }
}

impl ExactSizeIterator for PlBitmapIter<'_> {
    #[inline]
    fn len(&self) -> usize {
        match &self.repr {
            Repr::Flat { range, .. } => range.len(),
            Repr::Scalar { remaining, .. } => *remaining,
        }
    }
}

unsafe impl TrustedLen for PlBitmapIter<'_> {}

/// The validity mask of an element iterator, walked in lockstep with the values.
///
/// This is what keeps reading validity off the critical path of an element: the mask holds one bit
/// per element the values yield, so walking the two together needs no index of its own, and which
/// representation the mask is in — absent, flat, or the single bit every element shares — is
/// settled once rather than at every element.
///
/// The values iterator is the one that governs how many elements there are. A position past the
/// end of the mask therefore belongs to no element, and reads as valid rather than panicking.
#[derive(Clone)]
pub(crate) enum ValidityIter<'a> {
    /// No mask at all, which is an array without null elements.
    All,
    /// One bit per element, at the positions `front..back` of `bytes`.
    Flat {
        bytes: &'a [u8],
        front: usize,
        back: usize,
    },
    /// The single bit every element shares.
    Scalar(bool),
}

/// What a [`ValidityIter`] leaves a fold to walk, once its representation is hoisted out.
pub(crate) enum ValidityFold<'a> {
    /// Every element is valid, so the fold reads no mask at all.
    Valid,
    /// Every element is null, so the fold reads no mask and no value either.
    Null,
    /// One bit per element, to zip the values with.
    Bits(PlBitmapIter<'a>),
}

impl<'a> ValidityIter<'a> {
    #[inline]
    pub(crate) fn new(validity: Option<PlBitmapRef<'a>>) -> Self {
        let Some(validity) = validity else {
            return Self::All;
        };

        match validity.flat_bitmap() {
            Some(bitmap) => {
                let (bytes, offset, length) = bitmap.as_slice();
                Self::Flat {
                    bytes,
                    front: offset,
                    back: offset + length,
                }
            },
            // A mask that is not flat is scalar: every element reads the one bit it is backed by,
            // which an empty mask has no element left to read.
            None => Self::Scalar(validity.scalar_value().unwrap_or(true)),
        }
    }

    /// Whether the element the values are about to yield at the front is valid.
    #[inline(always)]
    pub(crate) fn next(&mut self) -> bool {
        match self {
            Self::All => true,
            Self::Flat { bytes, front, back } => {
                if *front >= *back {
                    return true;
                }
                let i = *front;
                *front = i + 1;
                bit(bytes, i)
            },
            Self::Scalar(value) => *value,
        }
    }

    /// Whether the element the values are about to yield at the back is valid.
    #[inline(always)]
    pub(crate) fn next_back(&mut self) -> bool {
        match self {
            Self::All => true,
            Self::Flat { bytes, front, back } => {
                if *front >= *back {
                    return true;
                }
                *back -= 1;
                bit(bytes, *back)
            },
            Self::Scalar(value) => *value,
        }
    }

    /// Whether the element the values are about to yield `n` positions on is valid.
    #[inline(always)]
    pub(crate) fn nth(&mut self, n: usize) -> bool {
        if let Self::Flat { front, .. } = self {
            *front = front.saturating_add(n);
        }

        self.next()
    }

    /// The mask the elements left to yield are under, with its representation hoisted out.
    #[inline]
    pub(crate) fn into_mask(self) -> ValidityFold<'a> {
        match self {
            Self::All | Self::Scalar(true) => ValidityFold::Valid,
            Self::Scalar(false) => ValidityFold::Null,
            Self::Flat { bytes, front, back } => {
                ValidityFold::Bits(PlBitmapIter::flat(bytes, front..back))
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow::bitmap::Bitmap;

    use crate::PlBitmap;
    use crate::iterator_tests::assert_iterates;

    #[test]
    fn flat() {
        let mask = PlBitmap::from_iter([true, false, true, true]);
        assert_iterates(mask.iter(), &[true, false, true, true]);
    }

    #[test]
    fn flat_at_an_offset() {
        // A sliced bitmap starts part way into a byte, which is the other path through the words
        // a flat mask is walked in.
        let mask = PlBitmap::from_iter([true; 200]).sliced(3, 130);
        assert_iterates(mask.iter(), &[true; 130]);

        let bits: Vec<bool> = (0..200).map(|i| i % 3 == 0).collect();
        let mask = PlBitmap::from_iter(bits.iter().copied()).sliced(5, 100);
        assert_iterates(mask.iter(), &bits[5..105]);
    }

    #[test]
    fn scalar() {
        let mask = PlBitmap::new_scalar(true, 5);
        assert_iterates(mask.iter(), &[true; 5]);
        assert_iterates(PlBitmap::new_scalar(false, 3).iter(), &[false; 3]);
    }

    #[test]
    fn empty() {
        assert_iterates(PlBitmap::new_empty().iter(), &[]);
        // An empty mask backed by a stray bit is scalar rather than flat, and reads no bit.
        assert_iterates(PlBitmap::new_scalar(true, 0).iter(), &[]);
    }

    #[test]
    fn one_bit_is_both_representations() {
        assert_iterates(PlBitmap::from_iter([true]).iter(), &[true]);
        assert_iterates(PlBitmap::new_scalar(false, 1).iter(), &[false]);
    }

    #[test]
    fn a_scalar_mask_is_not_materialized() {
        // Walking a billion bits would not finish; the scalar path must hit.
        let mask = PlBitmap::new_scalar(true, 1_000_000_000);

        assert_eq!(mask.iter().count(), 1_000_000_000);
        assert_eq!(mask.iter().nth(999_999_999), Some(true));
        assert_eq!(mask.iter().last(), Some(true));
        assert_eq!(mask.iter().len(), 1_000_000_000);
    }

    #[test]
    fn borrowed_masks_iterate_the_same() {
        let bitmap = Bitmap::from_iter([true, false, true]);
        assert_iterates(
            PlBitmap::from_bitmap(bitmap).as_ref().iter(),
            &[true, false, true],
        );
    }
}
