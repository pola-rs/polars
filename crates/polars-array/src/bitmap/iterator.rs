use std::ops::Range;

use polars_arrow::bitmap::utils::{BitmapIter, get_bit_unchecked};
use polars_arrow::trusted_len::TrustedLen;

use crate::bitmap::PlBitmapRef;

/// Iterator over the bits of a [`PlBitmap`](super::PlBitmap) or a [`PlBitmapRef`].
#[derive(Clone)]
pub struct PlBitmapIter<'a> {
    repr: BitsRepr<'a>,
}

/// The representation the mask turned out to be in, resolved once.
#[derive(Clone)]
enum BitsRepr<'a> {
    /// One bit per element, walked a word at a time.
    Flat(BitmapIter<'a>),
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
            None => Self {
                repr: BitsRepr::Scalar {
                    bit: mask.scalar_value().unwrap_or(false),
                    remaining: mask.len(),
                },
            },
        }
    }

    /// The bits of `bytes` that `range` covers.
    #[inline]
    pub(crate) fn flat(bytes: &'a [u8], range: Range<usize>) -> Self {
        assert!(range.end <= bytes.len() * 8);
        Self {
            repr: BitsRepr::Flat(BitmapIter::new(bytes, range.start, range.len())),
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
            BitsRepr::Flat(bits) => bits.next(),
            BitsRepr::Scalar { bit, remaining } => {
                *remaining = remaining.checked_sub(1)?;
                Some(*bit)
            },
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<bool> {
        match &mut self.repr {
            BitsRepr::Flat(bits) => bits.nth(n),
            BitsRepr::Scalar { bit, remaining } => {
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
        self.next_back()
    }

    /// Hoists the representation out of the loop: a scalar mask folds over the one bit it shares.
    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, bool) -> B,
    {
        match self.repr {
            BitsRepr::Flat(bits) => bits.fold(init, f),
            BitsRepr::Scalar {
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
            BitsRepr::Flat(bits) => bits.next_back(),
            BitsRepr::Scalar { bit, remaining } => {
                *remaining = remaining.checked_sub(1)?;
                Some(*bit)
            },
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<bool> {
        if let BitsRepr::Flat(bits) = &mut self.repr {
            return bits.nth_back(n);
        }

        self.nth(n)
    }

    /// Hoists the representation out of the loop, the way [`Iterator::fold`] does.
    #[inline]
    fn rfold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, bool) -> B,
    {
        match self.repr {
            BitsRepr::Flat(bits) => bits.rfold(init, f),
            BitsRepr::Scalar {
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
            BitsRepr::Flat(bits) => bits.len(),
            BitsRepr::Scalar { remaining, .. } => *remaining,
        }
    }
}

unsafe impl TrustedLen for PlBitmapIter<'_> {}

/// The validity mask of an element iterator, walked in lockstep with the values.
#[derive(Clone)]
pub(crate) enum ValidityIter<'a> {
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
    /// One bit per element, to read alongside the values.
    Bits(ValidityBits<'a>),
}

/// The bits of a flat validity mask, read by the position of the element they stand for.
#[derive(Clone, Copy)]
pub(crate) struct ValidityBits<'a> {
    /// The bytes the bits live in, of which only the ones `offset` and `len` cover are this mask's.
    bytes: &'a [u8],
    /// The position in `bytes` of the bit of the first element the mask still covers.
    offset: usize,
    /// How many elements the mask has a bit for, from `offset` on.
    len: usize,
}

impl<'a> ValidityBits<'a> {
    /// How many elements the mask has a bit for.
    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// The bits, to read one per value alongside the values themselves.
    #[inline]
    pub(crate) fn words(&self) -> BitmapIter<'a> {
        BitmapIter::new(self.bytes, self.offset, self.len)
    }

    /// The bits, to walk where there are no values to read them alongside.
    #[inline]
    pub(crate) fn iter(&self) -> PlBitmapIter<'a> {
        PlBitmapIter::flat(self.bytes, self.offset..self.offset + self.len)
    }
}

impl<'a> ValidityFold<'a> {
    /// Folds `f` over the elements `values` yields, `None` where the mask says the element is null.
    ///
    /// # Safety
    /// The mask must have a bit for every value `values` has left to yield.
    #[inline]
    pub(crate) unsafe fn fold_values<I, B, F>(self, values: I, init: B, mut f: F) -> B
    where
        I: Iterator,
        F: FnMut(B, Option<I::Item>) -> B,
    {
        match self {
            Self::Valid => values.fold(init, |acc, value| f(acc, Some(value))),
            Self::Null => values.fold(init, |acc, _| f(acc, None)),
            Self::Bits(mask) => {
                debug_assert_eq!(mask.len(), values.size_hint().0);

                let mut bits = mask.words();
                values.fold(init, |acc, value| {
                    // SAFETY: the mask has a bit for every value, and this is the next of them.
                    let is_valid = unsafe { bits.next().unwrap_unchecked() };
                    f(acc, is_valid.then_some(value))
                })
            },
        }
    }

    /// Folds `f` over the elements from the back, the way [`Self::fold_values`] does from the front.
    ///
    /// # Safety
    /// The mask must have a bit for every value `values` has left to yield.
    #[inline]
    pub(crate) unsafe fn rfold_values<I, B, F>(self, values: I, init: B, mut f: F) -> B
    where
        I: DoubleEndedIterator,
        F: FnMut(B, Option<I::Item>) -> B,
    {
        match self {
            Self::Valid => values.rfold(init, |acc, value| f(acc, Some(value))),
            Self::Null => values.rfold(init, |acc, _| f(acc, None)),
            Self::Bits(mask) => {
                debug_assert_eq!(mask.len(), values.size_hint().0);

                let mut bits = mask.words();
                values.rfold(init, |acc, value| {
                    // SAFETY: the mask has a bit for every value, and this is the last of them.
                    let is_valid = unsafe { bits.next_back().unwrap_unchecked() };
                    f(acc, is_valid.then_some(value))
                })
            },
        }
    }
}

impl<'a> ValidityIter<'a> {
    #[inline]
    pub(crate) fn new(validity: Option<PlBitmapRef<'a>>) -> Self {
        let Some(validity) = validity else {
            return Self::Scalar(true);
        };

        match validity.flat_bitmap() {
            Some(bitmap) => {
                let unset_bits = bitmap.unset_bits();
                if unset_bits == 0 {
                    return Self::Scalar(true);
                }
                if unset_bits == bitmap.len() {
                    return Self::Scalar(false);
                }

                let (bytes, offset, length) = bitmap.as_slice();
                Self::Flat {
                    bytes,
                    front: offset,
                    back: offset + length,
                }
            },
            None => Self::Scalar(validity.scalar_value().unwrap_or(true)),
        }
    }

    /// Whether the element the values are about to yield at the front is valid.
    #[inline(always)]
    pub(crate) fn next(&mut self) -> bool {
        match self {
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

    /// Whether the element the values are about to yield `n` positions in from the back is valid.
    #[inline(always)]
    pub(crate) fn nth_back(&mut self, n: usize) -> bool {
        if let Self::Flat { back, .. } = self {
            *back = back.saturating_sub(n);
        }

        self.next_back()
    }

    /// Whether the element the values are about to yield at the front is valid, unchecked.
    ///
    /// # Safety
    /// The mask must still cover an element at the front.
    #[inline(always)]
    pub(crate) unsafe fn next_unchecked(&mut self) -> bool {
        match self {
            Self::Flat { bytes, front, back } => {
                debug_assert!(*front < *back);
                let i = *front;
                *front = i + 1;
                bit(bytes, i)
            },
            Self::Scalar(value) => *value,
        }
    }

    /// Whether the element the values are about to yield at the back is valid, unchecked.
    ///
    /// # Safety
    /// The mask must still cover an element at the back.
    #[inline(always)]
    pub(crate) unsafe fn next_back_unchecked(&mut self) -> bool {
        match self {
            Self::Flat { bytes, front, back } => {
                debug_assert!(*front < *back);
                *back -= 1;
                bit(bytes, *back)
            },
            Self::Scalar(value) => *value,
        }
    }

    /// Whether the element the values are about to yield `n` positions on is valid, unchecked.
    ///
    /// # Safety
    /// The mask must still cover the element `n` positions on from the front.
    #[inline(always)]
    pub(crate) unsafe fn nth_unchecked(&mut self, n: usize) -> bool {
        if let Self::Flat { front, .. } = self {
            *front += n;
        }

        // SAFETY: the mask covers the element now at the front, per the caller.
        unsafe { self.next_unchecked() }
    }

    /// Whether the element `n` positions in from the back is valid, unchecked.
    ///
    /// # Safety
    /// The mask must still cover the element `n` positions in from the back.
    #[inline(always)]
    pub(crate) unsafe fn nth_back_unchecked(&mut self, n: usize) -> bool {
        if let Self::Flat { back, .. } = self {
            *back -= n;
        }

        // SAFETY: the mask covers the element now at the back, per the caller.
        unsafe { self.next_back_unchecked() }
    }

    /// Leaves the mask covering nothing, which is where walking it to its end leaves it.
    #[inline(always)]
    pub(crate) fn exhaust(&mut self) {
        if let Self::Flat { front, back, .. } = self {
            *back = *front;
        }
    }

    /// The mask the elements left to yield are under, with its representation hoisted out.
    #[inline]
    pub(crate) fn into_mask(self) -> ValidityFold<'a> {
        match self {
            Self::Scalar(true) => ValidityFold::Valid,
            Self::Scalar(false) => ValidityFold::Null,
            Self::Flat { bytes, front, back } => ValidityFold::Bits(ValidityBits {
                bytes,
                offset: front,
                len: back.saturating_sub(front),
            }),
        }
    }
}
