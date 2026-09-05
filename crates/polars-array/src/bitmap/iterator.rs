use std::ops::Range;

use arrow::bitmap::utils::get_bit_unchecked;
use arrow::trusted_len::TrustedLen;

use crate::bitmap::PlBitmapRef;

/// Iterator over the bits of a [`PlBitmap`](super::PlBitmap) or a [`PlBitmapRef`].
#[derive(Clone)]
pub struct PlBitmapIter<'a> {
    repr: ArrayRepr<'a>,
}

/// The representation the mask turned out to be in, resolved once.
#[derive(Clone)]
enum ArrayRepr<'a> {
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
                repr: ArrayRepr::Scalar {
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
            repr: ArrayRepr::Flat { bytes, range },
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
            ArrayRepr::Flat { bytes, range } => range.next().map(|i| bit(bytes, i)),
            ArrayRepr::Scalar { bit, remaining } => {
                *remaining = remaining.checked_sub(1)?;
                Some(*bit)
            },
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<bool> {
        match &mut self.repr {
            ArrayRepr::Flat { bytes, range } => range.nth(n).map(|i| bit(bytes, i)),
            ArrayRepr::Scalar { bit, remaining } => {
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
    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, bool) -> B,
    {
        match self.repr {
            ArrayRepr::Flat { bytes, range } => range.fold(init, |acc, i| f(acc, bit(bytes, i))),
            ArrayRepr::Scalar {
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
            ArrayRepr::Flat { bytes, range } => range.next_back().map(|i| bit(bytes, i)),
            ArrayRepr::Scalar { bit, remaining } => {
                *remaining = remaining.checked_sub(1)?;
                Some(*bit)
            },
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<bool> {
        if let ArrayRepr::Flat { bytes, range } = &mut self.repr {
            return range.nth_back(n).map(|i| bit(bytes, i));
        }

        // Every position of a scalar mask yields the one bit it is backed by, so walking in from
        // either end drops the same number of them and reads the same bit; the default would walk
        // there one position at a time, which a mask of a billion bits does not come back from.
        self.nth(n)
    }

    /// Hoists the representation out of the loop, the way [`Iterator::fold`] does.
    #[inline]
    fn rfold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, bool) -> B,
    {
        match self.repr {
            ArrayRepr::Flat { bytes, range } => range.rfold(init, |acc, i| f(acc, bit(bytes, i))),
            ArrayRepr::Scalar {
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
            ArrayRepr::Flat { range, .. } => range.len(),
            ArrayRepr::Scalar { remaining, .. } => *remaining,
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

/// The bits of a flat validity mask, read by the position of the element they stand for rather
/// than walked.
///
/// Reading a bit by position costs no branch and keeps no state, which is what lets the loop over
/// the values keep the shape their own representation gives it. Zipping an iterator of bits onto
/// them instead would put a step of that iterator — and the branch inside it — back into the loop,
/// and would leave the values stepped one at a time rather than folded.
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

    /// Whether the element `i` positions on from the front is valid.
    ///
    /// # Safety
    /// `i` must be below [`Self::len`].
    #[inline(always)]
    pub(crate) unsafe fn get_unchecked(&self, i: usize) -> bool {
        debug_assert!(i < self.len);
        // The positions the mask covers are in bounds of the bytes backing it, which is what
        // `ValidityIter` holds of the mask it was built from.
        bit(self.bytes, self.offset + i)
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
    /// The mask is hoisted out of the loop and `values` is folded rather than stepped, so the
    /// representation of both is resolved once per walk instead of once per element.
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

                let mut i = 0;
                values.fold(init, |acc, value| {
                    // SAFETY: the mask has a bit for every value, and this is the `i`th of them.
                    let is_valid = unsafe { mask.get_unchecked(i) };
                    i += 1;
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

                // The walk starts at the bit of the element at the back, which is the last of the
                // ones the mask covers, and steps down to the front.
                let mut i = mask.len();
                values.rfold(init, |acc, value| {
                    i -= 1;
                    // SAFETY: the mask has a bit for every value, and this is the `i`th of them.
                    let is_valid = unsafe { mask.get_unchecked(i) };
                    f(acc, is_valid.then_some(value))
                })
            },
        }
    }
}

impl<'a> ValidityIter<'a> {
    #[inline]
    pub(crate) fn new(validity: Option<PlBitmapRef<'a>>) -> Self {
        // An array without a mask has no null elements, which is the set bit they all share.
        let Some(validity) = validity else {
            return Self::Scalar(true);
        };

        match validity.flat_bitmap() {
            Some(bitmap) => {
                // A mask whose bits are all the same says no more than the single bit they share,
                // and saying it that way keeps the fold off the bit-reading path. The count is
                // only read where the bitmap already caches it, so this stays `O(1)`.
                if let Some(unset_bits) = bitmap.lazy_unset_bits() {
                    if unset_bits == 0 {
                        return Self::Scalar(true);
                    }
                    if unset_bits == bitmap.len() {
                        return Self::Scalar(false);
                    }
                }

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
            // Dropping more positions than the mask has left leaves the back at or before the
            // front, which is the mask covering nothing — the same as walking it to its end.
            *back = back.saturating_sub(n);
        }

        self.next_back()
    }

    /// Whether the element the values are about to yield at the front is valid, without checking
    /// that the mask still covers one.
    ///
    /// The mask of an element iterator is walked in lockstep with the values and holds a bit for
    /// every one of them, so a value yielded is itself the proof that a bit is there to read. The
    /// check [`Self::next`] makes instead is a branch the loop over the values cannot be unrolled
    /// across, which costs more than the read it guards.
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

    /// Whether the element the values are about to yield at the back is valid, without checking
    /// that the mask still covers one.
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

    /// Whether the element the values are about to yield `n` positions on is valid, without
    /// checking that the mask still covers one.
    ///
    /// # Safety
    /// The mask must still cover the element `n` positions on from the front.
    #[inline(always)]
    pub(crate) unsafe fn nth_unchecked(&mut self, n: usize) -> bool {
        if let Self::Flat { front, .. } = self {
            // The element `n` positions on is one the mask covers, so its bit is below the back.
            *front += n;
        }

        // SAFETY: the mask covers the element now at the front, per the caller.
        unsafe { self.next_unchecked() }
    }

    /// Whether the element the values are about to yield `n` positions in from the back is valid,
    /// without checking that the mask still covers one.
    ///
    /// # Safety
    /// The mask must still cover the element `n` positions in from the back.
    #[inline(always)]
    pub(crate) unsafe fn nth_back_unchecked(&mut self, n: usize) -> bool {
        if let Self::Flat { back, .. } = self {
            // The element `n` positions in is one the mask covers, so its bit is at or above the
            // front.
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
                // Walking past the end leaves the front at or beyond the back, which `nth` and
                // `nth_back` reach in one step; either way no bit is left.
                len: back.saturating_sub(front),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow::bitmap::Bitmap;

    use super::{ValidityFold, ValidityIter};
    use crate::bitmap::PlBitmapRef;
    use crate::iterator_tests::assert_iterates;
    use crate::{PlBitmap, PlPrimitiveArray};

    #[test]
    fn a_uniform_mask_is_walked_as_the_bit_it_shares() {
        fn uniform(bitmap: &Bitmap) -> ValidityFold<'_> {
            assert!(
                bitmap.lazy_unset_bits().is_some(),
                "the count must be cached, or nothing is collapsed",
            );
            let mask = PlBitmapRef::new(bitmap, bitmap.len());
            ValidityIter::new(Some(mask)).into_mask()
        }

        let set = Bitmap::new_with_value(true, 100);
        let unset = Bitmap::new_with_value(false, 100);
        assert!(matches!(uniform(&set), ValidityFold::Valid));
        assert!(matches!(uniform(&unset), ValidityFold::Null));

        // A mask that is not uniform is still read bit by bit.
        let mixed = Bitmap::from_iter([true, false, true]);
        assert_eq!(mixed.unset_bits(), 1);
        let mask = PlBitmapRef::new(&mixed, 3);
        assert!(matches!(
            ValidityIter::new(Some(mask)).into_mask(),
            ValidityFold::Bits(_),
        ));
    }

    #[test]
    fn collapsing_a_uniform_mask_keeps_the_elements_it_yields() {
        let values = vec![1i32, 2, 3];

        let all_valid = PlPrimitiveArray::from_vec(values.clone())
            .with_validity(Some(PlBitmap::from_bitmap(Bitmap::new_with_value(true, 3))));
        assert_iterates(all_valid.iter(), &[Some(1), Some(2), Some(3)]);

        let all_null = PlPrimitiveArray::from_vec(values).with_validity(Some(
            PlBitmap::from_bitmap(Bitmap::new_with_value(false, 3)),
        ));
        assert_iterates(all_null.iter(), &[None, None, None]);
    }

    #[test]
    fn flat() {
        let mask = PlBitmap::from_iter([true, false, true, true]);
        assert_iterates(mask.iter(), &[true, false, true, true]);
    }

    #[test]
    fn scalar() {
        let mask = PlBitmap::new_scalar(true, 5);
        assert_iterates(mask.iter(), &[true; 5]);
        assert_iterates(PlBitmap::new_scalar(false, 3).iter(), &[false; 3]);
    }

    #[test]
    fn a_scalar_mask_is_not_materialized() {
        // Walking a billion bits would not finish; the scalar path must hit.
        let mask = PlBitmap::new_scalar(true, 1_000_000_000);

        assert_eq!(mask.iter().count(), 1_000_000_000);
        assert_eq!(mask.iter().nth(999_999_999), Some(true));
        assert_eq!(mask.iter().nth_back(999_999_999), Some(true));
        assert_eq!(mask.iter().last(), Some(true));
        assert_eq!(mask.iter().len(), 1_000_000_000);
    }

    /// The bits of a flat mask, read by position rather than walked, which is what the fold of an
    /// element iterator reads them by.
    #[test]
    fn bits_are_read_by_position() {
        let bits = [true, false, true, true, false];
        let mask = PlBitmap::from_iter(bits);
        // A mask that starts partway into its bytes, as the mask of a sliced array does.
        let sliced = PlBitmap::from_iter([false, false].into_iter().chain(bits)).sliced(2, 5);

        for mask in [mask, sliced] {
            let ValidityFold::Bits(read) = ValidityIter::new(Some(mask.as_ref())).into_mask()
            else {
                panic!("a mask of mixed bits is read bit by bit");
            };

            assert_eq!(read.len(), bits.len());
            for (i, &bit) in bits.iter().enumerate() {
                // SAFETY: `i` is below the number of bits the mask covers.
                assert_eq!(unsafe { read.get_unchecked(i) }, bit, "bit {i}");
            }
            assert_eq!(read.iter().collect::<Vec<_>>(), bits, "walked");
        }
    }
}
