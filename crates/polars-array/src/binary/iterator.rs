use std::marker::PhantomData;
use std::ptr::NonNull;
use std::slice;

use arrow::trusted_len::TrustedLen;

use crate::bitmap::{PlBitmapRef, ValidityFold, ValidityIter};
use crate::broadcast::is_flat_offsets_len;

/// Iterator over the elements of a [`PlBinaryArray`](super::PlBinaryArray), ignoring validity.
#[derive(Clone)]
pub struct PlBinaryValuesIter<'a> {
    /// The bytes the offsets cut the elements out of. How many there are is never asked: every
    /// offset is in bounds of them, so the length would only be carried to be thrown away.
    values: NonNull<u8>,
    /// The offsets of the elements left to yield.
    offsets: NonNull<u64>,
    /// [`usize::MAX`] while the offsets are flat and `0` once they are scalar, to fold every
    /// position onto the one slot scalar offsets hold without branching on which they are.
    index_mask: usize,
    /// The number of elements left to yield, over the whole range of a `usize`: a scalar array is
    /// as long as it says it is, and is never walked to find out.
    remaining: usize,
    _lifetime: PhantomData<&'a [u8]>,
}

const _: () = {
    // Four words, down from the six two slices and a range take.
    assert!(size_of::<PlBinaryValuesIter<'static>>() == 4 * size_of::<usize>());
    // The niche of the pointers keeps `Option` free.
    assert!(size_of::<Option<PlBinaryValuesIter<'static>>>() == 4 * size_of::<usize>());
};

// SAFETY: the iterator holds nothing but the shared borrows of bytes and offsets it was built
// from, which are `Send` and `Sync` themselves; the raw pointers only drop their lengths.
unsafe impl Send for PlBinaryValuesIter<'_> {}
unsafe impl Sync for PlBinaryValuesIter<'_> {}

/// What a [`PlBinaryValuesIter`] leaves a loop to walk, once its representation is hoisted out.
pub enum PlBinaryValues<'a> {
    /// The bytes, and one start per element left plus the end of the last to cut them with.
    Flat {
        values: &'a [u8],
        offsets: &'a [u64],
    },
    /// The one element every position yields, and how many positions are left to yield it.
    Scalar { value: &'a [u8], count: usize },
}

impl<'a> PlBinaryValuesIter<'a> {
    /// # Safety
    /// `offsets` must be flat (`length + 1` offsets) or scalar (two offsets) for `length`, per
    /// [`crate::broadcast`], and must be ordered and bounded by the length of `values`.
    #[inline]
    pub(super) fn new(values: &'a [u8], offsets: &'a [u64], length: usize) -> Self {
        // Offsets that hold one start per element are flat, and offsets the caller promises are
        // valid are scalar when they are not. The two coincide for a single element, which either
        // reading yields the same bytes for.
        let scalar = !is_flat_offsets_len(offsets.len(), length);

        debug_assert!(!scalar || offsets.len() == 2, "neither flat nor scalar");
        debug_assert!(offsets.first() <= offsets.last(), "offsets out of order");
        debug_assert!(
            offsets
                .last()
                .is_some_and(|&end| end <= values.len() as u64),
            "offsets out of bounds of the values",
        );

        Self {
            values: NonNull::from(values).cast(),
            offsets: NonNull::from(offsets).cast(),
            // All ones for flat offsets, which leaves every position as it is, and none for scalar
            // ones, which folds every position onto the single range they hold.
            index_mask: (scalar as usize).wrapping_sub(1),
            remaining: length,
            _lifetime: PhantomData,
        }
    }

    /// Whether the offsets hold the one range every element covers, rather than one per element.
    #[inline(always)]
    fn is_scalar(&self) -> bool {
        self.index_mask == 0
    }

    /// How far the offsets walk per element dropped: one offset while they are flat, and nowhere
    /// at all once they are scalar, whose two slots every element reads.
    #[inline(always)]
    fn step(&self) -> usize {
        size_of::<u64>() & self.index_mask
    }

    /// The bytes the element `i` positions on covers.
    ///
    /// # Safety
    /// The offsets must hold a start `i` slots on and the end after it, which they do for every
    /// `i` below the number of elements left, and for every `i` at all once they are scalar.
    #[inline(always)]
    unsafe fn get_unchecked(&self, i: usize) -> &'a [u8] {
        debug_assert!(i < self.remaining || self.is_scalar());

        unsafe {
            // Scalar offsets fold every position onto the one range they hold; flat ones hold the
            // start of the element and the end that follows it.
            let offsets = self.offsets.as_ptr().add(i & self.index_mask);
            let start = offsets.read() as usize;
            let end = offsets.add(1).read() as usize;

            // SAFETY: the offsets are ordered, so the length does not wrap, and both of them are
            // in bounds of the values, so the bytes they cut out are a slice of them.
            slice::from_raw_parts(self.values.as_ptr().add(start), end - start)
        }
    }

    /// Drops the `n` elements at the front, which walks flat offsets along and leaves scalar ones
    /// where they are.
    ///
    /// # Safety
    /// `n` must not exceed the number of elements left.
    #[inline(always)]
    unsafe fn advance(&mut self, n: usize) {
        debug_assert!(n <= self.remaining);

        // Flat offsets are walked `n` slots on, which stays within the buffer holding them, and
        // scalar ones are walked nowhere.
        let step = n.wrapping_mul(self.step());

        // SAFETY: flat offsets hold one slot more than the elements left, so `n` of them is at
        // most one past their end.
        self.offsets = unsafe { self.offsets.byte_add(step) };
        self.remaining -= n;
    }

    /// Exhausts the iterator without walking it, which leaves it yielding nothing from either end.
    #[inline(always)]
    fn exhaust(&mut self) {
        self.remaining = 0;
    }

    /// The elements left to yield, with the representation of the offsets hoisted out of them.
    #[inline]
    pub fn split(self) -> PlBinaryValues<'a> {
        if self.is_scalar() {
            // SAFETY: offsets that are scalar hold the two slots of the one range every element
            // covers, whether or not there is an element left to cover it.
            return PlBinaryValues::Scalar {
                value: unsafe { self.get_unchecked(0) },
                count: self.remaining,
            };
        }

        // Flat offsets hold one start per element left to yield plus the end of the last, so
        // there are `remaining + 1` of them — which does not wrap, since a buffer that long does
        // not fit in memory — and the last of them is read here rather than indexed for, which
        // would leave the loop walking them behind a bounds check it can never fail.
        let offsets_ptr = self.offsets.as_ptr();
        // SAFETY: the last of the offsets is the end of the last element left to yield.
        let end = unsafe { offsets_ptr.add(self.remaining).read() } as usize;

        // SAFETY: the offsets hold one slot more than the elements left to yield.
        let offsets = unsafe { slice::from_raw_parts(offsets_ptr, self.remaining + 1) };
        // SAFETY: the offsets are ordered and in bounds of the values, so the values reach at
        // least as far as the last of them, which is as far as any element left reads.
        let values = unsafe { slice::from_raw_parts(self.values.as_ptr(), end) };

        PlBinaryValues::Flat { values, offsets }
    }
}

impl<'a> PlBinaryValues<'a> {
    /// Folds `f` over the elements, walking flat offsets as the consecutive ranges they are —
    /// carrying the end of each element into the start of the next, one offset read per element —
    /// and scalar ones over the single range they hold.
    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, &'a [u8]) -> B,
    {
        match self {
            Self::Scalar { value, count } => {
                let mut acc = init;
                for _ in 0..count {
                    acc = f(acc, value);
                }
                acc
            },
            Self::Flat { values, offsets } => {
                let Some((&first, ends)) = offsets.split_first() else {
                    return init;
                };

                let mut acc = init;
                let mut start = first as usize;
                for &offset in ends {
                    let end = offset as usize;
                    // SAFETY: the offsets are ordered and in bounds of the values.
                    acc = f(acc, unsafe { values.get_unchecked(start..end) });
                    start = end;
                }
                acc
            },
        }
    }

    /// Folds `f` over the elements from the back, the way [`Self::fold`] does from the front.
    #[inline]
    fn rfold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, &'a [u8]) -> B,
    {
        match self {
            Self::Scalar { value, count } => {
                let mut acc = init;
                for _ in 0..count {
                    acc = f(acc, value);
                }
                acc
            },
            Self::Flat { values, offsets } => {
                let Some((&last, starts)) = offsets.split_last() else {
                    return init;
                };

                let mut acc = init;
                let mut end = last as usize;
                for &offset in starts.iter().rev() {
                    let start = offset as usize;
                    // SAFETY: the offsets are ordered and in bounds of the values.
                    acc = f(acc, unsafe { values.get_unchecked(start..end) });
                    end = start;
                }
                acc
            },
        }
    }
}

impl<'a> Iterator for PlBinaryValuesIter<'a> {
    type Item = &'a [u8];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        // SAFETY: there is an element left, and position zero is the front of either
        // representation, so no position has to be folded onto a slot at all.
        let value = unsafe { self.get_unchecked(0) };
        // SAFETY: the element just read is one of the elements left.
        unsafe { self.advance(1) };

        Some(value)
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if n >= self.remaining {
            self.exhaust();
            return None;
        }

        // SAFETY: `n` elements are left to drop before the one asked for.
        unsafe { self.advance(n) };
        self.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }

    #[inline]
    fn count(self) -> usize {
        self.remaining
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    /// Hoists the representation of the offsets out of the loop, per [`Self::split`].
    #[inline]
    fn fold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        self.split().fold(init, f)
    }
}

impl DoubleEndedIterator for PlBinaryValuesIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let last = self.remaining.checked_sub(1)?;

        // SAFETY: `last` is the position of the element at the back, which is still left.
        let value = unsafe { self.get_unchecked(last) };
        self.remaining = last;

        Some(value)
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        if n >= self.remaining {
            self.exhaust();
            return None;
        }

        // `n` is below the number of elements left, so the position before it does not wrap.
        let last = self.remaining - (n + 1);
        // SAFETY: `last` is the position of an element that is still left.
        let value = unsafe { self.get_unchecked(last) };
        self.remaining = last;

        Some(value)
    }

    /// Hoists the representation of the offsets out of the loop, the way [`Iterator::fold`] does.
    #[inline]
    fn rfold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        self.split().rfold(init, f)
    }
}

impl ExactSizeIterator for PlBinaryValuesIter<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.remaining
    }
}

unsafe impl TrustedLen for PlBinaryValuesIter<'_> {}

/// Iterator over the optional elements of a [`PlBinaryArray`](super::PlBinaryArray).
#[derive(Clone)]
pub struct PlBinaryIter<'a> {
    values: PlBinaryValuesIter<'a>,
    validity: ValidityIter<'a>,
}

impl<'a> PlBinaryIter<'a> {
    /// # Safety
    /// `offsets` must be flat or scalar for `length`, per [`crate::broadcast`], and must be ordered
    /// and bounded by the length of `values`.
    #[inline]
    pub(super) fn new(
        values: &'a [u8],
        offsets: &'a [u64],
        validity: Option<PlBitmapRef<'a>>,
        length: usize,
    ) -> Self {
        assert!(validity.is_none_or(|validity| validity.len() == length));

        Self {
            values: PlBinaryValuesIter::new(values, offsets, length),
            validity: ValidityIter::new(validity),
        }
    }

    /// The values and the mask that says which of them are elements, to walk in one loop.
    #[inline]
    fn split(self) -> (PlBinaryValuesIter<'a>, ValidityFold<'a>) {
        (self.values, self.validity.into_mask())
    }
}

impl<'a> Iterator for PlBinaryIter<'a> {
    type Item = Option<&'a [u8]>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let value = self.values.next()?;
        Some(self.validity.next().then_some(value))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        // The mask is advanced alongside the values, whether or not there is a value left.
        let is_valid = self.validity.nth(n);
        let value = self.values.nth(n)?;
        Some(is_valid.then_some(value))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.values.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.values.count()
    }

    /// Walks to the last element from the back, rather than through every one before it.
    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    /// Hoists the validity mask out of the loop, and the representation of the offsets with it.
    #[inline]
    fn fold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let (values, mask) = self.split();
        // SAFETY: the mask has one bit per element, and the values and the mask are walked in
        // lockstep, so it has a bit for every value left to yield.
        unsafe { mask.fold_values(values, init, f) }
    }
}

impl DoubleEndedIterator for PlBinaryIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let value = self.values.next_back()?;
        Some(self.validity.next_back().then_some(value))
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        // The mask is advanced alongside the values, whether or not there is a value left.
        let is_valid = self.validity.nth_back(n);
        let value = self.values.nth_back(n)?;
        Some(is_valid.then_some(value))
    }

    /// Hoists the validity mask out of the loop, the way [`Iterator::fold`] does.
    #[inline]
    fn rfold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let (values, mask) = self.split();
        // SAFETY: the mask has a bit for every value left to yield, per `Iterator::fold`.
        unsafe { mask.rfold_values(values, init, f) }
    }
}

impl ExactSizeIterator for PlBinaryIter<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.values.len()
    }
}

unsafe impl TrustedLen for PlBinaryIter<'_> {}

#[cfg(test)]
mod tests {

    use crate::PlBinaryArray;
    use crate::iterator_tests::assert_iterates;

    /// The elements of a flat array, which are of different lengths and include an empty one.
    fn elements() -> [&'static [u8]; 3] {
        [b"ab", b"", b"cde"]
    }

    fn flat_array() -> PlBinaryArray {
        PlBinaryArray::from_iter(elements().map(Some))
    }

    #[test]
    fn flat() {
        let array = flat_array();

        assert_iterates(array.values_iter(), &elements());
        assert_iterates(array.iter(), &elements().map(Some));
    }

    #[test]
    fn scalar() {
        let array = PlBinaryArray::new_scalar(b"xy", 4);

        assert_iterates(array.values_iter(), &[b"xy".as_slice(); 4]);
        assert_iterates(array.iter(), &[Some(b"xy".as_slice()); 4]);
    }

    #[test]
    fn empty() {
        let array = PlBinaryArray::new_empty();

        assert_iterates(array.values_iter(), &[]);
        assert_iterates(array.iter(), &[]);
        // An array of no elements keeps no slot of the value a scalar one repeats.
        assert_iterates(PlBinaryArray::new_scalar(b"xy", 0).values_iter(), &[]);
    }
}
