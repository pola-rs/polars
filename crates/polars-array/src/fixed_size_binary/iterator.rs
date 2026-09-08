use arrow::trusted_len::TrustedLen;

use crate::bitmap::{PlBitmapRef, ValidityFold, ValidityIter};
use crate::broadcast::is_flat_fixed_size_values_len;

/// Iterator over the elements of a [`super::PlFixedSizeBinaryArray`], ignoring validity.
#[derive(Clone)]
pub struct PlFixedSizeBinaryValuesIter<'a> {
    /// The values from the element at the front on.
    front: &'a [u8],
    /// How many bytes every element is wide.
    width: usize,
    /// How far the front walks per element: one width for flat values, nowhere for scalar ones.
    stride: usize,
    /// How many elements are left to yield: a scalar array is as long as it says it is.
    remaining: usize,
}

impl<'a> PlFixedSizeBinaryValuesIter<'a> {
    /// # Safety
    /// `values` must be flat or scalar for `length`, per [`crate::broadcast`].
    #[inline]
    pub(super) fn new(values: &'a [u8], width: usize, length: usize) -> Self {
        // Values as long as one element hold the one every position reads; values the caller
        // promises are valid hold one element each when they are not. The two coincide for a
        // single element, and for elements no bytes wide — which the walk steps nowhere for
        // either way — so the scalar stride stands for both.
        let scalar = values.len() == width;

        debug_assert!(
            scalar || is_flat_fixed_size_values_len(values.len(), width, length),
            "neither flat nor scalar",
        );

        Self {
            front: values,
            width,
            stride: if scalar { 0 } else { width },
            remaining: length,
        }
    }

    /// The bytes of the element `n` strides on from the front.
    ///
    /// # Safety
    /// The values must reach `width` bytes on from that element, as they do for every one left.
    #[inline(always)]
    unsafe fn at(&self, n: usize) -> &'a [u8] {
        let start = n.wrapping_mul(self.stride);
        debug_assert!(start + self.width <= self.front.len());
        // SAFETY: the element is in bounds of the values, per the caller.
        unsafe { self.front.get_unchecked(start..start + self.width) }
    }

    /// Drops the front `n` elements.
    ///
    /// # Safety
    /// `n` must not exceed the number of elements left, so that the front lands in the values.
    #[inline(always)]
    unsafe fn advance(&mut self, n: usize) {
        debug_assert!(n <= self.remaining);
        let start = n.wrapping_mul(self.stride);
        // SAFETY: the values reach the front of every element left, and one width past the last
        // of them, so `start` is in bounds of them or one past their end.
        self.front = unsafe { self.front.get_unchecked(start..) };
        self.remaining -= n;
    }

    /// Exhausts the walk, which leaves it yielding nothing from either end.
    #[inline(always)]
    fn exhaust(&mut self) {
        self.remaining = 0;
    }
}

impl<'a> Iterator for PlFixedSizeBinaryValuesIter<'a> {
    type Item = &'a [u8];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        // SAFETY: an element is left, so the values reach a width on from the front, and the
        // front of the element after it is at most one width past their end.
        let value = unsafe { self.at(0) };
        unsafe { self.advance(1) };

        Some(value)
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if n >= self.remaining {
            self.exhaust();
            return None;
        }

        // SAFETY: the `n` elements dropped are the front `n` of the ones left, so the front
        // stays at an element the values hold.
        unsafe { self.advance(n) };
        self.next()
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

    /// Walks to the last element from the back, rather than through every one before it.
    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    /// Hoists the representation of the values out of the loop, leaving a plain walk of a stride.
    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let Self {
            mut front,
            width,
            stride,
            remaining,
        } = self;
        let mut acc = init;

        for _ in 0..remaining {
            // SAFETY: the values hold every element of the walk, a width each.
            let value = unsafe { front.get_unchecked(..width) };
            // SAFETY: the front of the element after the last one is one width past their end,
            // which is in bounds of a slice to take.
            front = unsafe { front.get_unchecked(stride..) };
            acc = f(acc, value);
        }

        acc
    }
}

impl DoubleEndedIterator for PlFixedSizeBinaryValuesIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let last = self.remaining.checked_sub(1)?;

        // SAFETY: the element at the back is the last one the values hold.
        let value = unsafe { self.at(last) };
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
        self.remaining -= n;
        self.next_back()
    }

    /// Hoists the representation of the values out of the loop, the way [`Iterator::fold`] does.
    #[inline]
    fn rfold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let Self {
            front,
            width,
            stride,
            remaining,
        } = self;
        // One element past the back, which the first step of the walk comes back down from.
        let mut start = remaining.wrapping_mul(stride);
        let mut acc = init;

        for _ in 0..remaining {
            start = start.wrapping_sub(stride);
            // SAFETY: the values hold every element of the walk, a width each.
            let value = unsafe { front.get_unchecked(start..start + width) };
            acc = f(acc, value);
        }

        acc
    }
}

impl ExactSizeIterator for PlFixedSizeBinaryValuesIter<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.remaining
    }
}

unsafe impl TrustedLen for PlFixedSizeBinaryValuesIter<'_> {}

/// Iterator over the optional elements of a [`super::PlFixedSizeBinaryArray`].
#[derive(Clone)]
pub struct PlFixedSizeBinaryIter<'a> {
    values: PlFixedSizeBinaryValuesIter<'a>,
    validity: ValidityIter<'a>,
}

impl<'a> PlFixedSizeBinaryIter<'a> {
    /// # Safety
    /// `values` must be flat or scalar for `length` and `width`, per [`crate::broadcast`].
    #[inline]
    pub(super) fn new(
        values: &'a [u8],
        width: usize,
        validity: Option<PlBitmapRef<'a>>,
        length: usize,
    ) -> Self {
        assert!(validity.is_none_or(|validity| validity.len() == length));

        Self {
            values: PlFixedSizeBinaryValuesIter::new(values, width, length),
            validity: ValidityIter::new(validity),
        }
    }

    /// The values and the mask that says which of them are elements, to walk in one loop.
    #[inline]
    fn split(self) -> (PlFixedSizeBinaryValuesIter<'a>, ValidityFold<'a>) {
        (self.values, self.validity.into_mask())
    }
}

crate::impl_optional_iter!(PlFixedSizeBinaryIter<'a>, &'a [u8]);

#[cfg(test)]
mod tests {

    use crate::PlFixedSizeBinaryArray;
    use crate::iterator_tests::assert_iterates;

    /// The elements of a flat array of three elements two bytes wide.
    fn elements() -> [&'static [u8]; 3] {
        [b"ab", b"cd", b"ef"]
    }

    fn flat_array() -> PlFixedSizeBinaryArray {
        PlFixedSizeBinaryArray::from_vec(b"abcdef".to_vec(), 2)
    }

    #[test]
    fn flat() {
        let array = flat_array();

        assert_iterates(array.values_iter(), &elements());
        assert_iterates(array.iter(), &elements().map(Some));
    }

    #[test]
    fn scalar() {
        let array = PlFixedSizeBinaryArray::new_scalar(b"xy", 4);

        assert_iterates(array.values_iter(), &[b"xy".as_slice(); 4]);
        assert_iterates(array.iter(), &[Some(b"xy".as_slice()); 4]);
    }

    #[test]
    fn a_broadcast_array_is_not_materialized() {
        // Walking a billion elements would not finish; the scalar path must hit.
        let array = PlFixedSizeBinaryArray::new_scalar(b"xy", 1_000_000_000);

        assert_eq!(array.values_iter().count(), 1_000_000_000);
        assert_eq!(array.values_iter().nth(999_999_999), Some(b"xy".as_slice()));
        assert_eq!(
            array.values_iter().nth_back(999_999_999),
            Some(b"xy".as_slice())
        );
        assert_eq!(array.iter().last(), Some(Some(b"xy".as_slice())));
        assert_eq!(
            array.iter().nth_back(999_999_999),
            Some(Some(b"xy".as_slice()))
        );
    }
}
