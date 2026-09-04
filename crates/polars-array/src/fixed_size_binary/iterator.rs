use arrow::trusted_len::TrustedLen;

use crate::bitmap::{PlBitmapRef, ValidityFold, ValidityIter};
use crate::broadcast::is_flat_fixed_size_values_len;

/// Iterator over the elements of a [`PlFixedSizeBinaryArray`](super::PlFixedSizeBinaryArray),
/// ignoring validity.
///
/// The representation of the values is resolved once, at construction, into the stride of the
/// walk: one width while they hold an element per position, and nowhere at all once they hold the
/// one element every position reads. Every element is then the same handful of instructions
/// whatever the values turned out to be — no branch on the representation, nothing in the loop for
/// the compiler to unswitch, and a trip count it can see, which is what lets it unroll the
/// caller's loop.
///
/// Deciding per element instead — matching an enum, or loading the length of the values and
/// selecting on it — leaves a diamond in the loop body and a stride the loop cannot hoist.
#[derive(Clone)]
pub struct PlFixedSizeBinaryValuesIter<'a> {
    /// The values from the element at the front on.
    ///
    /// Only the first [`Self::width`] bytes are ever read through it; the rest of its length is
    /// carried along so the walk stays in safe slice arithmetic.
    front: &'a [u8],
    /// How many bytes every element is wide.
    width: usize,
    /// How far the front walks per element: one width for flat values, and nowhere for scalar
    /// ones, which every position reads the same bytes of.
    stride: usize,
    /// How many elements are left to yield, over the whole range of a `usize`: a scalar array is
    /// as long as it says it is, and is never walked to find out.
    remaining: usize,
}

impl<'a> PlFixedSizeBinaryValuesIter<'a> {
    /// # Safety
    /// `values` must be flat (`length * width` bytes) or scalar (`width` bytes) for `length`, per
    /// [`crate::broadcast`].
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
    /// The values must reach `width` bytes on from that element, which they do for every element
    /// this iterator has left to yield.
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
    /// `n` must not exceed the number of elements left, so that the front lands on an element the
    /// values hold — or, for `n` elements exactly, one width past the last of them.
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

    /// Hoists the representation of the values out of the loop, leaving it the plain walk of a
    /// stride it is — which is what `collect` and `for_each` route through.
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

/// Iterator over the optional elements of a
/// [`PlFixedSizeBinaryArray`](super::PlFixedSizeBinaryArray).
#[derive(Clone)]
pub struct PlFixedSizeBinaryIter<'a> {
    values: PlFixedSizeBinaryValuesIter<'a>,
    validity: ValidityIter<'a>,
}

impl<'a> PlFixedSizeBinaryIter<'a> {
    /// # Panics
    /// Panics unless `validity` has `length` bits.
    ///
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

    /// The bytes of the element the values just yielded, `None` where the mask says it is null.
    ///
    /// # Safety
    /// The mask must still cover the element the values yielded, which it does for one they
    /// yielded at the front — the two are walked in lockstep, and the mask holds a bit for every
    /// element the values do.
    #[inline(always)]
    unsafe fn front(&mut self, value: &'a [u8], n: usize) -> Option<&'a [u8]> {
        // SAFETY: the mask covers the element the values yielded, per the caller.
        unsafe { self.validity.nth_unchecked(n) }.then_some(value)
    }

    /// The bytes of the element the values just yielded at the back, `None` where the mask says
    /// it is null.
    ///
    /// # Safety
    /// The mask must still cover the element the values yielded at the back.
    #[inline(always)]
    unsafe fn back(&mut self, value: &'a [u8], n: usize) -> Option<&'a [u8]> {
        // SAFETY: the mask covers the element the values yielded, per the caller.
        unsafe { self.validity.nth_back_unchecked(n) }.then_some(value)
    }

    /// The values and the mask that says which of them are elements, to walk in one loop.
    #[inline]
    fn split(self) -> (PlFixedSizeBinaryValuesIter<'a>, ValidityFold<'a>) {
        (self.values, self.validity.into_mask())
    }
}

impl<'a> Iterator for PlFixedSizeBinaryIter<'a> {
    type Item = Option<&'a [u8]>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let value = self.values.next()?;
        // SAFETY: the values yielded the element at the front, so the mask still covers it.
        Some(unsafe { self.front(value, 0) })
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        // The values are asked first, so that the mask is only read where one of them was there
        // to read it for; walking past the end leaves the mask covering nothing, the way walking
        // it to its end would.
        let Some(value) = self.values.nth(n) else {
            self.validity.exhaust();
            return None;
        };

        // SAFETY: the values yielded the element `n` positions on, so the mask still covers it.
        Some(unsafe { self.front(value, n) })
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

    /// Hoists the validity mask out of the loop, and the representation of the values with it.
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

impl DoubleEndedIterator for PlFixedSizeBinaryIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let value = self.values.next_back()?;
        // SAFETY: the values yielded the element at the back, so the mask still covers it.
        Some(unsafe { self.back(value, 0) })
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        // The values are asked first, the way `Iterator::nth` does.
        let Some(value) = self.values.nth_back(n) else {
            self.validity.exhaust();
            return None;
        };

        // SAFETY: the values yielded the element `n` positions in, so the mask still covers it.
        Some(unsafe { self.back(value, n) })
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

impl ExactSizeIterator for PlFixedSizeBinaryIter<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.values.len()
    }
}

unsafe impl TrustedLen for PlFixedSizeBinaryIter<'_> {}

#[cfg(test)]
mod tests {

    use arrow::bitmap::Bitmap;
    use polars_buffer::Buffer;

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

    /// A mask of mixed bits, which is read by position alongside the elements.
    #[test]
    fn mixed_validity() {
        let array = flat_array().with_validity(Some(Bitmap::from_iter([true, false, true])));

        assert_iterates(array.values_iter(), &elements());
        assert_iterates(
            array.iter(),
            &[Some(elements()[0]), None, Some(elements()[2])],
        );
    }

    /// A mask that starts partway into its bytes, walked in lockstep with values that start
    /// partway into theirs — which is what the elements are read out of step with if either end
    /// of the walk drops a position the other one keeps.
    #[test]
    fn sliced_mixed_validity() {
        let array = PlFixedSizeBinaryArray::from_vec(b"abcdefghij".to_vec(), 2)
            .with_validity(Some(Bitmap::from_iter([true, false, true, true, false])))
            .sliced(1, 3);

        assert_iterates(array.values_iter(), &[b"cd", b"ef", b"gh"]);
        assert_iterates(
            array.iter(),
            &[None, Some(b"ef".as_slice()), Some(b"gh".as_slice())],
        );
    }

    /// A mask under a scalar array, which every position reads the same bit of.
    #[test]
    fn scalar_values_under_a_mixed_mask() {
        let array = PlFixedSizeBinaryArray::new_scalar(b"xy", 3)
            .with_validity(Some(Bitmap::from_iter([true, false, true])));

        assert_iterates(
            array.iter(),
            &[Some(b"xy".as_slice()), None, Some(b"xy".as_slice())],
        );
    }

    /// The elements of a sliced array start partway into the values, which are still cut into the
    /// chunks the width makes of them.
    #[test]
    fn sliced() {
        let array = flat_array().sliced(1, 2);

        assert_iterates(array.values_iter(), &elements()[1..]);
        assert_iterates(
            array.iter(),
            &elements()[1..]
                .iter()
                .copied()
                .map(Some)
                .collect::<Vec<_>>(),
        );
    }

    /// An array of no elements, which keeps no element for a scalar one to repeat.
    #[test]
    fn empty() {
        assert_iterates(flat_array().sliced(0, 0).values_iter(), &[]);
        assert_iterates(
            PlFixedSizeBinaryArray::new_scalar(b"xy", 0).values_iter(),
            &[],
        );
    }

    /// Elements no bytes wide, which are all the same empty slice however many there are — and
    /// which the chunks of a flat walk would have no end of.
    #[test]
    fn a_width_of_zero() {
        let empty: &[u8] = b"";

        let flat = PlFixedSizeBinaryArray::new(Buffer::new(), 0, 3, None);
        assert_iterates(flat.values_iter(), &[empty; 3]);
        assert_iterates(flat.iter(), &[Some(empty); 3]);

        let scalar = PlFixedSizeBinaryArray::new_scalar(empty, 3);
        assert_iterates(scalar.values_iter(), &[empty; 3]);
        assert_iterates(scalar.iter(), &[Some(empty); 3]);
    }
}
