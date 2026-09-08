use arrow::trusted_len::TrustedLen;

use crate::array::PlArray;
use crate::bitmap::{PlBitmapRef, ValidityFold, ValidityIter};
use crate::broadcast::is_valid_fixed_size_values_len;

/// The offsets into the values at which the elements left to yield start.
#[derive(Clone)]
struct Offsets {
    /// Where the element at the front starts.
    front: usize,
    /// How far that start walks per element dropped.
    stride: usize,
    /// The number of elements left to yield: a scalar array is as long as it says it is.
    remaining: usize,
}

impl Offsets {
    /// # Safety
    /// The values must be flat or scalar for `length`, per [`crate::broadcast`].
    #[inline]
    fn new(values_len: usize, width: usize, length: usize) -> Self {
        // Values as long as one element hold the one every position reads; values the caller
        // promises are valid hold one element each when they are not. The two coincide for a
        // single element, and for elements no values wide, either of which the same range is cut
        // out for.
        let scalar = values_len == width;

        debug_assert!(
            is_valid_fixed_size_values_len(values_len, width, length),
            "neither flat nor scalar",
        );

        Self {
            front: 0,
            // Flat values lay the elements end to end, one width apart; scalar ones hold the one
            // range every element reads, which the walk never steps off.
            stride: if scalar { 0 } else { width },
            remaining: length,
        }
    }

    /// Where the element `n` positions on from the front starts.
    ///
    /// # Safety
    /// `n` must not exceed the number of elements left, so the offset stays within the values.
    #[inline(always)]
    fn at(&self, n: usize) -> usize {
        debug_assert!(n <= self.remaining);
        // Flat values reach `remaining * width` on from the front, so neither the product nor the
        // sum wraps; scalar ones are walked nowhere, whatever `n` is.
        self.front.wrapping_add(n.wrapping_mul(self.stride))
    }

    /// Exhausts the walk, which leaves it yielding nothing from either end.
    #[inline(always)]
    fn exhaust(&mut self) {
        self.remaining = 0;
    }
}

impl Iterator for Offsets {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.remaining == 0 {
            return None;
        }

        let front = self.front;
        // The front walks one element on, which for the last of them lands one width past their
        // end — an offset nothing reads.
        self.front = self.at(1);
        self.remaining -= 1;

        Some(front)
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<usize> {
        if n >= self.remaining {
            self.exhaust();
            return None;
        }

        // The `n` elements dropped before the one asked for are the front `n` of the ones left,
        // so the front stays in bounds of the values.
        self.front = self.at(n);
        self.remaining -= n;
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

    /// Walks the offsets as the affine sequence they are, rather than one `Option` at a time.
    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, usize) -> B,
    {
        let stride = self.stride;
        let mut front = self.front;
        let mut acc = init;

        for _ in 0..self.remaining {
            acc = f(acc, front);
            front = front.wrapping_add(stride);
        }

        acc
    }
}

impl DoubleEndedIterator for Offsets {
    #[inline]
    fn next_back(&mut self) -> Option<usize> {
        let last = self.remaining.checked_sub(1)?;

        let back = self.at(last);
        self.remaining = last;

        Some(back)
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<usize> {
        if n >= self.remaining {
            self.exhaust();
            return None;
        }

        // `n` is below the number of elements left, so the position before it does not wrap.
        self.remaining -= n;
        self.next_back()
    }

    /// Walks the offsets from the back, the way [`Iterator::fold`] does from the front.
    #[inline]
    fn rfold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, usize) -> B,
    {
        let stride = self.stride;
        // One element past the back, which the first step of the walk comes back down from.
        let mut back = self.at(self.remaining);
        let mut acc = init;

        for _ in 0..self.remaining {
            back = back.wrapping_sub(stride);
            acc = f(acc, back);
        }

        acc
    }
}

impl ExactSizeIterator for Offsets {
    #[inline]
    fn len(&self) -> usize {
        self.remaining
    }
}

unsafe impl TrustedLen for Offsets {}

/// The element the values hold at `offset`, which is a fresh box over the same buffers.
///
/// # Safety
/// The values must reach `width` on from `offset`, as they do for every offset [`Offsets`] yields.
#[inline(always)]
unsafe fn element(values: &dyn PlArray, width: usize, offset: usize) -> Box<dyn PlArray> {
    debug_assert!(offset + width <= values.len());
    // SAFETY: the element is in bounds of the values, per the caller.
    unsafe { values.sliced_unchecked(offset, width) }
}

/// Iterator over the elements of a [`super::PlFixedSizeListArray`], ignoring validity.
#[derive(Clone)]
pub struct PlFixedSizeListValuesIter<'a> {
    /// The values array the elements are cut out of.
    values: &'a dyn PlArray,
    /// How many values every element covers.
    width: usize,
    /// Where the elements left to yield start.
    offsets: Offsets,
}

impl<'a> PlFixedSizeListValuesIter<'a> {
    /// # Safety
    /// `values` must be flat or scalar for `length`, per [`crate::broadcast`].
    #[inline]
    pub(super) fn new(values: &'a dyn PlArray, width: usize, length: usize) -> Self {
        Self {
            values,
            width,
            // SAFETY: the values are flat or scalar for `length`, per the caller.
            offsets: Offsets::new(values.len(), width, length),
        }
    }
}

crate::impl_mapped_iter!(
    PlFixedSizeListValuesIter<'a>,
    Box<dyn PlArray>,
    over: offsets,
    with: [values, width],
    // SAFETY: the offset is the front of one of this iterator's elements.
    |offset| unsafe { element(values, width, offset) },
);

/// Iterator over the optional elements of a [`PlFixedSizeListArray`](super::PlFixedSizeListArray).
#[derive(Clone)]
pub struct PlFixedSizeListIter<'a> {
    values: &'a dyn PlArray,
    width: usize,
    offsets: Offsets,
    validity: ValidityIter<'a>,
}

impl<'a> PlFixedSizeListIter<'a> {
    /// # Safety
    /// `values` must be flat or scalar for `length` and `width`, per [`crate::broadcast`].
    #[inline]
    pub(super) fn new(
        values: &'a dyn PlArray,
        width: usize,
        validity: Option<PlBitmapRef<'a>>,
        length: usize,
    ) -> Self {
        assert!(validity.is_none_or(|validity| validity.len() == length));

        Self {
            values,
            width,
            // SAFETY: the values are flat or scalar for `length`, per the caller.
            offsets: Offsets::new(values.len(), width, length),
            validity: ValidityIter::new(validity),
        }
    }

    /// The element at `offset` if `is_valid`, built only where it is.
    ///
    /// # Safety
    /// `offset` must be one the iterator's own offsets yielded.
    #[inline(always)]
    unsafe fn get(&self, is_valid: bool, offset: usize) -> Option<Box<dyn PlArray>> {
        // SAFETY: the offset is the front of one of this iterator's elements.
        is_valid.then(|| unsafe { element(self.values, self.width, offset) })
    }

    /// The offsets of the elements left to yield and the mask that says which of them are elements.
    #[inline]
    fn split(self) -> (&'a dyn PlArray, usize, Offsets, ValidityFold<'a>) {
        (
            self.values,
            self.width,
            self.offsets,
            self.validity.into_mask(),
        )
    }
}

/// The iterator traits are written out rather than taken from `impl_optional_iter`, which reads
/// the mask only where the values yielded an element: an element here is an array of its own,
/// so the mask is read first and the walk over the offsets yields nothing but a shape, which
/// leaves a null position paying for no array at all.
impl Iterator for PlFixedSizeListIter<'_> {
    type Item = Option<Box<dyn PlArray>>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let offset = self.offsets.next()?;
        let is_valid = self.validity.next();
        // SAFETY: the offset is the front of one of this iterator's elements.
        Some(unsafe { self.get(is_valid, offset) })
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        // The mask is advanced alongside the offsets, whether or not there is an element left.
        let is_valid = self.validity.nth(n);
        let offset = self.offsets.nth(n)?;
        // SAFETY: the offset is the front of one of this iterator's elements.
        Some(unsafe { self.get(is_valid, offset) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.offsets.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.offsets.count()
    }

    /// Walks to the last element from the back, rather than through every one before it.
    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    /// Hoists the validity mask out of the loop, and the walk over the offsets with it.
    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let (values, width, offsets, mask) = self.split();
        // The element is built only where the mask says there is one, so a null position pays for
        // no box at all.
        let element = |offset: Option<usize>| {
            // SAFETY: the offset is the front of one of this iterator's elements.
            offset.map(|offset| unsafe { element(values, width, offset) })
        };

        // SAFETY: the mask has one bit per element, and the offsets and the mask are walked in
        // lockstep, so it has a bit for every offset left to yield.
        unsafe { mask.fold_values(offsets, init, |acc, offset| f(acc, element(offset))) }
    }
}

impl DoubleEndedIterator for PlFixedSizeListIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let offset = self.offsets.next_back()?;
        let is_valid = self.validity.next_back();
        // SAFETY: the offset is the front of one of this iterator's elements.
        Some(unsafe { self.get(is_valid, offset) })
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        // The mask is advanced alongside the offsets, whether or not there is an element left.
        let is_valid = self.validity.nth_back(n);
        let offset = self.offsets.nth_back(n)?;
        // SAFETY: the offset is the front of one of this iterator's elements.
        Some(unsafe { self.get(is_valid, offset) })
    }

    /// Hoists the validity mask out of the loop, the way [`Iterator::fold`] does.
    #[inline]
    fn rfold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let (values, width, offsets, mask) = self.split();
        // The element is built only where the mask says there is one, per [`Iterator::fold`].
        let element = |offset: Option<usize>| {
            // SAFETY: the offset is the front of one of this iterator's elements.
            offset.map(|offset| unsafe { element(values, width, offset) })
        };

        // SAFETY: the mask has a bit for every offset left to yield, per `Iterator::fold`.
        unsafe { mask.rfold_values(offsets, init, |acc, offset| f(acc, element(offset))) }
    }
}

impl ExactSizeIterator for PlFixedSizeListIter<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.offsets.len()
    }
}

unsafe impl TrustedLen for PlFixedSizeListIter<'_> {}

#[cfg(test)]
mod tests {

    use crate::iterator_tests::assert_iterates;
    use crate::{PlArray, PlFixedSizeListArray, PlPrimitiveArray};

    /// The list `values` are, as an element of a fixed size list array is.
    fn element(values: &[i32]) -> Box<dyn PlArray> {
        Box::new(PlPrimitiveArray::from_vec(values.to_vec()))
    }

    /// A flat array of the lists `[1, 2]`, `[3, 4]` and `[5, 6]`.
    fn flat_array() -> PlFixedSizeListArray {
        PlFixedSizeListArray::new(element(&[1, 2, 3, 4, 5, 6]), 2, 3, None)
    }

    fn elements() -> [Box<dyn PlArray>; 3] {
        [element(&[1, 2]), element(&[3, 4]), element(&[5, 6])]
    }

    #[test]
    fn flat() {
        let array = flat_array();

        assert_iterates(array.values_iter(), &elements());
        assert_iterates(array.iter(), &elements().map(Some));
    }

    #[test]
    fn scalar() {
        let array = PlFixedSizeListArray::new_scalar(element(&[1, 2]), 4);
        let expected = [(); 4].map(|()| element(&[1, 2]));

        assert_iterates(array.values_iter(), &expected);
        assert_iterates(array.iter(), &expected.map(Some));
    }

    /// The elements of a sliced array start partway into the values, cut out by the width.
    #[test]
    fn sliced() {
        let array = flat_array().sliced(1, 2);

        assert_iterates(array.values_iter(), &elements()[1..]);
        assert_iterates(
            array.iter(),
            &elements()[1..]
                .iter()
                .cloned()
                .map(Some)
                .collect::<Vec<_>>(),
        );
    }
}
