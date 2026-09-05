use arrow::trusted_len::TrustedLen;

use crate::array::PlArray;
use crate::bitmap::{PlBitmapRef, ValidityFold, ValidityIter};
use crate::broadcast::is_valid_fixed_size_values_len;

/// The offsets into the values at which the elements left to yield start.
///
/// The elements of a fixed size list array are a width apart, so their offsets are an affine
/// sequence, and the representation of the values sets nothing but its stride: one width while
/// they hold an element per position, and nowhere at all once they hold the one element every
/// position reads.
///
/// Resolving that stride once, at construction, is what leaves the walk a plain add. Deciding per
/// element instead leaves every read behind a *virtual* call for the length of the values and a
/// select on it — a call the loop cannot hoist, cannot fold against what the caller has asserted
/// about the array, and cannot see past.
#[derive(Clone)]
struct Offsets {
    /// Where the element at the front starts.
    front: usize,
    /// How far that start walks per element dropped.
    stride: usize,
    /// The number of elements left to yield, over the whole range of a `usize`: a scalar array is
    /// as long as it says it is, and is never walked to find out.
    remaining: usize,
}

impl Offsets {
    /// # Safety
    /// The values must be flat (`length * width` values) or scalar (`width` values) for `length`,
    /// per [`crate::broadcast`].
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
    /// `n` must not exceed the number of elements left, so that the offset stays in bounds of the
    /// values — or, for `n` elements exactly, one width past the last of them, which nothing reads.
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

    /// Walks the offsets as the affine sequence they are, rather than stepping one `Option` at a
    /// time — which is what `collect` and `for_each` route through.
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
/// The values must reach `width` on from `offset`, which they do for the offset of every element
/// an [`Offsets`] built for them yields.
#[inline(always)]
unsafe fn element(values: &dyn PlArray, width: usize, offset: usize) -> Box<dyn PlArray> {
    debug_assert!(offset + width <= values.len());
    // SAFETY: the element is in bounds of the values, per the caller.
    unsafe { values.sliced_unchecked(offset, width) }
}

/// Iterator over the elements of a [`PlFixedSizeListArray`](super::PlFixedSizeListArray), ignoring
/// validity.
///
/// The representation of the values is resolved once, at construction, into the stride of the
/// [`Offsets`] the walk steps through; see there.
#[derive(Clone)]
pub struct PlFixedSizeListValuesIter<'a> {
    /// The values array the elements are cut out of.
    ///
    /// Held as the borrow it is, rather than reached for through the array, so that the walk keeps
    /// it in a register instead of loading the box out of the array on every element.
    values: &'a dyn PlArray,
    /// How many values every element covers.
    width: usize,
    /// Where the elements left to yield start.
    offsets: Offsets,
}

impl<'a> PlFixedSizeListValuesIter<'a> {
    /// # Safety
    /// `values` must be flat (`length * width` values) or scalar (`width` values) for `length`,
    /// per [`crate::broadcast`].
    #[inline]
    pub(super) fn new(values: &'a dyn PlArray, width: usize, length: usize) -> Self {
        Self {
            values,
            width,
            // SAFETY: the values are flat or scalar for `length`, per the caller.
            offsets: Offsets::new(values.len(), width, length),
        }
    }

    /// The element at `offset`.
    ///
    /// # Safety
    /// `offset` must be one the iterator's own offsets yielded.
    #[inline(always)]
    unsafe fn get(&self, offset: usize) -> Box<dyn PlArray> {
        // SAFETY: the offset is the front of one of this iterator's elements.
        unsafe { element(self.values, self.width, offset) }
    }
}

impl Iterator for PlFixedSizeListValuesIter<'_> {
    type Item = Box<dyn PlArray>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let offset = self.offsets.next()?;
        // SAFETY: the offset is the front of one of this iterator's elements.
        Some(unsafe { self.get(offset) })
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let offset = self.offsets.nth(n)?;
        // SAFETY: the offset is the front of one of this iterator's elements.
        Some(unsafe { self.get(offset) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.offsets.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.offsets.count()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    /// Hoists the walk over the offsets out of the loop, per [`Offsets::fold`].
    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let (values, width) = (self.values, self.width);

        self.offsets.fold(init, |acc, offset| {
            // SAFETY: the offset is the front of one of this iterator's elements.
            f(acc, unsafe { element(values, width, offset) })
        })
    }
}

impl DoubleEndedIterator for PlFixedSizeListValuesIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let offset = self.offsets.next_back()?;
        // SAFETY: the offset is the front of one of this iterator's elements.
        Some(unsafe { self.get(offset) })
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let offset = self.offsets.nth_back(n)?;
        // SAFETY: the offset is the front of one of this iterator's elements.
        Some(unsafe { self.get(offset) })
    }

    /// Hoists the walk over the offsets out of the loop, the way [`Iterator::fold`] does.
    #[inline]
    fn rfold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let (values, width) = (self.values, self.width);

        self.offsets.rfold(init, |acc, offset| {
            // SAFETY: the offset is the front of one of this iterator's elements.
            f(acc, unsafe { element(values, width, offset) })
        })
    }
}

impl ExactSizeIterator for PlFixedSizeListValuesIter<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.offsets.len()
    }
}

unsafe impl TrustedLen for PlFixedSizeListValuesIter<'_> {}

/// Iterator over the optional elements of a
/// [`PlFixedSizeListArray`](super::PlFixedSizeListArray).
///
/// The mask gates the element rather than the other way around: an element of this array is a
/// fresh box over the values, so building one for a null position — only to throw it away — costs
/// an allocation and a free that reading the bit first does not pay at all.
#[derive(Clone)]
pub struct PlFixedSizeListIter<'a> {
    values: &'a dyn PlArray,
    width: usize,
    offsets: Offsets,
    validity: ValidityIter<'a>,
}

impl<'a> PlFixedSizeListIter<'a> {
    /// # Panics
    /// Panics unless `validity` has `length` bits.
    ///
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

    /// The offsets of the elements left to yield and the mask that says which of them are
    /// elements, to walk in one loop.
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

    /// Hoists the validity mask out of the loop, and the walk over the offsets with it. An array
    /// whose elements are all null never reaches the values at all.
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
    use arrow::bitmap::Bitmap;

    use crate::bitmap::PlBitmap;
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

    /// The elements of a sliced array start partway into the values, which the width still cuts
    /// them out of from their own front.
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

    /// An array of no elements, which keeps no element for a scalar one to repeat.
    #[test]
    fn empty() {
        assert_iterates(flat_array().sliced(0, 0).values_iter(), &[]);
        assert_iterates(flat_array().sliced(0, 0).iter(), &[]);
        assert_iterates(
            PlFixedSizeListArray::new_scalar(element(&[1, 2]), 0).values_iter(),
            &[],
        );
    }

    /// Elements no values wide, which are all the same empty list however many there are — and
    /// which a flat walk would step over without ever leaving the front.
    #[test]
    fn a_width_of_zero() {
        let flat = PlFixedSizeListArray::new(element(&[]), 0, 3, None);
        assert_iterates(flat.values_iter(), &[(); 3].map(|()| element(&[])));
        assert_iterates(flat.iter(), &[(); 3].map(|()| Some(element(&[]))));

        let scalar = PlFixedSizeListArray::new_scalar(element(&[]), 3);
        assert_iterates(scalar.values_iter(), &[(); 3].map(|()| element(&[])));
        assert_iterates(scalar.iter(), &[(); 3].map(|()| Some(element(&[]))));
    }

    /// A mask of mixed bits, which is read by position alongside the elements the width cuts out.
    #[test]
    fn mixed_validity() {
        let array = flat_array().with_validity(Some(PlBitmap::from_bitmap(Bitmap::from_iter([
            true, false, true,
        ]))));

        assert_iterates(array.values_iter(), &elements());
        assert_iterates(
            array.iter(),
            &[
                Some(elements()[0].clone()),
                None,
                Some(elements()[2].clone()),
            ],
        );
    }

    /// An array whose elements are all null, which the walk never reaches the values for.
    #[test]
    fn all_null() {
        let array = flat_array().with_validity(Some(PlBitmap::from_bitmap(Bitmap::new_zeroed(3))));

        assert_iterates(array.values_iter(), &elements());
        assert_iterates(array.iter(), &[None, None, None]);
    }

    /// A mask that broadcasts its single bit, which every element reads.
    #[test]
    fn scalar_validity() {
        let element = element(&[1, 2]);
        let null = PlFixedSizeListArray::new_full_null(element.clone(), 3);

        assert_iterates(null.iter(), &[None, None, None]);
        assert_eq!(null.iter().nth(2), Some(None));
        assert_eq!(null.iter().nth_back(2), Some(None));
        assert_eq!(null.iter().last(), Some(None));
    }

    #[test]
    fn a_broadcast_array_is_not_materialized() {
        // Walking a billion elements would not finish; the scalar path must hit.
        let array = PlFixedSizeListArray::new_scalar(element(&[1, 2]), 1_000_000_000);

        assert_eq!(array.values_iter().count(), 1_000_000_000);
        assert_eq!(array.values_iter().nth(999_999_999), Some(element(&[1, 2])));
        assert_eq!(
            array.values_iter().nth_back(999_999_999),
            Some(element(&[1, 2]))
        );
        assert_eq!(array.iter().last(), Some(Some(element(&[1, 2]))));
        assert_eq!(
            array.iter().nth_back(999_999_999),
            Some(Some(element(&[1, 2])))
        );
    }

    /// Nothing is stolen from the length to say which representation the values are in, so a
    /// scalar array reaches as far as a `usize` does.
    #[test]
    fn a_broadcast_array_is_as_long_as_it_says() {
        let array = PlFixedSizeListArray::new_scalar(element(&[1, 2]), usize::MAX);

        assert_eq!(array.values_iter().len(), usize::MAX);
        assert_eq!(array.values_iter().count(), usize::MAX);
        assert_eq!(
            array.values_iter().size_hint(),
            (usize::MAX, Some(usize::MAX))
        );
        assert_eq!(
            array.values_iter().nth(usize::MAX - 1),
            Some(element(&[1, 2]))
        );
        assert_eq!(array.values_iter().last(), Some(element(&[1, 2])));

        let mut iter = array.values_iter();
        assert_eq!(iter.next(), Some(element(&[1, 2])));
        assert_eq!(iter.next_back(), Some(element(&[1, 2])));
        assert_eq!(iter.len(), usize::MAX - 2);
    }
}
