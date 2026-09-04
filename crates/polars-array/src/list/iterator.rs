use std::marker::PhantomData;
use std::ops::Range;
use std::ptr::NonNull;

use arrow::trusted_len::TrustedLen;

use crate::array::PlArray;
use crate::bitmap::{PlBitmapRef, ValidityFold, ValidityIter};
use crate::broadcast::{is_flat_offsets_len, is_valid_offsets_len};

/// The ranges of the values the elements left to yield cover.
///
/// The representation of the offsets sets nothing but the step the walk takes through them: one
/// slot while they hold a start per element, and nowhere at all once they hold the one range every
/// element reads.
///
/// Resolving that step once, at construction, is what leaves the walk a plain add. Deciding per
/// element instead leaves every read behind a load of the offsets out of the array and a select on
/// how many there are — neither of which the loop can hoist, fold against what the caller has
/// asserted about the array, or see past.
#[derive(Clone)]
struct Ranges<'a> {
    /// The offsets of the elements left to yield.
    ///
    /// Flat offsets hold exactly `remaining + 1` slots from here, which walking the front keeps
    /// true; scalar offsets hold their two slots wherever the front has walked to, which is
    /// nowhere — the step every walk takes is zero for them.
    offsets: NonNull<u64>,
    /// [`usize::MAX`] while the offsets are flat and `0` once they are scalar, to fold every
    /// position onto the one range scalar offsets hold without branching on which they are.
    ///
    /// This is a field rather than a tag bit stolen from `offsets` so that the walk never has to
    /// read it back out of a pointer it is also stepping: held apart, it is loop invariant by
    /// construction, and is a constant outright wherever the caller has pinned the representation
    /// down, which leaves the offsets walked at a constant stride.
    index_mask: usize,
    /// The number of elements left to yield, over the whole range of a `usize`: a scalar array is
    /// as long as it says it is, and is never walked to find out.
    remaining: usize,
    _lifetime: PhantomData<&'a [u64]>,
}

// SAFETY: the walk holds nothing but the shared borrow of the offsets it was built from, which is
// `Send` and `Sync` itself; the raw pointer only drops the length.
unsafe impl Send for Ranges<'_> {}
unsafe impl Sync for Ranges<'_> {}

impl<'a> Ranges<'a> {
    /// # Safety
    /// `offsets` must be flat (`length + 1` offsets) or scalar (two offsets, or one for an empty
    /// array) for `length`, per [`crate::broadcast`], and must be ordered.
    #[inline]
    fn new(offsets: &'a [u64], length: usize) -> Self {
        // Offsets that hold one start per element are flat, and offsets the caller promises are
        // valid are scalar when they are not. The two coincide for a single element, which either
        // reading cuts the same range out for.
        let scalar = !is_flat_offsets_len(offsets.len(), length);

        debug_assert!(
            is_valid_offsets_len(offsets.len(), length),
            "neither flat nor scalar",
        );
        debug_assert!(offsets.first() <= offsets.last(), "offsets out of order");

        Self {
            offsets: NonNull::from(offsets).cast(),
            // All ones for flat offsets, which leaves every position as it is, and none for scalar
            // ones, which folds every position onto the single range they hold.
            index_mask: (scalar as usize).wrapping_sub(1),
            remaining: length,
            _lifetime: PhantomData,
        }
    }

    /// How far the offsets walk per element dropped: one slot while they are flat, and nowhere at
    /// all once they are scalar, whose two slots every element reads.
    #[inline(always)]
    fn step(&self) -> usize {
        size_of::<u64>() & self.index_mask
    }

    /// The range the element `n` positions on from the front covers.
    ///
    /// # Safety
    /// `n` must be below the number of elements left, so that the start it reads and the end after
    /// it are both in bounds of the offsets.
    #[inline(always)]
    unsafe fn at(&self, n: usize) -> Range<usize> {
        debug_assert!(n < self.remaining);

        unsafe {
            // Scalar offsets fold every position onto the one range they hold; flat ones hold the
            // start of the element and the end that follows it.
            let offsets = self.offsets.as_ptr().byte_add(n.wrapping_mul(self.step()));
            let start = offsets.read() as usize;
            let end = offsets.add(1).read() as usize;

            start..end
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

        // SAFETY: flat offsets hold one slot more than the elements left, so `n` of them is at
        // most one past their end; scalar ones are walked nowhere.
        self.offsets = unsafe { self.offsets.byte_add(n.wrapping_mul(self.step())) };
        self.remaining -= n;
    }

    /// Exhausts the walk, which leaves it yielding nothing from either end.
    ///
    /// The offsets are left where they are: a flat pointer still holds the one slot an empty walk
    /// reads, and nothing reads it.
    #[inline(always)]
    fn exhaust(&mut self) {
        self.remaining = 0;
    }
}

impl Iterator for Ranges<'_> {
    type Item = Range<usize>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        // SAFETY: there is an element left, so the front is the start of one.
        let range = unsafe { self.at(0) };
        // SAFETY: as above.
        unsafe { self.advance(1) };

        Some(range)
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if n >= self.remaining {
            self.exhaust();
            return None;
        }

        // SAFETY: the `n` elements dropped are the front `n` of the ones left.
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

    /// Walks the offsets as the buffer they are, rather than stepping one `Option` at a time —
    /// which is what `collect` and `for_each` route through.
    ///
    /// Flat offsets are read once each: the end of one element is the start of the next, so the
    /// loop carries it rather than loading it twice. Scalar ones are read once for the whole walk.
    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let mut acc = init;

        if self.remaining == 0 {
            return acc;
        }

        // SAFETY: there is an element left, so the front is the start of one.
        let front = unsafe { self.at(0) };

        if self.index_mask == 0 {
            // Scalar offsets hold the one range every element covers, read here and never again.
            for _ in 0..self.remaining {
                acc = f(acc, front.clone());
            }

            return acc;
        }

        let offsets = self.offsets.as_ptr();
        let mut start = front.start;

        for i in 1..=self.remaining {
            // SAFETY: flat offsets hold one slot more than the elements left to yield, so the end
            // of the last of them is the last slot read here.
            let end = unsafe { offsets.add(i).read() } as usize;

            acc = f(acc, start..end);
            start = end;
        }

        acc
    }
}

impl DoubleEndedIterator for Ranges<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let last = self.remaining.checked_sub(1)?;

        // SAFETY: the element at the back is one of the ones left.
        let range = unsafe { self.at(last) };
        self.remaining = last;

        Some(range)
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

    /// Walks the offsets from the back, the way [`Iterator::fold`] does from the front.
    #[inline]
    fn rfold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let mut acc = init;

        if self.remaining == 0 {
            return acc;
        }

        // SAFETY: there is an element left, so the front is the start of one.
        let front = unsafe { self.at(0) };

        if self.index_mask == 0 {
            // Scalar offsets hold the one range every element covers, whichever end it is read
            // from.
            for _ in 0..self.remaining {
                acc = f(acc, front.clone());
            }

            return acc;
        }

        let offsets = self.offsets.as_ptr();
        // SAFETY: flat offsets hold one slot more than the elements left, the last of which is the
        // end of the one at the back.
        let mut end = unsafe { offsets.add(self.remaining).read() } as usize;

        for i in (0..self.remaining).rev() {
            // SAFETY: `i` is below the number of elements left, so it is one of their starts.
            let start = unsafe { offsets.add(i).read() } as usize;

            acc = f(acc, start..end);
            end = start;
        }

        acc
    }
}

impl ExactSizeIterator for Ranges<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.remaining
    }
}

unsafe impl TrustedLen for Ranges<'_> {}

/// The element the values hold over `range`, which is a fresh box over the same buffers.
///
/// # Safety
/// `range` must be ordered and in bounds of the values, which it is for every range an
/// [`Ranges`] built for them yields.
#[inline(always)]
unsafe fn element(values: &dyn PlArray, range: Range<usize>) -> Box<dyn PlArray> {
    debug_assert!(range.start <= range.end);
    debug_assert!(range.end <= values.len());
    // SAFETY: the element is in bounds of the values, per the caller.
    unsafe { values.sliced_unchecked(range.start, range.end - range.start) }
}

/// Iterator over the elements of a [`PlListArray`](super::PlListArray), ignoring validity.
///
/// The representation of the offsets is resolved once, at construction, into the step the walk
/// takes through them; see [`Ranges`].
#[derive(Clone)]
pub struct PlListValuesIter<'a> {
    /// The values array the elements are cut out of.
    ///
    /// Held as the borrow it is, rather than reached for through the array, so that the walk keeps
    /// it in a register instead of loading the box out of the array on every element.
    values: &'a dyn PlArray,
    /// The ranges of the elements left to yield.
    ranges: Ranges<'a>,
}

impl<'a> PlListValuesIter<'a> {
    /// # Safety
    /// `offsets` must be flat (`length + 1` offsets) or scalar (two offsets) for `length`, per
    /// [`crate::broadcast`], and must be ordered and bounded by the length of `values`.
    #[inline]
    pub(super) fn new(values: &'a dyn PlArray, offsets: &'a [u64], length: usize) -> Self {
        Self {
            values,
            // SAFETY: the offsets are flat or scalar for `length`, per the caller.
            ranges: Ranges::new(offsets, length),
        }
    }

    /// The element covering `range`.
    ///
    /// # Safety
    /// `range` must be one the iterator's own ranges yielded.
    #[inline(always)]
    unsafe fn get(&self, range: Range<usize>) -> Box<dyn PlArray> {
        // SAFETY: the range is one of this iterator's elements.
        unsafe { element(self.values, range) }
    }
}

impl Iterator for PlListValuesIter<'_> {
    type Item = Box<dyn PlArray>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let range = self.ranges.next()?;
        // SAFETY: the range is one of this iterator's elements.
        Some(unsafe { self.get(range) })
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let range = self.ranges.nth(n)?;
        // SAFETY: the range is one of this iterator's elements.
        Some(unsafe { self.get(range) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.ranges.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.ranges.count()
    }

    /// Walks to the last element from the back, rather than through every one before it.
    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    /// Hoists the walk over the offsets out of the loop, per [`Ranges::fold`].
    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let values = self.values;

        self.ranges.fold(init, |acc, range| {
            // SAFETY: the range is one of this iterator's elements.
            f(acc, unsafe { element(values, range) })
        })
    }
}

impl DoubleEndedIterator for PlListValuesIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let range = self.ranges.next_back()?;
        // SAFETY: the range is one of this iterator's elements.
        Some(unsafe { self.get(range) })
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let range = self.ranges.nth_back(n)?;
        // SAFETY: the range is one of this iterator's elements.
        Some(unsafe { self.get(range) })
    }

    /// Hoists the walk over the offsets out of the loop, the way [`Iterator::fold`] does.
    #[inline]
    fn rfold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let values = self.values;

        self.ranges.rfold(init, |acc, range| {
            // SAFETY: the range is one of this iterator's elements.
            f(acc, unsafe { element(values, range) })
        })
    }
}

impl ExactSizeIterator for PlListValuesIter<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.ranges.len()
    }
}

unsafe impl TrustedLen for PlListValuesIter<'_> {}

/// Iterator over the optional elements of a [`PlListArray`](super::PlListArray).
///
/// The mask gates the element rather than the other way around: an element of this array is a
/// fresh box over the values, so building one for a null position — only to throw it away — costs
/// an allocation and a free that reading the bit first does not pay at all.
#[derive(Clone)]
pub struct PlListIter<'a> {
    values: &'a dyn PlArray,
    ranges: Ranges<'a>,
    validity: ValidityIter<'a>,
}

impl<'a> PlListIter<'a> {
    /// # Panics
    /// Panics unless `validity` has `length` bits.
    ///
    /// # Safety
    /// `offsets` must be flat or scalar for `length`, per [`crate::broadcast`], and must be ordered
    /// and bounded by the length of `values`.
    #[inline]
    pub(super) fn new(
        values: &'a dyn PlArray,
        offsets: &'a [u64],
        validity: Option<PlBitmapRef<'a>>,
        length: usize,
    ) -> Self {
        assert!(validity.is_none_or(|validity| validity.len() == length));

        Self {
            values,
            // SAFETY: the offsets are flat or scalar for `length`, per the caller.
            ranges: Ranges::new(offsets, length),
            validity: ValidityIter::new(validity),
        }
    }

    /// The element covering `range` if `is_valid`, built only where it is.
    ///
    /// # Safety
    /// `range` must be one the iterator's own ranges yielded.
    #[inline(always)]
    unsafe fn get(&self, is_valid: bool, range: Range<usize>) -> Option<Box<dyn PlArray>> {
        // SAFETY: the range is one of this iterator's elements.
        is_valid.then(|| unsafe { element(self.values, range) })
    }

    /// The ranges of the elements left to yield and the mask that says which of them are elements,
    /// to walk in one loop.
    #[inline]
    fn split(self) -> (&'a dyn PlArray, Ranges<'a>, ValidityFold<'a>) {
        (self.values, self.ranges, self.validity.into_mask())
    }
}

impl Iterator for PlListIter<'_> {
    type Item = Option<Box<dyn PlArray>>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let range = self.ranges.next()?;
        let is_valid = self.validity.next();
        // SAFETY: the range is one of this iterator's elements.
        Some(unsafe { self.get(is_valid, range) })
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        // The mask is advanced alongside the ranges, whether or not there is an element left.
        let is_valid = self.validity.nth(n);
        let range = self.ranges.nth(n)?;
        // SAFETY: the range is one of this iterator's elements.
        Some(unsafe { self.get(is_valid, range) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.ranges.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.ranges.count()
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
        let (values, ranges, mask) = self.split();
        // The element is built only where the mask says there is one, so a null position pays for
        // no box at all.
        let element = |range: Option<Range<usize>>| {
            // SAFETY: the range is one of this iterator's elements.
            range.map(|range| unsafe { element(values, range) })
        };

        // SAFETY: the mask has one bit per element, and the ranges and the mask are walked in
        // lockstep, so it has a bit for every range left to yield.
        unsafe { mask.fold_values(ranges, init, |acc, range| f(acc, element(range))) }
    }
}

impl DoubleEndedIterator for PlListIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let range = self.ranges.next_back()?;
        let is_valid = self.validity.next_back();
        // SAFETY: the range is one of this iterator's elements.
        Some(unsafe { self.get(is_valid, range) })
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        // The mask is advanced alongside the ranges, whether or not there is an element left.
        let is_valid = self.validity.nth_back(n);
        let range = self.ranges.nth_back(n)?;
        // SAFETY: the range is one of this iterator's elements.
        Some(unsafe { self.get(is_valid, range) })
    }

    /// Hoists the validity mask out of the loop, the way [`Iterator::fold`] does.
    #[inline]
    fn rfold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let (values, ranges, mask) = self.split();
        // The element is built only where the mask says there is one, per [`Iterator::fold`].
        let element = |range: Option<Range<usize>>| {
            // SAFETY: the range is one of this iterator's elements.
            range.map(|range| unsafe { element(values, range) })
        };

        // SAFETY: the mask has a bit for every range left to yield, per `Iterator::fold`.
        unsafe { mask.rfold_values(ranges, init, |acc, range| f(acc, element(range))) }
    }
}

impl ExactSizeIterator for PlListIter<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.ranges.len()
    }
}

unsafe impl TrustedLen for PlListIter<'_> {}

#[cfg(test)]
mod tests {
    use arrow::bitmap::Bitmap;
    use polars_buffer::Buffer;

    use crate::bitmap::PlBitmap;
    use crate::iterator_tests::assert_iterates;
    use crate::{PlArray, PlListArray, PlPrimitiveArray};

    /// The list `values` are, as an element of a list array is.
    fn element(values: &[i32]) -> Box<dyn PlArray> {
        Box::new(PlPrimitiveArray::from_vec(values.to_vec()))
    }

    /// A flat array of the lists `[1, 2]`, `[]` and `[3]`.
    fn flat_array() -> PlListArray {
        PlListArray::new(
            element(&[1, 2, 3]),
            Buffer::from_owner([0, 2, 2, 3]),
            3,
            None,
        )
    }

    fn elements() -> [Box<dyn PlArray>; 3] {
        [element(&[1, 2]), element(&[]), element(&[3])]
    }

    #[test]
    fn flat() {
        let array = flat_array();

        assert_iterates(array.values_iter(), &elements());
        assert_iterates(array.iter(), &elements().map(Some));
    }

    #[test]
    fn scalar() {
        let array = PlListArray::new_scalar(element(&[1, 2]), 4);
        let expected = [(); 4].map(|()| element(&[1, 2]));

        assert_iterates(array.values_iter(), &expected);
        assert_iterates(array.iter(), &expected.map(Some));
    }

    /// The elements of a sliced array start partway into the offsets, which still hold the end of
    /// the last of them.
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

    /// An array of no elements, whose offsets hold nothing but the end an empty walk never reads.
    #[test]
    fn empty() {
        assert_iterates(flat_array().sliced(0, 0).values_iter(), &[]);
        assert_iterates(flat_array().sliced(0, 0).iter(), &[]);
        assert_iterates(
            PlListArray::new_scalar(element(&[1, 2]), 0).values_iter(),
            &[],
        );
        assert_iterates(PlListArray::new_scalar(element(&[1, 2]), 0).iter(), &[]);
    }

    /// A mask of mixed bits, which is read by position alongside the elements the offsets cut out.
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
        let null = PlListArray::new_full_null(element(&[1, 2]), 3);

        assert_iterates(null.iter(), &[None, None, None]);
        assert_eq!(null.iter().nth(2), Some(None));
        assert_eq!(null.iter().nth_back(2), Some(None));
        assert_eq!(null.iter().last(), Some(None));
    }

    /// An array of a single element, whose offsets are flat and scalar at once, broadcast over a
    /// walk of many positions.
    #[test]
    fn broadcast() {
        let array = PlListArray::new(element(&[1, 2, 3]), Buffer::from_owner([1, 3]), 1, None);
        let expected = [(); 4].map(|()| element(&[2, 3]));

        assert_iterates(array.broadcast_values_iter(4), &expected);
        assert_iterates(array.broadcast_values_iter(1), &expected[..1]);
    }

    #[test]
    fn a_broadcast_array_is_not_materialized() {
        // Walking a billion elements would not finish; the scalar path must hit.
        let array = PlListArray::new_scalar(element(&[1, 2]), 1_000_000_000);

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

    /// Nothing is stolen from the length to say which representation the offsets are in, so a
    /// scalar array reaches as far as a `usize` does.
    #[test]
    fn a_broadcast_array_is_as_long_as_it_says() {
        let array = PlListArray::new_scalar(element(&[1, 2]), usize::MAX);

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
