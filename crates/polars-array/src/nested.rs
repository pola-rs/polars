//! The walk over the elements a nested array cuts its values into.
//!
//! A [`PlListArray`](crate::PlListArray) cuts them at offsets and a
//! [`PlFixedSizeListArray`](crate::PlFixedSizeListArray) cuts them every width; what they do with
//! the ranges that come out — box one element, or read the mask alongside them — is the same, and
//! is written here once over the [`Shape`] that says how the cutting goes.

use std::marker::PhantomData;
use std::ops::Range;
use std::ptr::NonNull;

use arrow::trusted_len::TrustedLen;

use crate::array::PlArray;
use crate::bitmap::{PlBitmapRef, ValidityFold, ValidityIter};
use crate::broadcast::{is_flat_offsets_len, is_valid_fixed_size_values_len, is_valid_offsets_len};

/// How a nested array cuts its values into the elements a walk has left to yield.
///
/// A shape carries a cursor of its own, which [`Self::advance`] walks and [`Self::at`] reads from;
/// the number of elements left is [`Ranges`]'s to keep, and is what bounds both of them.
pub trait Shape: Clone {
    /// The range of the values that the element `n` positions on from the front covers.
    ///
    /// # Safety
    /// `n` must be below the number of elements left, so that the cut it reads is one this shape
    /// still holds.
    unsafe fn at(&self, n: usize) -> Range<usize>;

    /// Drops the `n` elements at the front, leaving the cursor at the one after them.
    ///
    /// # Safety
    /// `n` must not exceed the number of elements left, so that the cursor lands on an element or
    /// one past the last of them.
    unsafe fn advance(&mut self, n: usize);

    /// Folds `f` over the ranges of the front `n` elements, in order.
    ///
    /// This is where a shape says how to walk its cuts as the sequence they are, rather than one
    /// [`Self::at`] per element.
    ///
    /// # Safety
    /// `n` must not exceed the number of elements left.
    unsafe fn fold<B, F>(self, n: usize, init: B, f: F) -> B
    where
        F: FnMut(B, Range<usize>) -> B;

    /// Folds `f` over the ranges of the front `n` elements from the back, per [`Self::fold`].
    ///
    /// # Safety
    /// `n` must not exceed the number of elements left.
    unsafe fn rfold<B, F>(self, n: usize, init: B, f: F) -> B
    where
        F: FnMut(B, Range<usize>) -> B;
}

/// The offsets at which the elements left to yield start, one slot each and one after the last.
#[derive(Clone)]
pub struct Offsets<'a> {
    /// The offsets of the elements left to yield.
    offsets: NonNull<u64>,
    /// Folds a position onto slot 0 when the offsets are scalar: [`usize::MAX`] flat, `0` scalar.
    index_mask: usize,
    _lifetime: PhantomData<&'a [u64]>,
}

// SAFETY: the walk holds nothing but the shared borrow of the offsets it was built from, which is
// `Send` and `Sync` itself; the raw pointer only drops the length.
unsafe impl Send for Offsets<'_> {}
unsafe impl Sync for Offsets<'_> {}

impl<'a> Offsets<'a> {
    /// # Safety
    /// `offsets` must be flat or scalar for `length`, per [`crate::broadcast`], and be ordered.
    #[inline]
    pub(crate) fn new(offsets: &'a [u64], length: usize) -> Self {
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
            _lifetime: PhantomData,
        }
    }

    /// How far the offsets walk per element dropped: one slot while flat, nowhere once scalar.
    #[inline(always)]
    fn step(&self) -> usize {
        size_of::<u64>() & self.index_mask
    }

    /// Whether every element reads the one range the offsets hold.
    #[inline(always)]
    fn is_scalar(&self) -> bool {
        self.index_mask == 0
    }
}

impl Shape for Offsets<'_> {
    #[inline(always)]
    unsafe fn at(&self, n: usize) -> Range<usize> {
        unsafe {
            // Scalar offsets fold every position onto the one range they hold; flat ones hold the
            // start of the element and the end that follows it.
            let offsets = self.offsets.as_ptr().byte_add(n.wrapping_mul(self.step()));
            let start = offsets.read() as usize;
            let end = offsets.add(1).read() as usize;

            start..end
        }
    }

    #[inline(always)]
    unsafe fn advance(&mut self, n: usize) {
        // SAFETY: flat offsets hold one slot more than the elements left, so `n` of them is at
        // most one past their end; scalar ones are walked nowhere.
        self.offsets = unsafe { self.offsets.byte_add(n.wrapping_mul(self.step())) };
    }

    /// Walks the offsets as the buffer they are, reading the end of one element as the start of
    /// the next rather than reading every slot twice.
    #[inline]
    unsafe fn fold<B, F>(self, n: usize, init: B, mut f: F) -> B
    where
        F: FnMut(B, Range<usize>) -> B,
    {
        let mut acc = init;

        if n == 0 {
            return acc;
        }

        // SAFETY: there is an element left, so the front is the start of one.
        let front = unsafe { self.at(0) };

        if self.is_scalar() {
            // Scalar offsets hold the one range every element covers, read here and never again.
            for _ in 0..n {
                acc = f(acc, front.clone());
            }

            return acc;
        }

        let offsets = self.offsets.as_ptr();
        let mut start = front.start;

        for i in 1..=n {
            // SAFETY: flat offsets hold one slot more than the elements left to yield, so the end
            // of the last of them is the last slot read here.
            let end = unsafe { offsets.add(i).read() } as usize;

            acc = f(acc, start..end);
            start = end;
        }

        acc
    }

    #[inline]
    unsafe fn rfold<B, F>(self, n: usize, init: B, mut f: F) -> B
    where
        F: FnMut(B, Range<usize>) -> B,
    {
        let mut acc = init;

        if n == 0 {
            return acc;
        }

        // SAFETY: there is an element left, so the front is the start of one.
        let front = unsafe { self.at(0) };

        if self.is_scalar() {
            // Scalar offsets hold the one range every element covers, whichever end it is read
            // from.
            for _ in 0..n {
                acc = f(acc, front.clone());
            }

            return acc;
        }

        let offsets = self.offsets.as_ptr();
        // SAFETY: flat offsets hold one slot more than the elements left, the last of which is the
        // end of the one at the back.
        let mut end = unsafe { offsets.add(n).read() } as usize;

        for i in (0..n).rev() {
            // SAFETY: `i` is below the number of elements left, so it is one of their starts.
            let start = unsafe { offsets.add(i).read() } as usize;

            acc = f(acc, start..end);
            end = start;
        }

        acc
    }
}

/// The width every element covers, from a front that walks one of them per element dropped.
#[derive(Clone)]
pub struct Stride {
    /// Where the element at the front starts.
    front: usize,
    /// How many values every element covers.
    width: usize,
    /// How far the front walks per element dropped: one width for flat values, nowhere for scalar
    /// ones.
    stride: usize,
}

impl Stride {
    /// # Safety
    /// The values must be flat or scalar for `length` and `width`, per [`crate::broadcast`].
    #[inline]
    pub(crate) fn new(values_len: usize, width: usize, length: usize) -> Self {
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
            width,
            // Flat values lay the elements end to end, one width apart; scalar ones hold the one
            // range every element reads, which the walk never steps off.
            stride: if scalar { 0 } else { width },
        }
    }

    /// Where the element `n` positions on from the front starts.
    #[inline(always)]
    fn start(&self, n: usize) -> usize {
        // Flat values reach `remaining * width` on from the front, so neither the product nor the
        // sum wraps; scalar ones are walked nowhere, whatever `n` is.
        self.front.wrapping_add(n.wrapping_mul(self.stride))
    }
}

impl Shape for Stride {
    #[inline(always)]
    unsafe fn at(&self, n: usize) -> Range<usize> {
        let start = self.start(n);
        start..start + self.width
    }

    #[inline(always)]
    unsafe fn advance(&mut self, n: usize) {
        self.front = self.start(n);
    }

    /// Walks the starts as the affine sequence they are, rather than one product per element.
    #[inline]
    unsafe fn fold<B, F>(self, n: usize, init: B, mut f: F) -> B
    where
        F: FnMut(B, Range<usize>) -> B,
    {
        let Self {
            mut front,
            width,
            stride,
        } = self;
        let mut acc = init;

        for _ in 0..n {
            acc = f(acc, front..front + width);
            front = front.wrapping_add(stride);
        }

        acc
    }

    #[inline]
    unsafe fn rfold<B, F>(self, n: usize, init: B, mut f: F) -> B
    where
        F: FnMut(B, Range<usize>) -> B,
    {
        // One element past the back, which the first step of the walk comes back down from.
        let mut back = self.start(n);
        let Self { width, stride, .. } = self;
        let mut acc = init;

        for _ in 0..n {
            back = back.wrapping_sub(stride);
            acc = f(acc, back..back + width);
        }

        acc
    }
}

/// The ranges of the values the elements left to yield cover, walked from either end.
#[derive(Clone)]
pub struct Ranges<S> {
    /// How the values are cut into elements, and where the front of the walk is.
    shape: S,
    /// The number of elements left to yield: a scalar array is as long as it says it is.
    remaining: usize,
}

impl<S: Shape> Ranges<S> {
    /// # Safety
    /// `shape` must cut `length` elements out of the values it was built from.
    #[inline]
    fn new(shape: S, length: usize) -> Self {
        Self {
            shape,
            remaining: length,
        }
    }

    /// Exhausts the walk, which leaves it yielding nothing from either end.
    #[inline(always)]
    fn exhaust(&mut self) {
        self.remaining = 0;
    }
}

impl<S: Shape> Iterator for Ranges<S> {
    type Item = Range<usize>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        // SAFETY: there is an element left, so the front is the start of one.
        let range = unsafe { self.shape.at(0) };
        // SAFETY: as above.
        unsafe { self.shape.advance(1) };
        self.remaining -= 1;

        Some(range)
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if n >= self.remaining {
            self.exhaust();
            return None;
        }

        // SAFETY: the `n` elements dropped are the front `n` of the ones left.
        unsafe { self.shape.advance(n) };
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

    /// Hoists the cuts out of the loop, per [`Shape::fold`].
    #[inline]
    fn fold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        // SAFETY: the shape cuts out every element the walk has left to yield.
        unsafe { self.shape.fold(self.remaining, init, f) }
    }
}

impl<S: Shape> DoubleEndedIterator for Ranges<S> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let last = self.remaining.checked_sub(1)?;

        // SAFETY: the element at the back is one of the ones left.
        let range = unsafe { self.shape.at(last) };
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

    /// Hoists the cuts out of the loop, the way [`Iterator::fold`] does.
    #[inline]
    fn rfold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        // SAFETY: the shape cuts out every element the walk has left to yield.
        unsafe { self.shape.rfold(self.remaining, init, f) }
    }
}

impl<S: Shape> ExactSizeIterator for Ranges<S> {
    #[inline]
    fn len(&self) -> usize {
        self.remaining
    }
}

// SAFETY: the walk yields one range per element left, which is what it says it has.
unsafe impl<S: Shape> TrustedLen for Ranges<S> {}

/// The element the values hold over `range`, which is a fresh box over the same buffers.
///
/// # Safety
/// `range` must be ordered and in bounds of the values, as every range [`Ranges`] yields is.
#[inline(always)]
unsafe fn element(values: &dyn PlArray, range: Range<usize>) -> Box<dyn PlArray> {
    debug_assert!(range.start <= range.end);
    debug_assert!(range.end <= values.len());
    // SAFETY: the element is in bounds of the values, per the caller.
    unsafe { values.sliced_unchecked(range.start, range.end - range.start) }
}

/// Iterator over the elements of a nested array, ignoring validity.
#[derive(Clone)]
pub struct NestedValuesIter<'a, S> {
    /// The values array the elements are cut out of.
    values: &'a dyn PlArray,
    /// The ranges of the elements left to yield.
    ranges: Ranges<S>,
}

impl<'a, S: Shape> NestedValuesIter<'a, S> {
    /// # Safety
    /// `shape` must cut `length` elements out of `values`.
    #[inline]
    pub(crate) fn new(values: &'a dyn PlArray, shape: S, length: usize) -> Self {
        Self {
            values,
            ranges: Ranges::new(shape, length),
        }
    }
}

crate::impl_mapped_iter!(
    [S: Shape] NestedValuesIter<'a, S>,
    Box<dyn PlArray>,
    over: ranges,
    with: [values],
    // SAFETY: the range is one of this iterator's elements.
    |range| unsafe { element(values, range) },
);

/// Iterator over the optional elements of a nested array.
#[derive(Clone)]
pub struct NestedIter<'a, S> {
    values: &'a dyn PlArray,
    ranges: Ranges<S>,
    validity: ValidityIter<'a>,
}

impl<'a, S: Shape> NestedIter<'a, S> {
    /// # Safety
    /// `shape` must cut `length` elements out of `values`.
    #[inline]
    pub(crate) fn new(
        values: &'a dyn PlArray,
        shape: S,
        validity: Option<PlBitmapRef<'a>>,
        length: usize,
    ) -> Self {
        assert!(validity.is_none_or(|validity| validity.len() == length));

        Self {
            values,
            ranges: Ranges::new(shape, length),
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

    /// The ranges of the elements left to yield and the mask that says which of them are elements.
    #[inline]
    fn split(self) -> (&'a dyn PlArray, Ranges<S>, ValidityFold<'a>) {
        (self.values, self.ranges, self.validity.into_mask())
    }
}

/// The iterator traits are written out rather than taken from `impl_optional_iter`, which reads
/// the mask only where the values yielded an element: an element here is an array of its own, so
/// the mask is read first and the walk over the ranges yields nothing but a shape, which leaves a
/// null position paying for no array at all.
impl<S: Shape> Iterator for NestedIter<'_, S> {
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

    /// Hoists the validity mask out of the loop, and the walk over the ranges with it.
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

impl<S: Shape> DoubleEndedIterator for NestedIter<'_, S> {
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

impl<S: Shape> ExactSizeIterator for NestedIter<'_, S> {
    #[inline]
    fn len(&self) -> usize {
        self.ranges.len()
    }
}

// SAFETY: the ranges are trusted to yield as many elements as they say they will, and the mask is
// walked alongside them.
unsafe impl<S: Shape> TrustedLen for NestedIter<'_, S> {}
