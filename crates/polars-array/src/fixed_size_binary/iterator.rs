use std::slice::ChunksExact;

use arrow::trusted_len::TrustedLen;

use crate::bitmap::{PlBitmapRef, ValidityFold, ValidityIter};
use crate::broadcast::is_flat_fixed_size_values_len;

/// Iterator over the elements of a [`PlFixedSizeBinaryArray`](super::PlFixedSizeBinaryArray),
/// ignoring validity.
///
/// The representation of the values is resolved once, at construction, so that the flat arm is the
/// plain [`ChunksExact`] it is. Deciding per element instead leaves every read behind a load of the
/// length of the values and a select on it, and leaves the stride of the walk a value the loop
/// cannot hoist.
#[derive(Clone)]
pub struct PlFixedSizeBinaryValuesIter<'a> {
    repr: Repr<'a>,
}

/// The representation the values turned out to be in, resolved once.
#[derive(Clone)]
enum Repr<'a> {
    /// One element per position, `width` bytes each, walked as the chunks they are.
    Flat(ChunksExact<'a, u8>),
    /// The one element every position reads, and how many are left to read it.
    Scalar { value: &'a [u8], remaining: usize },
}

impl<'a> PlFixedSizeBinaryValuesIter<'a> {
    /// # Safety
    /// `values` must be flat (`length * width` bytes) or scalar (`width` bytes) for `length`, per
    /// [`crate::broadcast`].
    #[inline]
    pub(super) fn new(values: &'a [u8], width: usize, length: usize) -> Self {
        // Values as long as one element hold the one every position reads; values the caller
        // promises are valid hold one element each when they are not. The two coincide for a
        // single element, and for elements no bytes wide — which `chunks_exact` has no chunk for
        // at all — either of which the scalar arm yields the same bytes for.
        let repr = if values.len() == width {
            Repr::Scalar {
                value: values,
                remaining: length,
            }
        } else {
            // Values that are not as long as one element are flat, which for a width of zero
            // they cannot be: the flat length is zero, and that is the width.
            debug_assert!(
                is_flat_fixed_size_values_len(values.len(), width, length),
                "neither flat nor scalar",
            );
            Repr::Flat(values.chunks_exact(width))
        };

        Self { repr }
    }
}

impl<'a> Iterator for PlFixedSizeBinaryValuesIter<'a> {
    type Item = &'a [u8];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.repr {
            Repr::Flat(chunks) => chunks.next(),
            Repr::Scalar { value, remaining } => {
                *remaining = remaining.checked_sub(1)?;
                Some(value)
            },
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        match &mut self.repr {
            Repr::Flat(chunks) => chunks.nth(n),
            Repr::Scalar { value, remaining } => {
                let Some(left) = remaining.checked_sub(n + 1) else {
                    *remaining = 0;
                    return None;
                };
                *remaining = left;
                Some(value)
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
    fn last(self) -> Option<Self::Item> {
        match self.repr {
            Repr::Flat(chunks) => chunks.last(),
            Repr::Scalar { value, remaining } => (remaining != 0).then_some(value),
        }
    }

    /// Hoists the representation of the values out of the loop: flat ones fold as the consecutive
    /// elements they are, and scalar ones fold over the one element they hold.
    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        match self.repr {
            Repr::Flat(chunks) => chunks.fold(init, f),
            Repr::Scalar { value, remaining } => {
                let mut acc = init;
                for _ in 0..remaining {
                    acc = f(acc, value);
                }
                acc
            },
        }
    }
}

impl DoubleEndedIterator for PlFixedSizeBinaryValuesIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        match &mut self.repr {
            Repr::Flat(chunks) => chunks.next_back(),
            Repr::Scalar { value, remaining } => {
                *remaining = remaining.checked_sub(1)?;
                Some(value)
            },
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        if let Repr::Flat(chunks) = &mut self.repr {
            return chunks.nth_back(n);
        }

        // Every position of scalar values reads the one element they hold, so walking in from
        // either end drops the same number of them and reads the same bytes.
        self.nth(n)
    }

    /// Hoists the representation of the values out of the loop, the way [`Iterator::fold`] does.
    #[inline]
    fn rfold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        match self.repr {
            Repr::Flat(chunks) => chunks.rfold(init, f),
            Repr::Scalar { value, remaining } => {
                let mut acc = init;
                for _ in 0..remaining {
                    acc = f(acc, value);
                }
                acc
            },
        }
    }
}

impl ExactSizeIterator for PlFixedSizeBinaryValuesIter<'_> {
    #[inline]
    fn len(&self) -> usize {
        match &self.repr {
            Repr::Flat(chunks) => chunks.len(),
            Repr::Scalar { remaining, .. } => *remaining,
        }
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
