use std::ops::Range;

use arrow::trusted_len::TrustedLen;

use crate::bitmap::{PlBitmapRef, ValidityFold, ValidityIter};

/// Iterator over the elements of a [`PlFixedSizeBinaryArray`](super::PlFixedSizeBinaryArray),
/// ignoring validity.
#[derive(Clone)]
pub struct PlFixedSizeBinaryValuesIter<'a> {
    values: &'a [u8],
    width: usize,
    range: Range<usize>,
}

impl<'a> PlFixedSizeBinaryValuesIter<'a> {
    /// # Safety
    /// `values` must be flat (`length * width` bytes) or scalar (`width` bytes) for `length`, per
    /// [`crate::broadcast`].
    #[inline]
    pub(super) fn new(values: &'a [u8], width: usize, length: usize) -> Self {
        Self {
            values,
            width,
            range: 0..length,
        }
    }

    /// Whether the values hold the one element every element covers, rather than one per element.
    #[inline(always)]
    fn is_scalar(&self) -> bool {
        self.values.len() == self.width
    }

    #[inline(always)]
    fn get(&self, i: usize) -> &'a [u8] {
        let start = if self.is_scalar() { 0 } else { i * self.width };
        // SAFETY: `i` comes from `self.range`, so the element it reads is in bounds of the values:
        // either flat over every position, or the one element all of them read from the start.
        unsafe { self.values.get_unchecked(start..start + self.width) }
    }
}

impl<'a> Iterator for PlFixedSizeBinaryValuesIter<'a> {
    type Item = &'a [u8];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.range.next().map(|i| self.get(i))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.range.nth(n).map(|i| self.get(i))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.range.len()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    /// Hoists the representation of the values out of the loop: flat ones are walked as the
    /// consecutive elements they are, and scalar ones fold over the one element they hold.
    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let count = self.range.len();

        if self.is_scalar() {
            // The values are scalar, so every element covers the one element they hold — which
            // is what position zero reads, and is sliced once here rather than per element. This
            // is also where a width of zero lands, which `chunks_exact` has no chunk for.
            if count == 0 {
                return init;
            }
            let value = self.get(0);
            return (0..count).fold(init, |acc, _| f(acc, value));
        }

        // SAFETY: flat values lay the elements end to end, `width` bytes each, so the ones the
        // range covers are in bounds.
        let elements = unsafe {
            self.values
                .get_unchecked(self.range.start * self.width..self.range.end * self.width)
        };

        elements.chunks_exact(self.width).fold(init, f)
    }
}

impl DoubleEndedIterator for PlFixedSizeBinaryValuesIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.range.next_back().map(|i| self.get(i))
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.range.nth_back(n).map(|i| self.get(i))
    }
}

impl ExactSizeIterator for PlFixedSizeBinaryValuesIter<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.range.len()
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
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        match self.split() {
            (values, ValidityFold::Valid) => values.fold(init, |acc, value| f(acc, Some(value))),
            (values, ValidityFold::Null) => values.fold(init, |acc, _| f(acc, None)),
            (values, ValidityFold::Bits(mask)) => {
                values.zip(mask).fold(init, |acc, (value, is_valid)| {
                    f(acc, is_valid.then_some(value))
                })
            },
        }
    }
}

impl DoubleEndedIterator for PlFixedSizeBinaryIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let value = self.values.next_back()?;
        Some(self.validity.next_back().then_some(value))
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
        assert_eq!(array.iter().last(), Some(Some(b"xy".as_slice())));
    }
}
