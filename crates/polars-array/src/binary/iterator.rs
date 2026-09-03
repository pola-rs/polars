use std::ops::Range;

use arrow::trusted_len::TrustedLen;

use crate::bitmap::{PlBitmapRef, ValidityFold, ValidityIter};

/// Iterator over the elements of a [`PlBinaryArray`](super::PlBinaryArray), ignoring validity.
#[derive(Clone)]
pub struct PlBinaryValuesIter<'a> {
    values: &'a [u8],
    offsets: &'a [u64],
    range: Range<usize>,
}

impl<'a> PlBinaryValuesIter<'a> {
    /// # Safety
    /// `offsets` must be flat (`length + 1` offsets) or scalar (two offsets) for `length`, per
    /// [`crate::broadcast`], and must be ordered and bounded by the length of `values`.
    #[inline]
    pub(super) fn new(values: &'a [u8], offsets: &'a [u64], length: usize) -> Self {
        Self {
            values,
            offsets,
            range: 0..length,
        }
    }

    /// Whether the offsets hold the one range every element covers, rather than one per element.
    ///
    /// Two offsets are a single range, which is what scalar offsets are — and what the offsets of
    /// an array of one element are too, whichever representation it is in. Either way position
    /// zero is the only start there is. This does not depend on the position being read, so a
    /// loop over the elements settles it once rather than at every one of them.
    #[inline(always)]
    fn is_scalar(&self) -> bool {
        self.offsets.len() == 2
    }

    /// The bytes the element at position `i` covers.
    #[inline(always)]
    fn get(&self, i: usize) -> &'a [u8] {
        // SAFETY: `i` is a start the offsets hold and `i + 1` the end after it; the offsets are
        // ordered and bounded by the length of the values, so the range they cover is in bounds.
        unsafe {
            let slot = if self.is_scalar() { 0 } else { i };
            let start = *self.offsets.get_unchecked(slot) as usize;
            let end = *self.offsets.get_unchecked(slot + 1) as usize;
            self.values.get_unchecked(start..end)
        }
    }
}

impl<'a> Iterator for PlBinaryValuesIter<'a> {
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

    /// Hoists the representation of the offsets out of the loop: flat ones are walked as the
    /// consecutive ranges they are, and scalar ones fold over the one range they hold.
    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let count = self.range.len();

        if self.is_scalar() {
            // The offsets are scalar, so every element covers the one range their two slots hold
            // — which is what position zero reads, and is read once here rather than per element.
            if count == 0 {
                return init;
            }
            let value = self.get(0);
            return (0..count).fold(init, |acc, _| f(acc, value));
        }

        let (values, offsets) = (self.values, self.offsets);
        // SAFETY: flat offsets hold the start of every element plus the end of the last, so the
        // starts the range covers and the end that follows them are in bounds.
        let starts = unsafe { offsets.get_unchecked(self.range.start..=self.range.end) };

        starts.windows(2).fold(init, |acc, range| {
            // SAFETY: the offsets are ordered and bounded by the length of the values.
            let value = unsafe { values.get_unchecked(range[0] as usize..range[1] as usize) };
            f(acc, value)
        })
    }
}

impl DoubleEndedIterator for PlBinaryValuesIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.range.next_back().map(|i| self.get(i))
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.range.nth_back(n).map(|i| self.get(i))
    }
}

impl ExactSizeIterator for PlBinaryValuesIter<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.range.len()
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
    /// `offsets` must be flat or scalar for `length`, per [`crate::broadcast`], and must be
    /// ordered and bounded by the length of `values`.
    ///
    /// # Panics
    /// Panics unless `validity` has `length` bits.
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
    ///
    /// The mask has its representation hoisted out, so that the loop the caller runs reads no
    /// mask at all where every element is valid, and neither mask nor values where none is.
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

impl DoubleEndedIterator for PlBinaryIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let value = self.values.next_back()?;
        Some(self.validity.next_back().then_some(value))
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
    fn a_broadcast_array_is_not_materialized() {
        // Walking a billion elements would not finish; the scalar path must hit.
        let array = PlBinaryArray::new_scalar(b"xy", 1_000_000_000);

        assert_eq!(array.values_iter().count(), 1_000_000_000);
        assert_eq!(array.values_iter().nth(999_999_999), Some(b"xy".as_slice()));
        assert_eq!(array.iter().last(), Some(Some(b"xy".as_slice())));
    }
}
