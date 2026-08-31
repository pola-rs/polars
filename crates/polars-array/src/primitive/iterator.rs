use std::ops::Range;

use arrow::bitmap::Bitmap;
use arrow::trusted_len::TrustedLen;
use arrow::types::NativeType;

use crate::broadcast::broadcast_index;

/// Iterator over the values of a [`PlPrimitiveArray`](super::PlPrimitiveArray), ignoring validity.
#[derive(Clone)]
pub struct PlPrimitiveValuesIter<'a, T: NativeType> {
    values: &'a [T],
    range: Range<usize>,
}

impl<'a, T: NativeType> PlPrimitiveValuesIter<'a, T> {
    /// # Safety
    /// `values` must be dense or broadcast for `length`, per [`crate::broadcast`].
    #[inline]
    pub(super) fn new(values: &'a [T], length: usize) -> Self {
        Self {
            values,
            range: 0..length,
        }
    }

    #[inline(always)]
    fn get(&self, i: usize) -> T {
        // SAFETY: `i` comes from `self.range`, so the broadcast index is in bounds.
        unsafe {
            *self
                .values
                .get_unchecked(broadcast_index(i, self.values.len()))
        }
    }
}

impl<T: NativeType> Iterator for PlPrimitiveValuesIter<'_, T> {
    type Item = T;

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
}

impl<T: NativeType> DoubleEndedIterator for PlPrimitiveValuesIter<'_, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.range.next_back().map(|i| self.get(i))
    }
}

impl<T: NativeType> ExactSizeIterator for PlPrimitiveValuesIter<'_, T> {}
unsafe impl<T: NativeType> TrustedLen for PlPrimitiveValuesIter<'_, T> {}

/// Iterator over the optional elements of a [`PlPrimitiveArray`](super::PlPrimitiveArray).
#[derive(Clone)]
pub struct PlPrimitiveIter<'a, T: NativeType> {
    values: &'a [T],
    validity: Option<&'a Bitmap>,
    range: Range<usize>,
}

impl<'a, T: NativeType> PlPrimitiveIter<'a, T> {
    /// # Safety
    /// `values` and `validity` must be dense or broadcast for `length`, per [`crate::broadcast`].
    #[inline]
    pub(super) fn new(values: &'a [T], validity: Option<&'a Bitmap>, length: usize) -> Self {
        Self {
            values,
            validity,
            range: 0..length,
        }
    }

    #[inline(always)]
    fn get(&self, i: usize) -> Option<T> {
        // SAFETY: `i` comes from `self.range`, so the broadcast indices are in bounds.
        let is_valid = match self.validity {
            None => true,
            Some(validity) => unsafe {
                validity.get_bit_unchecked(broadcast_index(i, validity.len()))
            },
        };

        is_valid.then(|| unsafe {
            *self
                .values
                .get_unchecked(broadcast_index(i, self.values.len()))
        })
    }
}

impl<T: NativeType> Iterator for PlPrimitiveIter<'_, T> {
    type Item = Option<T>;

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
}

impl<T: NativeType> DoubleEndedIterator for PlPrimitiveIter<'_, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.range.next_back().map(|i| self.get(i))
    }
}

impl<T: NativeType> ExactSizeIterator for PlPrimitiveIter<'_, T> {}
unsafe impl<T: NativeType> TrustedLen for PlPrimitiveIter<'_, T> {}
