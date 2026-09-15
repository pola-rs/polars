use arrow::trusted_len::TrustedLen;
use arrow::types::NativeType;
use polars_utils::slice_broadcast_iter::SliceBroadcastIter;

use crate::bitmap::{PlBitmapRef, ValidityFold, ValidityIter};
use crate::broadcast::broadcast_slice;

/// Iterator over the values of a [`PlPrimitiveArray`](super::PlPrimitiveArray), ignoring validity.
#[derive(Clone)]
pub struct PlPrimitiveValuesIter<'a, T: NativeType> {
    values: SliceBroadcastIter<'a, T>,
}

impl<'a, T: NativeType> PlPrimitiveValuesIter<'a, T> {
    #[inline]
    pub(super) fn new(values: &'a [T], length: usize) -> Self {
        Self {
            values: broadcast_slice(values, length),
        }
    }
}

impl<T: NativeType> Iterator for PlPrimitiveValuesIter<'_, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.values.next().copied()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.values.nth(n).copied()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.values.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.values.count()
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        self.values.last().copied()
    }

    /// Hoists the representation out of the loop: flat values fold as the slice they are.
    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        self.values.fold(init, |acc, value| f(acc, *value))
    }
}

impl<T: NativeType> DoubleEndedIterator for PlPrimitiveValuesIter<'_, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.values.next_back().copied()
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.values.nth_back(n).copied()
    }

    /// Hoists the representation out of the loop, the way [`Iterator::fold`] does.
    #[inline]
    fn rfold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        self.values.rfold(init, |acc, value| f(acc, *value))
    }
}

impl<T: NativeType> ExactSizeIterator for PlPrimitiveValuesIter<'_, T> {
    #[inline]
    fn len(&self) -> usize {
        self.values.len()
    }
}

unsafe impl<T: NativeType> TrustedLen for PlPrimitiveValuesIter<'_, T> {}

/// Iterator over the optional elements of a [`PlPrimitiveArray`](super::PlPrimitiveArray).
#[derive(Clone)]
pub struct PlPrimitiveIter<'a, T: NativeType> {
    values: PlPrimitiveValuesIter<'a, T>,
    validity: ValidityIter<'a>,
}

impl<'a, T: NativeType> PlPrimitiveIter<'a, T> {
    #[inline]
    pub(super) fn new(values: &'a [T], validity: Option<PlBitmapRef<'a>>, length: usize) -> Self {
        assert!(validity.is_none_or(|validity| validity.len() == length));

        Self {
            values: PlPrimitiveValuesIter::new(values, length),
            validity: ValidityIter::new(validity),
        }
    }

    /// The values and the mask that says which of them are elements, to walk in one loop.
    #[inline]
    fn split(self) -> (PlPrimitiveValuesIter<'a, T>, ValidityFold<'a>) {
        (self.values, self.validity.into_mask())
    }
}

crate::impl_optional_iter!([T: NativeType] PlPrimitiveIter<'a, T>, T);
