use super::ListArray;
use crate::array::iterator::NonNullValuesIter;
use crate::array::{Array, ArrayAccessor, ArrayValuesIter};
use crate::bitmap::utils::{BitmapIter, ZipValidity};
use crate::offset::Offset;

unsafe impl<'a, O: Offset> ArrayAccessor<'a> for ListArray<O> {
    type Item = Box<dyn Array>;

    #[inline]
    unsafe fn value_unchecked(&'a self, index: usize) -> Self::Item {
        self.value_unchecked(index)
    }

    #[inline]
    fn len(&self) -> usize {
        self.len()
    }
}

/// Iterator of values of a [`ListArray`].
pub type ListValuesIter<'a, O> = ArrayValuesIter<'a, ListArray<O>>;

type ZipIter<'a, O> = ZipValidity<Box<dyn Array>, ListValuesIter<'a, O>, BitmapIter<'a>>;

impl<'a, O: Offset> IntoIterator for &'a ListArray<O> {
    type Item = Option<Box<dyn Array>>;
    type IntoIter = ZipIter<'a, O>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, O: Offset> ListArray<O> {
    /// Returns an iterator of `Option<Box<dyn Array>>`
    pub fn iter(&'a self) -> ZipIter<'a, O> {
        ZipValidity::new_with_validity(ListValuesIter::new(self), self.validity.as_ref())
    }

    /// Returns an iterator of `Box<dyn Array>`
    pub fn values_iter(&'a self) -> ListValuesIter<'a, O> {
        ListValuesIter::new(self)
    }

    /// Returns an iterator of the non-null values `Box<dyn Array>`.
    #[inline]
    pub fn non_null_values_iter(&'a self) -> NonNullValuesIter<'a, ListArray<O>> {
        NonNullValuesIter::new(self, self.validity())
    }
}
