use crate::{
    array::{Array, ArrayAccessor, ArrayValuesIter},
    bitmap::utils::{BitmapIter, ZipValidity},
};

use super::FixedSizeListArray;

unsafe impl<'a> ArrayAccessor<'a> for FixedSizeListArray {
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

/// Iterator of values of a [`FixedSizeListArray`].
pub type FixedSizeListValuesIter<'a> = ArrayValuesIter<'a, FixedSizeListArray>;

type ZipIter<'a> = ZipValidity<Box<dyn Array>, FixedSizeListValuesIter<'a>, BitmapIter<'a>>;

impl<'a> IntoIterator for &'a FixedSizeListArray {
    type Item = Option<Box<dyn Array>>;
    type IntoIter = ZipIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> FixedSizeListArray {
    /// Returns an iterator of `Option<Box<dyn Array>>`
    pub fn iter(&'a self) -> ZipIter<'a> {
        ZipValidity::new_with_validity(FixedSizeListValuesIter::new(self), self.validity())
    }

    /// Returns an iterator of `Box<dyn Array>`
    pub fn values_iter(&'a self) -> FixedSizeListValuesIter<'a> {
        FixedSizeListValuesIter::new(self)
    }
}
