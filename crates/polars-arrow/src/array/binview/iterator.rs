use super::BinaryViewArrayGeneric;
use crate::array::binview::ViewType;
use crate::array::{ArrayAccessor, ArrayValuesIter};
use crate::bitmap::utils::{BitmapIter, ZipValidity};
use crate::trusted_len::TrustedLen;

unsafe impl<'a, T: ViewType + ?Sized> ArrayAccessor<'a> for BinaryViewArrayGeneric<T> {
    type Item = &'a T;

    #[inline]
    unsafe fn value_unchecked(&'a self, index: usize) -> Self::Item {
        self.value_unchecked(index)
    }

    #[inline]
    fn len(&self) -> usize {
        self.views.len()
    }
}

/// Iterator of values of an [`BinaryArray`].
pub type BinaryViewValueIter<'a, T> = ArrayValuesIter<'a, BinaryViewArrayGeneric<T>>;

impl<'a, T: ViewType + ?Sized> IntoIterator for &'a BinaryViewArrayGeneric<T> {
    type Item = Option<&'a T>;
    type IntoIter = ZipValidity<&'a T, BinaryViewValueIter<'a, T>, BitmapIter<'a>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
