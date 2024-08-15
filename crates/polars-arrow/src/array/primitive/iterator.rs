use super::{MutablePrimitiveArray, PrimitiveArray};
use crate::array::{ArrayAccessor, MutableArray};
use crate::bitmap::utils::{BitmapIter, ZipValidity};
use crate::bitmap::IntoIter as BitmapIntoIter;
use crate::buffer::IntoIter;
use crate::types::NativeType;

unsafe impl<'a, T: NativeType> ArrayAccessor<'a> for [T] {
    type Item = T;

    #[inline]
    unsafe fn value_unchecked(&'a self, index: usize) -> Self::Item {
        *self.get_unchecked(index)
    }

    #[inline]
    fn len(&self) -> usize {
        (*self).len()
    }
}

impl<T: NativeType> IntoIterator for PrimitiveArray<T> {
    type Item = Option<T>;
    type IntoIter = ZipValidity<T, IntoIter<T>, BitmapIntoIter>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let (_, values, validity) = self.into_inner();
        let values = values.into_iter();
        let validity =
            validity.and_then(|validity| (validity.unset_bits() > 0).then(|| validity.into_iter()));
        ZipValidity::new(values, validity)
    }
}

impl<'a, T: NativeType> IntoIterator for &'a PrimitiveArray<T> {
    type Item = Option<&'a T>;
    type IntoIter = ZipValidity<&'a T, std::slice::Iter<'a, T>, BitmapIter<'a>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T: NativeType> MutablePrimitiveArray<T> {
    /// Returns an iterator over `Option<T>`
    #[inline]
    pub fn iter(&'a self) -> ZipValidity<&'a T, std::slice::Iter<'a, T>, BitmapIter<'a>> {
        ZipValidity::new(
            self.values().iter(),
            self.validity().as_ref().map(|x| x.iter()),
        )
    }

    /// Returns an iterator of `T`
    #[inline]
    pub fn values_iter(&'a self) -> std::slice::Iter<'a, T> {
        self.values().iter()
    }
}
