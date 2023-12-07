use polars_utils::iter::IntoIteratorCopied;

use super::{MutablePrimitiveArray, PrimitiveArray};
use crate::array::MutableArray;
use crate::bitmap::iterator::TrueIdxIter;
use crate::bitmap::utils::{BitmapIter, ZipValidity};
use crate::bitmap::IntoIter as BitmapIntoIter;
use crate::buffer::IntoIter;
use crate::trusted_len::TrustedLen;
use crate::types::NativeType;

pub struct NonNullValuesIter<'a, T> {
    values: &'a [T],
    idxs: TrueIdxIter<'a>,
}

impl<'a, T: NativeType> NonNullValuesIter<'a, T> {
    pub fn new(arr: &'a PrimitiveArray<T>) -> Self {
        Self {
            values: arr.values(),
            idxs: TrueIdxIter::new(arr.len(), arr.validity()),
        }
    }
}

impl<'a, T: NativeType> Iterator for NonNullValuesIter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.idxs.next().map(|i| unsafe { self.values.get_unchecked(i) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.idxs.size_hint()
    }
}

unsafe impl<'a, T: NativeType> TrustedLen for NonNullValuesIter<'a, T> { }
    

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

impl<T: NativeType> IntoIteratorCopied for PrimitiveArray<T> {
    type OwnedItem = Option<T>;
    type IntoIterCopied = Self::IntoIter;

    fn into_iter(self) -> <Self as IntoIteratorCopied>::IntoIterCopied {
        <Self as IntoIterator>::into_iter(self)
    }
}
