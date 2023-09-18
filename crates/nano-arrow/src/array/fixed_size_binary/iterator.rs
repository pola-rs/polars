use crate::{
    array::MutableArray,
    bitmap::utils::{BitmapIter, ZipValidity},
};

use super::{FixedSizeBinaryArray, MutableFixedSizeBinaryArray};

impl<'a> IntoIterator for &'a FixedSizeBinaryArray {
    type Item = Option<&'a [u8]>;
    type IntoIter = ZipValidity<&'a [u8], std::slice::ChunksExact<'a, u8>, BitmapIter<'a>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> FixedSizeBinaryArray {
    /// constructs a new iterator
    pub fn iter(
        &'a self,
    ) -> ZipValidity<&'a [u8], std::slice::ChunksExact<'a, u8>, BitmapIter<'a>> {
        ZipValidity::new_with_validity(self.values_iter(), self.validity())
    }

    /// Returns iterator over the values of [`FixedSizeBinaryArray`]
    pub fn values_iter(&'a self) -> std::slice::ChunksExact<'a, u8> {
        self.values().chunks_exact(self.size)
    }
}

impl<'a> IntoIterator for &'a MutableFixedSizeBinaryArray {
    type Item = Option<&'a [u8]>;
    type IntoIter = ZipValidity<&'a [u8], std::slice::ChunksExact<'a, u8>, BitmapIter<'a>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> MutableFixedSizeBinaryArray {
    /// constructs a new iterator
    pub fn iter(
        &'a self,
    ) -> ZipValidity<&'a [u8], std::slice::ChunksExact<'a, u8>, BitmapIter<'a>> {
        ZipValidity::new(self.iter_values(), self.validity().map(|x| x.iter()))
    }

    /// Returns iterator over the values of [`MutableFixedSizeBinaryArray`]
    pub fn iter_values(&'a self) -> std::slice::ChunksExact<'a, u8> {
        self.values().chunks_exact(self.size())
    }
}
