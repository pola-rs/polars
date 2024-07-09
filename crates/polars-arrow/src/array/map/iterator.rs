use super::MapArray;
use crate::array::Array;
use crate::bitmap::utils::{BitmapIter, ZipValidity};
use crate::trusted_len::TrustedLen;

/// Iterator of values of an [`ListArray`].
#[derive(Clone, Debug)]
pub struct MapValuesIter<'a> {
    array: &'a MapArray,
    index: usize,
    end: usize,
}

impl<'a> MapValuesIter<'a> {
    #[inline]
    pub fn new(array: &'a MapArray) -> Self {
        Self {
            array,
            index: 0,
            end: array.len(),
        }
    }
}

impl<'a> Iterator for MapValuesIter<'a> {
    type Item = Box<dyn Array>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.end {
            return None;
        }
        let old = self.index;
        self.index += 1;
        // SAFETY:
        // self.end is maximized by the length of the array
        Some(unsafe { self.array.value_unchecked(old) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.end - self.index, Some(self.end - self.index))
    }
}

unsafe impl<'a> TrustedLen for MapValuesIter<'a> {}

impl<'a> DoubleEndedIterator for MapValuesIter<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index == self.end {
            None
        } else {
            self.end -= 1;
            // SAFETY:
            // self.end is maximized by the length of the array
            Some(unsafe { self.array.value_unchecked(self.end) })
        }
    }
}

impl<'a> IntoIterator for &'a MapArray {
    type Item = Option<Box<dyn Array>>;
    type IntoIter = ZipValidity<Box<dyn Array>, MapValuesIter<'a>, BitmapIter<'a>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> MapArray {
    /// Returns an iterator of `Option<Box<dyn Array>>`
    pub fn iter(&'a self) -> ZipValidity<Box<dyn Array>, MapValuesIter<'a>, BitmapIter<'a>> {
        ZipValidity::new_with_validity(MapValuesIter::new(self), self.validity())
    }

    /// Returns an iterator of `Box<dyn Array>`
    pub fn values_iter(&'a self) -> MapValuesIter<'a> {
        MapValuesIter::new(self)
    }
}
