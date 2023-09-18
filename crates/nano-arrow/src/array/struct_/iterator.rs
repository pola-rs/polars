use crate::{
    bitmap::utils::{BitmapIter, ZipValidity},
    scalar::{new_scalar, Scalar},
    trusted_len::TrustedLen,
};

use super::StructArray;

pub struct StructValueIter<'a> {
    array: &'a StructArray,
    index: usize,
    end: usize,
}

impl<'a> StructValueIter<'a> {
    #[inline]
    pub fn new(array: &'a StructArray) -> Self {
        Self {
            array,
            index: 0,
            end: array.len(),
        }
    }
}

impl<'a> Iterator for StructValueIter<'a> {
    type Item = Vec<Box<dyn Scalar>>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.end {
            return None;
        }
        let old = self.index;
        self.index += 1;

        // Safety:
        // self.end is maximized by the length of the array
        Some(
            self.array
                .values()
                .iter()
                .map(|v| new_scalar(v.as_ref(), old))
                .collect(),
        )
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.end - self.index, Some(self.end - self.index))
    }
}

unsafe impl<'a> TrustedLen for StructValueIter<'a> {}

impl<'a> DoubleEndedIterator for StructValueIter<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index == self.end {
            None
        } else {
            self.end -= 1;

            // Safety:
            // self.end is maximized by the length of the array
            Some(
                self.array
                    .values()
                    .iter()
                    .map(|v| new_scalar(v.as_ref(), self.end))
                    .collect(),
            )
        }
    }
}

type ValuesIter<'a> = StructValueIter<'a>;
type ZipIter<'a> = ZipValidity<Vec<Box<dyn Scalar>>, ValuesIter<'a>, BitmapIter<'a>>;

impl<'a> IntoIterator for &'a StructArray {
    type Item = Option<Vec<Box<dyn Scalar>>>;
    type IntoIter = ZipIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> StructArray {
    /// Returns an iterator of `Option<Box<dyn Array>>`
    pub fn iter(&'a self) -> ZipIter<'a> {
        ZipValidity::new_with_validity(StructValueIter::new(self), self.validity())
    }

    /// Returns an iterator of `Box<dyn Array>`
    pub fn values_iter(&'a self) -> ValuesIter<'a> {
        StructValueIter::new(self)
    }
}
