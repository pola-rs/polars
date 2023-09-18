use super::UnionArray;
use crate::{scalar::Scalar, trusted_len::TrustedLen};

#[derive(Debug, Clone)]
pub struct UnionIter<'a> {
    array: &'a UnionArray,
    current: usize,
}

impl<'a> UnionIter<'a> {
    #[inline]
    pub fn new(array: &'a UnionArray) -> Self {
        Self { array, current: 0 }
    }
}

impl<'a> Iterator for UnionIter<'a> {
    type Item = Box<dyn Scalar>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current == self.array.len() {
            None
        } else {
            let old = self.current;
            self.current += 1;
            Some(unsafe { self.array.value_unchecked(old) })
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.array.len() - self.current;
        (len, Some(len))
    }
}

impl<'a> IntoIterator for &'a UnionArray {
    type Item = Box<dyn Scalar>;
    type IntoIter = UnionIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> UnionArray {
    /// constructs a new iterator
    #[inline]
    pub fn iter(&'a self) -> UnionIter<'a> {
        UnionIter::new(self)
    }
}

impl<'a> std::iter::ExactSizeIterator for UnionIter<'a> {}

unsafe impl<'a> TrustedLen for UnionIter<'a> {}
