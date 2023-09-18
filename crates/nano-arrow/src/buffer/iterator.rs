use crate::trusted_len::TrustedLen;

use super::Buffer;

/// This crates' equivalent of [`std::vec::IntoIter`] for [`Buffer`].
#[derive(Debug, Clone)]
pub struct IntoIter<T: Copy> {
    values: Buffer<T>,
    index: usize,
    end: usize,
}

impl<T: Copy> IntoIter<T> {
    /// Creates a new [`Buffer`]
    #[inline]
    pub fn new(values: Buffer<T>) -> Self {
        let end = values.len();
        Self {
            values,
            index: 0,
            end,
        }
    }
}

impl<T: Copy> Iterator for IntoIter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.end {
            return None;
        }
        let old = self.index;
        self.index += 1;
        Some(*unsafe { self.values.get_unchecked(old) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.end - self.index, Some(self.end - self.index))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let new_index = self.index + n;
        if new_index > self.end {
            self.index = self.end;
            None
        } else {
            self.index = new_index;
            self.next()
        }
    }
}

impl<T: Copy> DoubleEndedIterator for IntoIter<T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index == self.end {
            None
        } else {
            self.end -= 1;
            Some(*unsafe { self.values.get_unchecked(self.end) })
        }
    }
}

unsafe impl<T: Copy> TrustedLen for IntoIter<T> {}
