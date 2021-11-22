use crate::chunked_array::object::{ObjectArray, PolarsObject};
use arrow::array::Array;

/// An iterator that returns Some(T) or None, that can be used on any ObjectArray
// Note: This implementation is based on std's [Vec]s' [IntoIter].
pub struct ObjectIter<'a, T: PolarsObject> {
    array: &'a ObjectArray<T>,
    current: usize,
    current_end: usize,
}

impl<'a, T: PolarsObject> ObjectIter<'a, T> {
    /// create a new iterator
    pub fn new(array: &'a ObjectArray<T>) -> Self {
        ObjectIter::<T> {
            array,
            current: 0,
            current_end: array.len(),
        }
    }
}

impl<'a, T: PolarsObject> std::iter::Iterator for ObjectIter<'a, T> {
    type Item = Option<&'a T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current == self.current_end {
            None
        // Safety:
        // Se comment below
        } else if unsafe { self.array.is_null_unchecked(self.current) } {
            self.current += 1;
            Some(None)
        } else {
            let old = self.current;
            self.current += 1;
            // Safety:
            // we just checked bounds in `self.current_end == self.current`
            // this is safe on the premise that this struct is initialized with
            // current = array.len()
            // and that current_end is ever only decremented
            unsafe { Some(Some(self.array.value_unchecked(old))) }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.array.len() - self.current,
            Some(self.array.len() - self.current),
        )
    }
}

impl<'a, T: PolarsObject> std::iter::DoubleEndedIterator for ObjectIter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.current_end == self.current {
            None
        } else {
            self.current_end -= 1;
            Some(if self.array.is_null(self.current_end) {
                None
            } else {
                // Safety:
                // we just checked bounds in `self.current_end == self.current`
                // this is safe on the premise that this struct is initialized with
                // current = array.len()
                // and that current_end is ever only decremented
                unsafe { Some(self.array.value_unchecked(self.current_end)) }
            })
        }
    }
}

/// all arrays have known size.
impl<'a, T: PolarsObject> std::iter::ExactSizeIterator for ObjectIter<'a, T> {}

impl<'a, T: PolarsObject> IntoIterator for &'a ObjectArray<T> {
    type Item = Option<&'a T>;
    type IntoIter = ObjectIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        ObjectIter::<'a, T>::new(self)
    }
}

pub struct OwnedObjectIter<T: PolarsObject> {
    array: ObjectArray<T>,
    current: usize,
    current_end: usize,
}

impl<T: PolarsObject> OwnedObjectIter<T> {
    /// create a new iterator
    pub fn new(array: ObjectArray<T>) -> Self {
        let current_end = array.len();
        OwnedObjectIter::<T> {
            array,
            current: 0,
            current_end,
        }
    }
}

impl<T: PolarsObject> ObjectArray<T> {
    pub(crate) fn into_iter_cloned(self) -> OwnedObjectIter<T> {
        OwnedObjectIter::<T>::new(self)
    }
}
impl<T: PolarsObject> std::iter::Iterator for OwnedObjectIter<T> {
    type Item = Option<T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current == self.current_end {
            None
        // Safety:
        // Se comment below
        } else if unsafe { self.array.is_null_unchecked(self.current) } {
            self.current += 1;
            Some(None)
        } else {
            let old = self.current;
            self.current += 1;
            // Safety:
            // we just checked bounds in `self.current_end == self.current`
            // this is safe on the premise that this struct is initialized with
            // current = array.len()
            // and that current_end is ever only decremented
            unsafe { Some(Some(self.array.value_unchecked(old).clone())) }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.array.len() - self.current,
            Some(self.array.len() - self.current),
        )
    }
}
