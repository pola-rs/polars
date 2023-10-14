use crate::trusted_len::TrustedLen;

mod private {
    pub trait Sealed {}

    impl<'a, T: super::ArrayAccessor<'a>> Sealed for T {}
}

/// Sealed trait representing assess to a value of an array.
/// # Safety
/// Implementers of this trait guarantee that
/// `value_unchecked` is safe when called up to `len`
pub unsafe trait ArrayAccessor<'a>: private::Sealed {
    type Item: 'a;
    unsafe fn value_unchecked(&'a self, index: usize) -> Self::Item;
    fn len(&self) -> usize;
}

/// Iterator of values of an [`ArrayAccessor`].
#[derive(Debug, Clone)]
pub struct ArrayValuesIter<'a, A: ArrayAccessor<'a>> {
    array: &'a A,
    index: usize,
    end: usize,
}

impl<'a, A: ArrayAccessor<'a>> ArrayValuesIter<'a, A> {
    /// Creates a new [`ArrayValuesIter`]
    #[inline]
    pub fn new(array: &'a A) -> Self {
        Self {
            array,
            index: 0,
            end: array.len(),
        }
    }
}

impl<'a, A: ArrayAccessor<'a>> Iterator for ArrayValuesIter<'a, A> {
    type Item = A::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.end {
            return None;
        }
        let old = self.index;
        self.index += 1;
        Some(unsafe { self.array.value_unchecked(old) })
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

impl<'a, A: ArrayAccessor<'a>> DoubleEndedIterator for ArrayValuesIter<'a, A> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index == self.end {
            None
        } else {
            self.end -= 1;
            Some(unsafe { self.array.value_unchecked(self.end) })
        }
    }
}

unsafe impl<'a, A: ArrayAccessor<'a>> TrustedLen for ArrayValuesIter<'a, A> {}
impl<'a, A: ArrayAccessor<'a>> ExactSizeIterator for ArrayValuesIter<'a, A> {}
