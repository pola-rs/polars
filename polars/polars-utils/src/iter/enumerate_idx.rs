use crate::IdxSize;

/// An iterator that yields the current count and the element during iteration.
///
/// This `struct` is created by the [`enumerate`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`enumerate`]: Iterator::enumerate
/// [`Iterator`]: trait.Iterator.html
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct EnumerateIdx<I> {
    iter: I,
    count: IdxSize,
}

impl<I> Iterator for EnumerateIdx<I>
where
    I: Iterator,
{
    type Item = (IdxSize, <I as Iterator>::Item);

    /// # Overflow Behavior
    ///
    /// The method does no guarding against overflows, so enumerating more than
    /// `idx::MAX` elements either produces the wrong result or panics. If
    /// debug assertions are enabled, a panic is guaranteed.
    ///
    /// # Panics
    ///
    /// Might panic if the index of the element overflows a `idx`.
    #[inline]
    fn next(&mut self) -> Option<(IdxSize, <I as Iterator>::Item)> {
        let a = self.iter.next()?;
        let i = self.count;
        self.count += 1;
        Some((i, a))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<(IdxSize, I::Item)> {
        let a = self.iter.nth(n)?;
        let i = self.count + (n as IdxSize);
        self.count = i + 1;
        Some((i, a))
    }

    #[inline]
    fn count(self) -> usize {
        self.iter.count()
    }
}

impl<I> DoubleEndedIterator for EnumerateIdx<I>
where
    I: ExactSizeIterator + DoubleEndedIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<(IdxSize, <I as Iterator>::Item)> {
        let a = self.iter.next_back()?;
        let len = self.iter.len();
        // Can safely add, `ExactSizeIterator` promises that the number of
        // elements fits into a `usize`.
        Some((self.count + len as IdxSize, a))
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<(IdxSize, <I as Iterator>::Item)> {
        let a = self.iter.nth_back(n)?;
        let len = self.iter.len();
        // Can safely add, `ExactSizeIterator` promises that the number of
        // elements fits into a `usize`.
        Some((self.count + len as IdxSize, a))
    }
}

impl<I> ExactSizeIterator for EnumerateIdx<I>
where
    I: ExactSizeIterator,
{
    fn len(&self) -> usize {
        self.iter.len()
    }
}

pub trait EnumerateIdxTrait: Iterator {
    fn enumerate_idx(self) -> EnumerateIdx<Self>
    where
        Self: Sized,
    {
        EnumerateIdx {
            iter: self,
            count: 0,
        }
    }
}

impl<T: ?Sized> EnumerateIdxTrait for T where T: Iterator {}
