use num_traits::{FromPrimitive, One, Zero};

/// An iterator that yields the current count and the element during iteration.
///
/// This `struct` is created by the [`enumerate`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`enumerate`]: Iterator::enumerate
/// [`Iterator`]: trait.Iterator.html
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct EnumerateIdx<I, IdxType> {
    iter: I,
    count: IdxType,
}

impl<I, IdxType: Zero> EnumerateIdx<I, IdxType> {
    pub fn new(iter: I) -> Self {
        Self {
            iter,
            count: IdxType::zero(),
        }
    }
}

impl<I, IdxType> Iterator for EnumerateIdx<I, IdxType>
where
    I: Iterator,
    IdxType: std::ops::Add<Output = IdxType> + FromPrimitive + std::ops::AddAssign + One + Copy,
{
    type Item = (IdxType, <I as Iterator>::Item);

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
    fn next(&mut self) -> Option<Self::Item> {
        let a = self.iter.next()?;
        let i = self.count;
        self.count += IdxType::one();
        Some((i, a))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let a = self.iter.nth(n)?;
        let i = self.count + IdxType::from_usize(n).unwrap();
        self.count = i + IdxType::one();
        Some((i, a))
    }

    #[inline]
    fn count(self) -> usize {
        self.iter.count()
    }
}

impl<I, IdxType> DoubleEndedIterator for EnumerateIdx<I, IdxType>
where
    I: ExactSizeIterator + DoubleEndedIterator,
    IdxType: std::ops::Add<Output = IdxType> + FromPrimitive + std::ops::AddAssign + One + Copy,
{
    #[inline]
    fn next_back(&mut self) -> Option<(IdxType, <I as Iterator>::Item)> {
        let a = self.iter.next_back()?;
        let len = IdxType::from_usize(self.iter.len()).unwrap();
        // Can safely add, `ExactSizeIterator` promises that the number of
        // elements fits into a `usize`.
        Some((self.count + len, a))
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<(IdxType, <I as Iterator>::Item)> {
        let a = self.iter.nth_back(n)?;
        let len = IdxType::from_usize(self.iter.len()).unwrap();
        // Can safely add, `ExactSizeIterator` promises that the number of
        // elements fits into a `usize`.
        Some((self.count + len, a))
    }
}

impl<I, IdxType> ExactSizeIterator for EnumerateIdx<I, IdxType>
where
    I: ExactSizeIterator,
    IdxType: std::ops::Add<Output = IdxType> + FromPrimitive + std::ops::AddAssign + One + Copy,
{
    fn len(&self) -> usize {
        self.iter.len()
    }
}
