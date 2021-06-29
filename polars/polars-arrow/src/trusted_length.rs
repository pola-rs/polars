use std::slice::Iter;

/// An iterator of known, fixed size.
/// A trait denoting Rusts' unstable [TrustedLen](https://doc.rust-lang.org/std/iter/trait.TrustedLen.html).
/// This is re-defined here and implemented for some iterators until `std::iter::TrustedLen`
/// is stabilized.
pub unsafe trait TrustedLen: Iterator {}

unsafe impl<T> TrustedLen for Iter<'_, T> {}

unsafe impl<B, I: TrustedLen, T: FnMut(I::Item) -> B> TrustedLen for std::iter::Map<I, T> {}

unsafe impl<'a, I, T: 'a> TrustedLen for std::iter::Copied<I>
where
    I: TrustedLen<Item = &'a T>,
    T: Copy,
{
}

unsafe impl<I> TrustedLen for std::iter::Enumerate<I> where I: TrustedLen {}

unsafe impl<A, B> TrustedLen for std::iter::Zip<A, B>
where
    A: TrustedLen,
    B: TrustedLen,
{
}

unsafe impl<T> TrustedLen for std::slice::Windows<'_, T> {}

unsafe impl<A, B> TrustedLen for std::iter::Chain<A, B>
where
    A: TrustedLen,
    B: TrustedLen<Item = A::Item>,
{
}

unsafe impl<T> TrustedLen for std::iter::Once<T> {}

unsafe impl<T> TrustedLen for std::vec::IntoIter<T> {}

unsafe impl<A: Clone> TrustedLen for std::iter::Repeat<A> {}
unsafe impl<A, F: FnMut() -> A> TrustedLen for std::iter::RepeatWith<F> {}
unsafe impl<A: TrustedLen> TrustedLen for std::iter::Take<A> {}

/// Now add a trait so we can check if an iterators' length can be trusted
pub unsafe trait IsTrustedLen {
    fn is_trusted_len() -> bool {
        true
    }
}

/// Utility struct only used to check if an iterators length can be trusted
pub struct CheckIsTrusted<I>(pub I);

impl<I: IsTrustedLen> CheckIsTrusted<I> {
    pub fn is_trusted_len(&self) -> bool {
        I::is_trusted_len()
    }
}

unsafe impl<T> IsTrustedLen for CheckIsTrusted<Iter<'_, T>> {}
unsafe impl<B, I: TrustedLen, T: FnMut(I::Item) -> B> IsTrustedLen
    for CheckIsTrusted<std::iter::Map<I, T>>
{
}

unsafe impl<'a, I, T: 'a> IsTrustedLen for CheckIsTrusted<std::iter::Copied<I>>
where
    I: TrustedLen<Item = &'a T>,
    T: Copy,
{
}

unsafe impl<I> IsTrustedLen for CheckIsTrusted<std::iter::Enumerate<I>> where I: TrustedLen {}

unsafe impl<A, B> IsTrustedLen for CheckIsTrusted<std::iter::Zip<A, B>>
where
    A: TrustedLen,
    B: TrustedLen,
{
}

unsafe impl<T> IsTrustedLen for CheckIsTrusted<std::slice::Windows<'_, T>> {}

unsafe impl<A, B> IsTrustedLen for CheckIsTrusted<std::iter::Chain<A, B>>
where
    A: TrustedLen,
    B: TrustedLen<Item = A::Item>,
{
}

unsafe impl<T> IsTrustedLen for CheckIsTrusted<std::iter::Once<T>> {}

unsafe impl<T> IsTrustedLen for CheckIsTrusted<std::vec::IntoIter<T>> {}

unsafe impl<A: Clone> IsTrustedLen for CheckIsTrusted<std::iter::Repeat<A>> {}
unsafe impl<A, F: FnMut() -> A> IsTrustedLen for CheckIsTrusted<std::iter::RepeatWith<F>> {}
unsafe impl<A: TrustedLen> IsTrustedLen for CheckIsTrusted<std::iter::Take<A>> {}
unsafe impl<I, J> IsTrustedLen for CheckIsTrusted<TrustMyLength<I, J>> where I: Iterator<Item = J> {}

/// An Iterator whose length can be trusted
pub struct TrustMyLength<I: Iterator<Item = J>, J> {
    iter: I,
    len: usize,
}

impl<I, J> TrustMyLength<I, J>
where
    I: Iterator<Item = J>,
{
    #[inline]
    pub fn new(iter: I, len: usize) -> Self {
        Self { iter, len }
    }
}

impl<I, J> Iterator for TrustMyLength<I, J>
where
    I: Iterator<Item = J>,
{
    type Item = J;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<I, J> ExactSizeIterator for TrustMyLength<I, J> where I: Iterator<Item = J> {}

impl<I, J> DoubleEndedIterator for TrustMyLength<I, J>
where
    I: Iterator<Item = J> + DoubleEndedIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}

/// Implement for all other iterators and return false for them
unsafe impl<T: ?Sized> IsTrustedLen for T
where
    T: Iterator,
{
    fn is_trusted_len() -> bool {
        false
    }
}
