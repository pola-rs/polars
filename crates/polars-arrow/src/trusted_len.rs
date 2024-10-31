//! Declares [`TrustedLen`].
use std::iter::Scan;
use std::slice::Iter;

/// An iterator of known, fixed size.
///
/// A trait denoting Rusts' unstable [TrustedLen](https://doc.rust-lang.org/std/iter/trait.TrustedLen.html).
/// This is re-defined here and implemented for some iterators until `std::iter::TrustedLen`
/// is stabilized.
///
/// # Safety
/// This trait must only be implemented when the contract is upheld.
/// Consumers of this trait must inspect Iterator::size_hint()’s upper bound.
pub unsafe trait TrustedLen: Iterator {}

unsafe impl<T> TrustedLen for Iter<'_, T> {}

unsafe impl<'a, I, T: 'a> TrustedLen for std::iter::Copied<I>
where
    I: TrustedLen<Item = &'a T>,
    T: Copy,
{
}
unsafe impl<'a, I, T: 'a> TrustedLen for std::iter::Cloned<I>
where
    I: TrustedLen<Item = &'a T>,
    T: Clone,
{
}

unsafe impl<I> TrustedLen for std::iter::Enumerate<I> where I: TrustedLen {}

unsafe impl<A, B> TrustedLen for std::iter::Zip<A, B>
where
    A: TrustedLen,
    B: TrustedLen,
{
}

unsafe impl<T> TrustedLen for std::slice::ChunksExact<'_, T> {}

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

unsafe impl<T> TrustedLen for &mut dyn TrustedLen<Item = T> {}
unsafe impl<T> TrustedLen for Box<dyn TrustedLen<Item = T> + '_> {}

unsafe impl<B, I: TrustedLen, T: FnMut(I::Item) -> B> TrustedLen for std::iter::Map<I, T> {}

unsafe impl<I: TrustedLen + DoubleEndedIterator> TrustedLen for std::iter::Rev<I> {}

unsafe impl<I: Iterator<Item = J>, J> TrustedLen for TrustMyLength<I, J> {}
unsafe impl<T> TrustedLen for std::ops::Range<T> where std::ops::Range<T>: Iterator {}
unsafe impl<T> TrustedLen for std::ops::RangeInclusive<T> where std::ops::RangeInclusive<T>: Iterator
{}
unsafe impl<A: TrustedLen> TrustedLen for std::iter::StepBy<A> {}

unsafe impl<I, St, F, B> TrustedLen for Scan<I, St, F>
where
    F: FnMut(&mut St, I::Item) -> Option<B>,
    I: TrustedLen,
{
}

unsafe impl<K, V> TrustedLen for hashbrown::hash_map::IntoIter<K, V> {}

#[derive(Clone)]
pub struct TrustMyLength<I: Iterator<Item = J>, J> {
    iter: I,
    len: usize,
}

impl<I, J> TrustMyLength<I, J>
where
    I: Iterator<Item = J>,
{
    /// Create a new `TrustMyLength` iterator
    ///
    /// # Safety
    ///
    /// This is safe if the iterator always has the exact length given by `len`.
    #[inline]
    pub unsafe fn new(iter: I, len: usize) -> Self {
        Self { iter, len }
    }
}

impl<J: Clone> TrustMyLength<std::iter::Take<std::iter::Repeat<J>>, J> {
    /// Create a new `TrustMyLength` iterator that repeats `value` `len` times.
    pub fn new_repeat_n(value: J, len: usize) -> Self {
        // SAFETY: This is always safe since repeat(..).take(n) always repeats exactly `n` times`.
        unsafe { Self::new(std::iter::repeat(value).take(len), len) }
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

    #[inline]
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
