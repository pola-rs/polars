mod boolean;
mod push_unchecked;
mod rev;

use crate::utils::TrustMyLength;
use arrow::bitmap::utils::{BitmapIter, ZipValidity};
pub use push_unchecked::*;
pub use rev::FromIteratorReversed;
use std::iter::Scan;
use std::slice::Iter;

/// An iterator of known, fixed size.
/// A trait denoting Rusts' unstable [TrustedLen](https://doc.rust-lang.org/std/iter/trait.TrustedLen.html).
/// This is re-defined here and implemented for some iterators until `std::iter::TrustedLen`
/// is stabilized.
/// *Implementation from Jorge Leitao on Arrow2
/// # Safety
/// length of the iterator must be correct
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

unsafe impl<I: TrustedLen + DoubleEndedIterator> TrustedLen for std::iter::Rev<I> {}

unsafe impl<I: Iterator<Item = J>, J> TrustedLen for TrustMyLength<I, J> {}
unsafe impl<T> TrustedLen for std::ops::Range<T> where std::ops::Range<T>: Iterator {}
unsafe impl TrustedLen for arrow::array::Utf8ValuesIter<'_, i64> {}
unsafe impl<T, I: TrustedLen + Iterator<Item = T>> TrustedLen for ZipValidity<'_, T, I> {}
unsafe impl TrustedLen for BitmapIter<'_> {}
unsafe impl<A: TrustedLen> TrustedLen for std::iter::StepBy<A> {}

unsafe impl<I, St, F, B> TrustedLen for Scan<I, St, F>
where
    F: FnMut(&mut St, I::Item) -> Option<B>,
    I: TrustedLen + Iterator<Item = B>,
{
}

unsafe impl<K, V> TrustedLen for hashbrown::hash_map::IntoIter<K, V> {}
