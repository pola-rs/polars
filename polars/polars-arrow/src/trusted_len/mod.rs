mod boolean;
mod rev;

use crate::utils::{FromTrustedLenIterator, TrustMyLength};
use arrow::bitmap::utils::{BitmapIter, ZipValidity};
use arrow::buffer::MutableBuffer;
use arrow::types::NativeType;
pub use rev::FromIteratorReversed;
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

impl<T: arrow::types::NativeType> FromTrustedLenIterator<T> for MutableBuffer<T> {
    fn from_iter_trusted_length<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        // Safety:
        // Guarded by trait system
        unsafe { MutableBuffer::from_trusted_len_iter_unchecked(iter) }
    }
}

pub trait PushUnchecked<T> {
    /// Will push an item and not check if there is enough capacity
    ///
    /// # Safety
    /// Caller must ensure the array has enough capacity to hold `T`.
    unsafe fn push_unchecked(&mut self, value: T);

    /// Will push an item and not check if there is enough capacity nor update the array's length
    /// # Safety
    /// Caller must ensure the array has enough capacity to hold `T`.
    /// Caller must update the length when its done updating the vector.
    unsafe fn push_unchecked_no_len_set(&mut self, value: T);

    /// Extend the array with an iterator who's length can be trusted
    fn extend_trusted_len<I: IntoIterator<Item = T> + TrustedLen>(&mut self, iter: I);
}

impl<T: NativeType> PushUnchecked<T> for MutableBuffer<T> {
    unsafe fn push_unchecked(&mut self, value: T) {
        let end = self.as_mut_ptr().add(self.len());
        std::ptr::write(end, value);
        self.set_len(self.len() + 1);
    }

    unsafe fn push_unchecked_no_len_set(&mut self, value: T) {
        let end = self.as_mut_ptr().add(self.len());
        std::ptr::write(end, value);
    }

    fn extend_trusted_len<I: IntoIterator<Item = T> + TrustedLen>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let upper = iter.size_hint().1.expect("must have an upper bound");
        self.reserve(upper);

        unsafe {
            let mut dst = self.as_mut_ptr().add(self.len());
            for value in iter {
                std::ptr::write(dst, value);
                dst = dst.add(1)
            }
            self.set_len(self.len() + upper)
        }
    }
}

impl<T> PushUnchecked<T> for Vec<T> {
    #[inline]
    unsafe fn push_unchecked(&mut self, value: T) {
        let end = self.as_mut_ptr().add(self.len());
        std::ptr::write(end, value);
        self.set_len(self.len() + 1);
    }

    #[inline]
    unsafe fn push_unchecked_no_len_set(&mut self, value: T) {
        let end = self.as_mut_ptr().add(self.len());
        std::ptr::write(end, value);
    }

    #[inline]
    fn extend_trusted_len<I: IntoIterator<Item = T> + TrustedLen>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let upper = iter.size_hint().1.expect("must have an upper bound");
        self.reserve(upper);

        unsafe {
            let mut dst = self.as_mut_ptr().add(self.len());
            for value in iter {
                std::ptr::write(dst, value);
                dst = dst.add(1)
            }
            self.set_len(self.len() + upper)
        }
    }
}
