pub mod boolean;

use crate::utils::TrustMyLength;
use crate::vec::AlignedVec;
use arrow::buffer::MutableBuffer;
use arrow::datatypes::{ArrowNativeType, ArrowPrimitiveType};
use arrow::util::bit_util;
use std::slice::Iter;

/// An iterator of known, fixed size.
/// A trait denoting Rusts' unstable [TrustedLen](https://doc.rust-lang.org/std/iter/trait.TrustedLen.html).
/// This is re-defined here and implemented for some iterators until `std::iter::TrustedLen`
/// is stabilized.
/// *Implementation from Jorge Leitao on Arrow2
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
unsafe impl<T: ArrowPrimitiveType> TrustedLen for arrow::array::PrimitiveIter<'_, T> {}
unsafe impl TrustedLen for arrow::array::GenericStringIter<'_, i64> {}
unsafe impl TrustedLen for arrow::array::BooleanIter<'_> {}

///
/// unzips an iterator over an Option<T> into a given validity buffer and value buffer
///
/// # Safety
/// - iterator must be TrustedLen
/// - values length must have additional capacity to fit the iterators length
/// - validity length must have additional capacity to fit the iterators length
#[inline]
pub unsafe fn trusted_len_unzip_extend<I, P, T>(
    iterator: I,
    values: &mut AlignedVec<T>,
    validity: &mut MutableBuffer,
) where
    T: ArrowNativeType,
    P: std::borrow::Borrow<Option<T>>,
    I: Iterator<Item = P>,
{
    let (_, upper) = iterator.size_hint();
    let upper = upper.expect("trusted_len_unzip requires an upper limit");
    let offset = values.len();

    let dst_validity = validity.as_mut_ptr();
    let mut dst = values.as_mut_ptr() as *mut T;
    dst = dst.add(offset);
    let start = dst;
    for (i, item) in iterator.enumerate() {
        let item = item.borrow();
        if let Some(item) = item {
            std::ptr::write(dst, *item);
            bit_util::set_bit_raw(dst_validity, i + offset);
        } else {
            std::ptr::write(dst, T::default());
        }
        dst = dst.add(1);
    }
    assert_eq!(
        dst.offset_from(start) as usize,
        upper,
        "Trusted iterator length was not accurately reported"
    );
    values.set_len(values.len() + upper)
}

pub trait PushUnchecked<T> {
    /// Will push an item and not check if there is enough capacity
    ///
    /// # Safety
    /// Caller must ensure the array has enough capacity to hold `T`.
    unsafe fn push_unchecked(&mut self, value: T);

    /// Will push an item and not check if there is enough capacity nor update the array's lenght
    /// # Safety
    /// Caller must ensure the array has enough capacity to hold `T`.
    /// Caller must update the length when its done updating the vector.
    unsafe fn push_unchecked_no_len_set(&mut self, value: T);

    /// Extend the array with an iterator who's length can be trusted
    fn extend_trusted_len<I: IntoIterator<Item = T> + TrustedLen>(&mut self, iter: I);
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
