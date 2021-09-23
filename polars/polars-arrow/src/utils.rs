use crate::trusted_len::{FromIteratorReversed, PushUnchecked, TrustedLen};
use arrow::bitmap::Bitmap;
use std::ops::BitAnd;

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

pub fn combine_validities(opt_l: Option<&Bitmap>, opt_r: Option<&Bitmap>) -> Option<Bitmap> {
    match (opt_l, opt_r) {
        (Some(l), Some(r)) => Some(l.bitand(r)),
        (None, Some(r)) => Some(r.clone()),
        (Some(l), None) => (Some(l.clone())),
        (None, None) => None,
    }
}
unsafe impl<I, J> arrow::trusted_len::TrustedLen for TrustMyLength<I, J> where I: Iterator<Item = J> {}

pub trait CustomIterTools: Iterator {
    fn fold_first_<F>(mut self, f: F) -> Option<Self::Item>
    where
        Self: Sized,
        F: FnMut(Self::Item, Self::Item) -> Self::Item,
    {
        let first = self.next()?;
        Some(self.fold(first, f))
    }

    fn trust_my_length(self, length: usize) -> TrustMyLength<Self, Self::Item>
    where
        Self: Sized,
    {
        TrustMyLength::new(self, length)
    }

    fn collect_trusted<T: FromTrustedLenIterator<Self::Item>>(self) -> T
    where
        Self: Sized + TrustedLen,
    {
        FromTrustedLenIterator::from_iter_trusted_length(self)
    }

    fn collect_reversed<T: FromIteratorReversed<Self::Item>>(self) -> T
    where
        Self: Sized + TrustedLen,
    {
        FromIteratorReversed::from_trusted_len_iter_rev(self)
    }
}

pub trait CustomIterToolsSized: Iterator + Sized {}

impl<T: ?Sized> CustomIterTools for T where T: Iterator {}

pub trait FromTrustedLenIterator<A>: Sized {
    fn from_iter_trusted_length<T: IntoIterator<Item = A>>(iter: T) -> Self
    where
        T::IntoIter: TrustedLen;
}

impl<T> FromTrustedLenIterator<T> for Vec<T> {
    fn from_iter_trusted_length<I: IntoIterator<Item = T>>(iter: I) -> Self
    where
        I::IntoIter: TrustedLen,
    {
        let iter = iter.into_iter();
        let len = iter.size_hint().0;
        let mut v = Vec::with_capacity(len);
        v.extend_trusted_len(iter);
        v
    }
}
