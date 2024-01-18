use crate::trusted_len::TrustedLen;

pub trait TrustedLenPush<T> {
    /// Will push an item and not check if there is enough capacity.
    ///
    /// # Safety
    /// Caller must ensure the array has enough capacity to hold `T`.
    unsafe fn push_unchecked(&mut self, value: T);

    /// Extend the array with an iterator who's length can be trusted.
    fn extend_trusted_len<I: IntoIterator<Item = T, IntoIter = J>, J: TrustedLen>(
        &mut self,
        iter: I,
    ) {
        unsafe { self.extend_trusted_len_unchecked(iter) }
    }

    /// # Safety
    /// Caller must ensure the iterators reported length is correct.
    unsafe fn extend_trusted_len_unchecked<I: IntoIterator<Item = T>>(&mut self, iter: I);

    /// # Safety
    /// Caller must ensure the iterators reported length is correct.
    unsafe fn try_extend_trusted_len_unchecked<E, I: IntoIterator<Item = Result<T, E>>>(
        &mut self,
        iter: I,
    ) -> Result<(), E>;

    fn from_trusted_len_iter<I: IntoIterator<Item = T, IntoIter = J>, J: TrustedLen>(
        iter: I,
    ) -> Self
    where
        Self: Sized,
    {
        unsafe { Self::from_trusted_len_iter_unchecked(iter) }
    }
    /// # Safety
    /// Caller must ensure the iterators reported length is correct.
    unsafe fn from_trusted_len_iter_unchecked<I: IntoIterator<Item = T>>(iter: I) -> Self;

    fn try_from_trusted_len_iter<
        E,
        I: IntoIterator<Item = Result<T, E>, IntoIter = J>,
        J: TrustedLen,
    >(
        iter: I,
    ) -> Result<Self, E>
    where
        Self: Sized,
    {
        unsafe { Self::try_from_trusted_len_iter_unchecked(iter) }
    }
    /// # Safety
    /// Caller must ensure the iterators reported length is correct.
    unsafe fn try_from_trusted_len_iter_unchecked<E, I: IntoIterator<Item = Result<T, E>>>(
        iter: I,
    ) -> Result<Self, E>
    where
        Self: Sized;
}

impl<T> TrustedLenPush<T> for Vec<T> {
    #[inline]
    unsafe fn push_unchecked(&mut self, value: T) {
        debug_assert!(self.capacity() > self.len());
        let end = self.as_mut_ptr().add(self.len());
        std::ptr::write(end, value);
        self.set_len(self.len() + 1);
    }

    #[inline]
    unsafe fn extend_trusted_len_unchecked<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let upper = iter.size_hint().1.expect("must have an upper bound");
        self.reserve(upper);

        let mut dst = self.as_mut_ptr().add(self.len());
        for value in iter {
            std::ptr::write(dst, value);
            dst = dst.add(1)
        }
        self.set_len(self.len() + upper)
    }

    unsafe fn try_extend_trusted_len_unchecked<E, I: IntoIterator<Item = Result<T, E>>>(
        &mut self,
        iter: I,
    ) -> Result<(), E> {
        let iter = iter.into_iter();
        let upper = iter.size_hint().1.expect("must have an upper bound");
        self.reserve(upper);

        let mut dst = self.as_mut_ptr().add(self.len());
        for value in iter {
            std::ptr::write(dst, value?);
            dst = dst.add(1)
        }
        self.set_len(self.len() + upper);
        Ok(())
    }

    #[inline]
    unsafe fn from_trusted_len_iter_unchecked<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut v = vec![];
        v.extend_trusted_len_unchecked(iter);
        v
    }

    unsafe fn try_from_trusted_len_iter_unchecked<E, I: IntoIterator<Item = Result<T, E>>>(
        iter: I,
    ) -> Result<Self, E>
    where
        Self: Sized,
    {
        let mut v = vec![];
        v.try_extend_trusted_len_unchecked(iter)?;
        Ok(v)
    }
}
