use super::*;

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

    fn from_trusted_len_iter<I: IntoIterator<Item = T> + TrustedLen>(iter: I) -> Self;
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

    fn from_trusted_len_iter<I: IntoIterator<Item = T> + TrustedLen>(iter: I) -> Self {
        let mut v = vec![];
        v.extend_trusted_len(iter);
        v
    }
}
