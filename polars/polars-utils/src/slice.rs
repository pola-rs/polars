use core::slice::SliceIndex;
use std::cmp::Ordering;

pub trait Extrema<T> {
    fn min_value(&self) -> Option<&T>;
    fn max_value(&self) -> Option<&T>;
}

impl<T: PartialOrd> Extrema<T> for [T] {
    fn min_value(&self) -> Option<&T> {
        self.iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
    }

    fn max_value(&self) -> Option<&T> {
        self.iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
    }
}

pub trait GetSaferUnchecked<T> {
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*
    /// even if the resulting reference is not used.
    unsafe fn get_unchecked_release<I>(&self, index: I) -> &<I as SliceIndex<[T]>>::Output
    where
        I: SliceIndex<[T]>;

    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*
    /// even if the resulting reference is not used.
    unsafe fn get_unchecked_release_mut<I>(
        &mut self,
        index: I,
    ) -> &mut <I as SliceIndex<[T]>>::Output
    where
        I: SliceIndex<[T]>;
}

impl<T> GetSaferUnchecked<T> for [T] {
    #[inline(always)]
    unsafe fn get_unchecked_release<I>(&self, index: I) -> &<I as SliceIndex<[T]>>::Output
    where
        I: SliceIndex<[T]>,
    {
        if cfg!(debug_assertions) {
            &self[index]
        } else {
            self.get_unchecked(index)
        }
    }

    #[inline(always)]
    unsafe fn get_unchecked_release_mut<I>(
        &mut self,
        index: I,
    ) -> &mut <I as SliceIndex<[T]>>::Output
    where
        I: SliceIndex<[T]>,
    {
        if cfg!(debug_assertions) {
            &mut self[index]
        } else {
            self.get_unchecked_mut(index)
        }
    }
}
