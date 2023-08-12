use core::slice::SliceIndex;
use std::cmp::Ordering;
use std::mem::MaybeUninit;

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

pub trait SortedSlice<T> {
    fn is_sorted_ascending(&self) -> bool;
}

impl<T: PartialOrd + Copy> SortedSlice<T> for [T] {
    fn is_sorted_ascending(&self) -> bool {
        if self.is_empty() {
            true
        } else {
            let mut previous = self[0];
            let mut sorted = true;

            // don't early stop or branch
            // so it autovectorizes
            for &v in &self[1..] {
                sorted &= previous <= v;
                previous = v;
            }
            sorted
        }
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

pub trait Slice2Uninit<T> {
    fn as_uninit(&self) -> &[MaybeUninit<T>];
}

impl<T> Slice2Uninit<T> for [T] {
    #[inline]
    fn as_uninit(&self) -> &[MaybeUninit<T>] {
        unsafe { std::slice::from_raw_parts(self.as_ptr() as *const MaybeUninit<T>, self.len()) }
    }
}
