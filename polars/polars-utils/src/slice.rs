use core::slice::SliceIndex;

pub trait GetSaferUnchecked<T> {
    unsafe fn get_unchecked_release<I>(&self, index: I) -> &<I as SliceIndex<[T]>>::Output
    where
        I: SliceIndex<[T]>;

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
