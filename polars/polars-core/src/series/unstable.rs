use std::marker::PhantomData;
use std::ptr::NonNull;

use crate::prelude::*;

/// A wrapper type that should make it a bit more clear that we should not clone Series
#[derive(Copy, Clone)]
#[cfg(feature = "private")]
pub struct UnstableSeries<'a> {
    lifetime: PhantomData<&'a Series>,
    // A series containing a single chunk ArrayRef
    // the ArrayRef will be replaced by amortized_iter
    // use with caution!
    container: *mut Series,
    // the ptr to the inner chunk, this saves some ptr chasing
    inner: NonNull<ArrayRef>,
}

/// We don't implement Deref so that the caller is aware of converting to Series
impl AsRef<Series> for UnstableSeries<'_> {
    fn as_ref(&self) -> &Series {
        unsafe { &*self.container }
    }
}

impl AsMut<Series> for UnstableSeries<'_> {
    fn as_mut(&mut self) -> &mut Series {
        unsafe { &mut *self.container }
    }
}

pub type ArrayBox = Box<dyn Array>;

impl<'a> UnstableSeries<'a> {
    pub fn new(series: &'a mut Series) -> Self {
        debug_assert_eq!(series.chunks().len(), 1);
        let container = series as *mut Series;
        let inner_chunk = series.array_ref(0);
        UnstableSeries {
            lifetime: PhantomData,
            container,
            inner: NonNull::new(inner_chunk as *const ArrayRef as *mut ArrayRef).unwrap(),
        }
    }

    /// Creates a new `[UnsafeSeries]`
    /// # Safety
    /// Inner chunks must be from `Series` otherwise the dtype may be incorrect and lead to UB.
    pub(crate) unsafe fn new_with_chunk(series: &'a mut Series, inner_chunk: &ArrayRef) -> Self {
        UnstableSeries {
            lifetime: PhantomData,
            container: series,
            inner: NonNull::new(inner_chunk as *const ArrayRef as *mut ArrayRef).unwrap(),
        }
    }

    pub fn deep_clone(&self) -> Series {
        unsafe {
            let s = &(*self.container);
            debug_assert_eq!(s.chunks().len(), 1);
            let array_ref = s.chunks().get_unchecked(0).clone();
            let name = s.name();
            Series::from_chunks_and_dtype_unchecked(name, vec![array_ref], s.dtype())
        }
    }

    #[inline]
    /// Swaps inner state with the `array`. Prefer `UnstableSeries::with_array` as this
    /// restores the state.
    pub fn swap(&mut self, array: &mut ArrayRef) {
        unsafe { std::mem::swap(self.inner.as_mut(), array) }
        // ensure lengths are correct.
        self.as_mut()._get_inner_mut().compute_len();
    }

    /// Temporary swaps out the array, and restores the original state
    /// when application of the function `f` is done.
    #[inline]
    pub fn with_array<F, T>(&mut self, array: &mut ArrayRef, f: F) -> T
    where
        F: Fn(&UnstableSeries) -> T,
    {
        self.swap(array);
        let out = f(self);
        self.swap(array);
        out
    }
}
