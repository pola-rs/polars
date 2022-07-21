use crate::prelude::*;
use std::convert::TryFrom;
use std::marker::PhantomData;
use std::ptr::NonNull;

/// A wrapper type that should make it a bit more clear that we should not clone Series
#[derive(Debug, Copy, Clone)]
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
        let container = series as *mut Series;
        let inner_chunk = series.array_ref(0);
        UnstableSeries {
            lifetime: PhantomData,
            container: container,
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
        let array_ref = unsafe { (*self.container).chunks() }[0].clone();
        let name = unsafe { (*self.container).name() };
        Series::try_from((name, array_ref)).unwrap()
    }

    pub fn swap(&mut self, array: ArrayRef) {
        unsafe { *self.inner.as_mut() = array };
    }
}
