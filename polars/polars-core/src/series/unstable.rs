use crate::prelude::*;
use std::convert::TryFrom;
use std::ptr::NonNull;

/// A wrapper type that should make it a bit more clear that we should not clone Series
#[derive(Debug, Copy, Clone)]
#[cfg(feature = "private")]
pub struct UnstableSeries<'a> {
    // A series containing a single chunk ArrayRef
    // the ArrayRef will be replaced by amortized_iter
    // use with caution!
    container: &'a Series,
    // the ptr to the inner chunk, this saves some ptr chasing
    inner: NonNull<ArrayRef>,
}

/// We don't implement Deref so that the caller is aware of converting to Series
impl AsRef<Series> for UnstableSeries<'_> {
    fn as_ref(&self) -> &Series {
        self.container
    }
}

pub type ArrayBox = Box<dyn Array>;

impl<'a> UnstableSeries<'a> {
    pub fn new(series: &'a Series) -> Self {
        let inner_chunk = series.array_ref(0);
        UnstableSeries {
            container: series,
            inner: NonNull::new(inner_chunk as *const ArrayRef as *mut ArrayRef).unwrap(),
        }
    }

    /// Creates a new `[UnsafeSeries]`
    /// # Safety
    /// Inner chunks must be from `Series` otherwise the dtype may be incorrect and lead to UB.
    pub(crate) unsafe fn new_with_chunk(series: &'a Series, inner_chunk: &ArrayRef) -> Self {
        UnstableSeries {
            container: series,
            inner: NonNull::new(inner_chunk as *const ArrayRef as *mut ArrayRef).unwrap(),
        }
    }

    pub fn deep_clone(&self) -> Series {
        let array_ref = self.container.chunks()[0].clone();
        Series::try_from((self.container.name(), array_ref)).unwrap()
    }

    pub fn swap(&mut self, array: ArrayRef) {
        unsafe { *self.inner.as_mut() = array };
    }
}
