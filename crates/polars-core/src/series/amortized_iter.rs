use std::rc::Rc;

use crate::prelude::*;

/// A [`Series`] that amortizes a few allocations during iteration.
#[derive(Clone)]
pub struct AmortSeries {
    container: Rc<Series>,
}

/// We don't implement Deref so that the caller is aware of converting to Series
impl AsRef<Series> for AmortSeries {
    fn as_ref(&self) -> &Series {
        self.container.as_ref()
    }
}

pub type ArrayBox = Box<dyn Array>;

impl AmortSeries {
    pub fn new(container: Rc<Series>) -> Self {
        debug_assert_eq!(container.chunks().len(), 1);
        AmortSeries { container }
    }

    pub fn deep_clone(&self) -> Series {
        unsafe {
            let s = &(*self.container);
            debug_assert_eq!(s.chunks().len(), 1);
            let array_ref = s.chunks().get_unchecked(0).clone();
            let name = s.name().clone();
            Series::from_chunks_and_dtype_unchecked(name, vec![array_ref], s.dtype())
        }
    }

    #[inline]
    /// Swaps inner state with the `array`. Prefer `AmortSeries::with_array` as this
    /// restores the state.
    /// # Safety
    /// This swaps an underlying pointer that might be hold by other cloned series.
    pub unsafe fn swap(&mut self, array: &mut ArrayRef) {
        let inner = self.container.array_ref(0) as *const ArrayRef as *mut ArrayRef;
        let inner = inner.as_mut().unwrap();
        std::mem::swap(inner, array);

        // ensure lengths are correct.
        unsafe {
            let ptr = Rc::as_ptr(&self.container) as *mut Series;
            (*ptr)._get_inner_mut().compute_len();
        }
    }

    /// Temporary swaps out the array, and restores the original state
    /// when application of the function `f` is done.
    ///
    /// # Safety
    /// Array must be from `Series` physical dtype.
    #[inline]
    pub unsafe fn with_array<F, T>(&mut self, array: &mut ArrayRef, f: F) -> T
    where
        F: Fn(&AmortSeries) -> T,
    {
        unsafe {
            self.swap(array);
            let out = f(self);
            self.swap(array);
            out
        }
    }
}

// SAFETY:
// type must be matching
pub(crate) unsafe fn unstable_series_container_and_ptr(
    name: PlSmallStr,
    inner_values: ArrayRef,
    iter_dtype: &DataType,
) -> (Series, *mut ArrayRef) {
    let series_container = {
        let mut s = Series::from_chunks_and_dtype_unchecked(name, vec![inner_values], iter_dtype);
        s.clear_flags();
        s
    };

    let ptr = series_container.array_ref(0) as *const ArrayRef as *mut ArrayRef;
    (series_container, ptr)
}
