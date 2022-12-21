/// Utility that allows use to send pointers to another thread.
/// This is better than going through `usize` as MIRI can follow these.
#[derive(Copy, Clone)]
pub struct SyncPtr<T>(*mut T);

impl<T> SyncPtr<T> {
    /// # Safety
    ///
    /// This will make a pointer sync and send.
    /// Ensure that you don't break aliasing rules.
    pub unsafe fn new(ptr: *mut T) -> Self {
        Self(ptr)
    }

    #[inline(always)]
    pub fn get(self) -> *mut T {
        self.0
    }
}

unsafe impl<T> Sync for SyncPtr<T> {}
unsafe impl<T> Send for SyncPtr<T> {}
