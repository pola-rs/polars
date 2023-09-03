/// Utility that allows use to send pointers to another thread.
/// This is better than going through `usize` as MIRI can follow these.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct SyncPtr<T>(*mut T);

impl<T> SyncPtr<T> {
    /// # Safety
    ///
    /// This will make a pointer sync and send.
    /// Ensure that you don't break aliasing rules.
    pub unsafe fn new(ptr: *mut T) -> Self {
        Self(ptr)
    }

    /// # Safety
    ///
    /// This will make a pointer sync and send.
    /// Ensure that you don't break aliasing rules.
    pub unsafe fn from_const(ptr: *const T) -> Self {
        Self(ptr as *mut T)
    }

    pub fn new_null() -> Self {
        Self(std::ptr::null_mut())
    }

    #[inline(always)]
    pub fn get(&self) -> *mut T {
        self.0
    }

    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// # Safety
    /// Derefs a raw pointer, no guarantees whatsoever.
    pub unsafe fn deref_unchecked(&self) -> &'static T {
        &*(self.0 as *const T)
    }
}

unsafe impl<T> Sync for SyncPtr<T> {}
unsafe impl<T> Send for SyncPtr<T> {}
