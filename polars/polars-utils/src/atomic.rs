use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::atomic::AtomicUsize;

#[derive(Clone)]
/// A utility to create a sharable counter
/// This does not implement drop as the user
/// needs to decide when to drop it. Which is likely
/// the moment the last thread is finished.
pub struct SyncCounter {
    count: NonNull<AtomicUsize>,
}

impl Default for SyncCounter {
    fn default() -> Self {
        Self::new(0)
    }
}

impl SyncCounter {
    pub fn new(value: usize) -> Self {
        let count = Box::new(AtomicUsize::new(value));
        let ptr = Box::leak(count);

        // leak a box so that we get a pointer that remains valid until we drop Self
        let count = unsafe { NonNull::new_unchecked(ptr) };
        SyncCounter { count }
    }

    /// # Safety
    /// This will deref the pointer and after this all autoderef will be invalid.
    pub unsafe fn manual_drop(&mut self) {
        // recreate the box and drop it
        unsafe { drop(Box::from_raw(self.count.as_ptr())) };
    }
}

impl Deref for SyncCounter {
    type Target = AtomicUsize;

    fn deref(&self) -> &Self::Target {
        unsafe { self.count.as_ref() }
    }
}

impl DerefMut for SyncCounter {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.count.as_mut() }
    }
}

unsafe impl Sync for SyncCounter {}
unsafe impl Send for SyncCounter {}
