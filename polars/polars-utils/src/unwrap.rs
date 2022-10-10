use std::fmt::Debug;

pub trait UnwrapUncheckedRelease<T> {
    /// # Safety
    ///
    /// unwrap without checking the invariant
    unsafe fn unwrap_unchecked_release(self) -> T;
}

impl<T> UnwrapUncheckedRelease<T> for Option<T> {
    unsafe fn unwrap_unchecked_release(self) -> T {
        if cfg!(debug_assertions) {
            self.unwrap()
        } else {
            self.unwrap_unchecked()
        }
    }
}

impl<T, E: Debug> UnwrapUncheckedRelease<T> for Result<T, E> {
    unsafe fn unwrap_unchecked_release(self) -> T {
        if cfg!(debug_assertions) {
            self.unwrap()
        } else {
            self.unwrap_unchecked()
        }
    }
}
