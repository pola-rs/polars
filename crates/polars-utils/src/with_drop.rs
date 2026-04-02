// A copy from the yet unstable library/core/src/mem/drop_guard.rs.

use core::fmt::{self, Debug};
use core::mem::ManuallyDrop;
use core::ops::{Deref, DerefMut};

pub struct WithDrop<T, F>
where
    F: FnOnce(T),
{
    inner: ManuallyDrop<T>,
    f: ManuallyDrop<F>,
}

impl<T, F> WithDrop<T, F>
where
    F: FnOnce(T),
{
    #[must_use]
    pub const fn new(inner: T, f: F) -> Self {
        Self {
            inner: ManuallyDrop::new(inner),
            f: ManuallyDrop::new(f),
        }
    }

    #[inline]
    pub fn into_inner(guard: Self) -> T {
        // First we ensure that dropping the guard will not trigger
        // its destructor
        let mut guard = ManuallyDrop::new(guard);

        // Next we manually read the stored value from the guard.
        //
        // SAFETY: this is safe because we've taken ownership of the guard.
        let value = unsafe { ManuallyDrop::take(&mut guard.inner) };

        // Finally we drop the stored closure. We do this *after* having read
        // the value, so that even if the closure's `drop` function panics,
        // unwinding still tries to drop the value.
        //
        // SAFETY: this is safe because we've taken ownership of the guard.
        unsafe { ManuallyDrop::drop(&mut guard.f) };
        value
    }
}

impl<T, F> Deref for WithDrop<T, F>
where
    F: FnOnce(T),
{
    type Target = T;

    fn deref(&self) -> &T {
        &self.inner
    }
}

impl<T, F> DerefMut for WithDrop<T, F>
where
    F: FnOnce(T),
{
    fn deref_mut(&mut self) -> &mut T {
        &mut self.inner
    }
}

impl<T, F> Drop for WithDrop<T, F>
where
    F: FnOnce(T),
{
    fn drop(&mut self) {
        // SAFETY: `WithDrop` is in the process of being dropped.
        let inner = unsafe { ManuallyDrop::take(&mut self.inner) };

        // SAFETY: `WithDrop` is in the process of being dropped.
        let f = unsafe { ManuallyDrop::take(&mut self.f) };

        f(inner);
    }
}

impl<T, F> Debug for WithDrop<T, F>
where
    T: Debug,
    F: FnOnce(T),
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}
