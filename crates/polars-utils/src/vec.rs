use bytemuck::Pod;

use crate::with_drop::WithDrop;

/// Re-uses the memory for a vec while clearing it. Allows casting the type of
/// the vec at the same time. The stdlib specializes collect() to re-use the
/// memory.
pub fn reuse_vec<T, U>(v: Vec<T>) -> Vec<U> {
    const {
        assert!(core::mem::size_of::<T>() == core::mem::size_of::<U>());
        assert!(core::mem::align_of::<T>() == core::mem::align_of::<U>());
    }
    v.into_iter().filter_map(|_| None).collect()
}

pub trait PushUnchecked<T> {
    /// Will push an item and not check if there is enough capacity
    ///
    /// # Safety
    /// Caller must ensure the array has enough capacity to hold `T`.
    unsafe fn push_unchecked(&mut self, value: T);
}

impl<T> PushUnchecked<T> for Vec<T> {
    #[inline]
    unsafe fn push_unchecked(&mut self, value: T) {
        unsafe {
            let len = self.len();
            debug_assert!(self.capacity() > len);
            let end = self.as_mut_ptr().add(len);
            std::ptr::write(end, value);
            self.set_len(len + 1);
        }
    }
}

pub fn with_cast_mut_vec<T: Pod, U: Pod, R, F: FnOnce(&mut Vec<U>) -> R>(
    v: &mut Vec<T>,
    f: F,
) -> R {
    let mut vu = WithDrop::new(bytemuck::cast_vec::<T, U>(core::mem::take(v)), |vu| {
        *v = bytemuck::cast_vec::<U, T>(vu)
    });
    f(&mut vu)
}
