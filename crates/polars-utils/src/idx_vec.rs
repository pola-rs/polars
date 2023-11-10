use std::alloc::{GlobalAlloc, Layout, System};
use std::num::NonZeroUsize;

use crate::IdxSize;

/// A type logically equivalent to `Vec<IdxSize>`, but which does not do a
/// memory allocation until at least two elements have been pushed, storing the
/// first element in the data pointer directly.
pub struct IdxVec {
    len: usize,
    capacity: NonZeroUsize,
    data: *mut IdxSize,
}

unsafe impl Send for IdxVec {}
unsafe impl Sync for IdxVec {}

impl IdxVec {
    #[inline(always)]
    fn data_ptr_mut(&mut self) -> *mut IdxSize {
        let external = self.data;
        let inline = &mut self.data as *mut *mut IdxSize as *mut IdxSize;
        if self.capacity.get() == 1 {
            inline
        } else {
            external
        }
    }

    #[inline(always)]
    fn data_ptr(&self) -> *const IdxSize {
        let external = self.data;
        let inline = &self.data as *const *mut IdxSize as *mut IdxSize;
        if self.capacity.get() == 1 {
            inline
        } else {
            external
        }
    }

    pub fn new() -> Self {
        Self {
            len: 0,
            capacity: NonZeroUsize::new(1).unwrap(),
            data: std::ptr::null_mut(),
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.capacity.get()
    }

    #[inline(always)]
    pub fn push(&mut self, idx: IdxSize) {
        if self.len == self.capacity.get() {
            self.reserve(1);
        }

        unsafe {
            self.data_ptr_mut().add(self.len).write(idx);
            self.len += 1;
        }
    }

    #[cold]
    #[inline(never)]
    pub fn reserve(&mut self, additional: usize) {
        if self.len + additional > self.capacity.get() {
            let double = self.capacity.get() * 2;
            self.realloc(double.max(self.len + additional).max(8));
        }
    }

    fn realloc(&mut self, new_cap: usize) {
        assert!(new_cap >= self.len);
        unsafe {
            let layout = Layout::array::<IdxSize>(new_cap).unwrap();
            let buffer = System.alloc(layout) as *mut IdxSize;
            std::ptr::copy(self.data_ptr(), buffer, self.len);
            self.dealloc();
            self.data = buffer;
            self.capacity = NonZeroUsize::new(new_cap).unwrap();
        }
    }

    fn dealloc(&mut self) {
        unsafe {
            if self.capacity.get() > 1 {
                let layout = Layout::array::<IdxSize>(self.capacity.get()).unwrap();
                System.dealloc(self.data as *mut u8, layout);
                self.capacity = NonZeroUsize::new(1).unwrap();
            }
        }
    }

    pub fn iter(&self) -> std::slice::Iter<'_, IdxSize> {
        self.as_slice().iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, IdxSize> {
        self.as_mut_slice().iter_mut()
    }

    pub fn as_slice(&self) -> &[IdxSize] {
        self.as_ref()
    }

    pub fn as_mut_slice(&mut self) -> &mut [IdxSize] {
        self.as_mut()
    }
}

impl Drop for IdxVec {
    fn drop(&mut self) {
        self.dealloc()
    }
}

impl Default for IdxVec {
    fn default() -> Self {
        Self {
            len: 0,
            capacity: NonZeroUsize::new(1).unwrap(),
            data: std::ptr::null_mut(),
        }
    }
}

impl AsRef<[IdxSize]> for IdxVec {
    fn as_ref(&self) -> &[IdxSize] {
        unsafe { std::slice::from_raw_parts(self.data_ptr(), self.len) }
    }
}

impl AsMut<[IdxSize]> for IdxVec {
    fn as_mut(&mut self) -> &mut [IdxSize] {
        unsafe { std::slice::from_raw_parts_mut(self.data_ptr_mut(), self.len) }
    }
}
