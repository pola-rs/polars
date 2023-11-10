use std::alloc::{GlobalAlloc, Layout, System};
use std::fmt::{Debug, Formatter};
use std::num::NonZeroUsize;
use std::ops::Deref;

use crate::IdxSize;

/// A type logically equivalent to `Vec<IdxSize>`, but which does not do a
/// memory allocation until at least two elements have been pushed, storing the
/// first element in the data pointer directly.
#[derive(Eq)]
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

        unsafe { self.push_unchecked(idx) }
    }

    #[inline(always)]
    /// # Safety
    /// Caller must ensure that `IdxVec` has enough capacity.
    pub unsafe fn push_unchecked(&mut self, idx: IdxSize) {
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
            let mut me = std::mem::ManuallyDrop::new(Vec::with_capacity(new_cap));
            let buffer = me.as_mut_ptr();
            std::ptr::copy(self.data_ptr(), buffer, self.len);
            self.dealloc();
            self.data = buffer;
            self.capacity = NonZeroUsize::new(new_cap).unwrap();
        }
    }

    fn dealloc(&mut self) {
        unsafe {
            if self.capacity.get() > 1 {
                let _ = Vec::from_raw_parts(self.data, self.len, self.capacity());
                self.capacity = NonZeroUsize::new(1).unwrap();
            }
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let mut new = Self::new();
        new.reserve(capacity);
        new
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

impl Clone for IdxVec {
    fn clone(&self) -> Self {
        unsafe {
            let layout = Layout::array::<IdxSize>(self.len).unwrap();
            let buffer = System.alloc(layout) as *mut IdxSize;
            std::ptr::copy(self.data_ptr(), buffer, self.len);
            IdxVec {
                data: buffer,
                len: self.len,
                capacity: NonZeroUsize::new(std::cmp::max(self.len, 1)).unwrap(),
            }
        }
    }
}

impl Debug for IdxVec {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "IdxVec: {:?}", self.as_slice())
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

impl Deref for IdxVec {
    type Target = [IdxSize];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
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

impl PartialEq for IdxVec {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl FromIterator<IdxSize> for IdxVec {
    fn from_iter<T: IntoIterator<Item = IdxSize>>(iter: T) -> Self {
        let iter = iter.into_iter();
        if iter.size_hint().0 <= 1 {
            let mut new = IdxVec::new();
            for v in iter {
                new.push(v)
            }
            new
        } else {
            let v = iter.collect::<Vec<_>>();
            v.into()
        }
    }
}

impl From<Vec<IdxSize>> for IdxVec {
    fn from(value: Vec<IdxSize>) -> Self {
        if value.capacity() <= 1 {
            let mut new = IdxVec::new();
            if let Some(v) = value.first() {
                new.push(*v)
            }
            new
        } else {
            let mut me = std::mem::ManuallyDrop::new(value);
            IdxVec {
                data: me.as_mut_ptr(),
                capacity: NonZeroUsize::new(me.capacity()).unwrap(),
                len: me.len(),
            }
        }
    }
}

impl From<&[IdxSize]> for IdxVec {
    fn from(value: &[IdxSize]) -> Self {
        if value.len() <= 1 {
            let mut new = IdxVec::new();
            if let Some(v) = value.first() {
                new.push(*v)
            }
            new
        } else {
            value.to_vec().into()
        }
    }
}

#[macro_export]
macro_rules! idxvec {
    () => (
        $crate::idx_vec::IdxVec::new()
    );
    ($elem:expr; $n:expr) => (
        let mut new = $crate::idx_vec::IdxVec::new();
        for _ in 0..$n {
            new.push($elem)
        }
        new
    );
    ($elem:expr) => (
        {let mut new = $crate::idx_vec::IdxVec::new();
        // SAFETY: first element always fits.
        unsafe { new.push_unchecked($elem) };
        new}
    );
    ($($x:expr),+ $(,)?) => (
            vec![$($x),+].into()
    );
}
