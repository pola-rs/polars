use std::fmt::{Debug, Formatter};
use std::ops::Deref;

use crate::index::{IdxSize, NonZeroIdxSize};

pub type IdxVec = UnitVec<IdxSize>;

/// A type logically equivalent to `Vec<T>`, but which does not do a
/// memory allocation until at least two elements have been pushed, storing the
/// first element in the data pointer directly.
///
/// Uses IdxSize internally to store lengths, will panic if trying to reserve
/// for more elements.
#[derive(Eq)]
pub struct UnitVec<T> {
    len: IdxSize,
    capacity: NonZeroIdxSize,
    data: *mut T,
}

unsafe impl<T: Send + Sync> Send for UnitVec<T> {}
unsafe impl<T: Send + Sync> Sync for UnitVec<T> {}

impl<T> UnitVec<T> {
    #[inline(always)]
    fn data_ptr_mut(&mut self) -> *mut T {
        let external = self.data;
        let inline = &mut self.data as *mut *mut T as *mut T;
        if self.capacity.get() == 1 {
            inline
        } else {
            external
        }
    }

    #[inline(always)]
    fn data_ptr(&self) -> *const T {
        let external = self.data;
        let inline = &self.data as *const *mut T as *mut T;
        if self.capacity.get() == 1 {
            inline
        } else {
            external
        }
    }

    #[inline]
    pub fn new() -> Self {
        // This is optimized away, all const.
        assert!(size_of::<T>() <= size_of::<*mut T>() && align_of::<T>() <= align_of::<*mut T>());
        Self {
            len: 0,
            capacity: NonZeroIdxSize::new(1).unwrap(),
            data: std::ptr::null_mut(),
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.capacity.get() as usize
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    #[inline(always)]
    pub fn push(&mut self, idx: T) {
        if self.len == self.capacity.get() {
            self.reserve(1);
        }

        unsafe { self.push_unchecked(idx) }
    }

    #[inline(always)]
    /// # Safety
    /// Caller must ensure that `UnitVec` has enough capacity.
    pub unsafe fn push_unchecked(&mut self, idx: T) {
        unsafe {
            self.data_ptr_mut().add(self.len as usize).write(idx);
            self.len += 1;
        }
    }

    #[cold]
    #[inline(never)]
    pub fn reserve(&mut self, additional: usize) {
        let new_len = self
            .len
            .checked_add(additional.try_into().unwrap())
            .unwrap();
        if new_len > self.capacity.get() {
            let double = self.capacity.get() * 2;
            self.realloc(double.max(new_len).max(8));
        }
    }

    /// # Panics
    /// Panics if `new_cap <= 1` or `new_cap < self.len`
    fn realloc(&mut self, new_cap: IdxSize) {
        assert!(new_cap > 1 && new_cap >= self.len);
        unsafe {
            let mut me = std::mem::ManuallyDrop::new(Vec::with_capacity(new_cap as usize));
            let buffer = me.as_mut_ptr();
            std::ptr::copy(self.data_ptr(), buffer, self.len as usize);
            self.dealloc();
            self.data = buffer;
            self.capacity = NonZeroIdxSize::new(new_cap).unwrap();
        }
    }

    fn dealloc(&mut self) {
        unsafe {
            if self.capacity.get() > 1 {
                let _ = Vec::from_raw_parts(self.data, self.len as usize, self.capacity());
                self.capacity = NonZeroIdxSize::new(1).unwrap();
            }
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        if capacity <= 1 {
            Self::new()
        } else {
            let mut me = std::mem::ManuallyDrop::new(Vec::with_capacity(capacity));
            let data = me.as_mut_ptr();
            Self {
                len: 0,
                capacity: NonZeroIdxSize::new(capacity.try_into().unwrap()).unwrap(),
                data,
            }
        }
    }

    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.as_mut_slice().iter_mut()
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.as_ref()
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.as_mut()
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            unsafe {
                self.len -= 1;
                Some(std::ptr::read(self.as_ptr().add(self.len())))
            }
        }
    }
}

impl<T> Extend<T> for UnitVec<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        self.reserve(iter.size_hint().0);
        for v in iter {
            self.push(v)
        }
    }
}

impl<T> Drop for UnitVec<T> {
    fn drop(&mut self) {
        self.dealloc()
    }
}

impl<T> Clone for UnitVec<T> {
    fn clone(&self) -> Self {
        unsafe {
            if self.capacity.get() == 1 {
                Self { ..*self }
            } else {
                let mut copy = Self::with_capacity(self.len as usize);
                std::ptr::copy(self.data_ptr(), copy.data_ptr_mut(), self.len as usize);
                copy.len = self.len;
                copy
            }
        }
    }
}

impl<T: Debug> Debug for UnitVec<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "UnitVec: {:?}", self.as_slice())
    }
}

impl<T> Default for UnitVec<T> {
    fn default() -> Self {
        Self {
            len: 0,
            capacity: NonZeroIdxSize::new(1).unwrap(),
            data: std::ptr::null_mut(),
        }
    }
}

impl<T> Deref for UnitVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> AsRef<[T]> for UnitVec<T> {
    fn as_ref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data_ptr(), self.len as usize) }
    }
}

impl<T> AsMut<[T]> for UnitVec<T> {
    fn as_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data_ptr_mut(), self.len as usize) }
    }
}

impl<T: PartialEq> PartialEq for UnitVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T> FromIterator<T> for UnitVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        if iter.size_hint().0 <= 1 {
            let mut new = UnitVec::new();
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

impl<T> From<Vec<T>> for UnitVec<T> {
    fn from(mut value: Vec<T>) -> Self {
        if value.capacity() <= 1 {
            let mut new = UnitVec::new();
            if let Some(v) = value.pop() {
                new.push(v)
            }
            new
        } else {
            let mut me = std::mem::ManuallyDrop::new(value);
            UnitVec {
                data: me.as_mut_ptr(),
                capacity: NonZeroIdxSize::new(me.capacity().try_into().unwrap()).unwrap(),
                len: me.len().try_into().unwrap(),
            }
        }
    }
}

impl<T: Clone> From<&[T]> for UnitVec<T> {
    fn from(value: &[T]) -> Self {
        if value.len() <= 1 {
            let mut new = UnitVec::new();
            if let Some(v) = value.first() {
                new.push(v.clone())
            }
            new
        } else {
            value.to_vec().into()
        }
    }
}

#[macro_export]
macro_rules! unitvec {
    () => (
        $crate::idx_vec::UnitVec::new()
    );
    ($elem:expr; $n:expr) => (
        let mut new = $crate::idx_vec::UnitVec::new();
        for _ in 0..$n {
            new.push($elem)
        }
        new
    );
    ($elem:expr) => (
        {let mut new = $crate::idx_vec::UnitVec::new();
        let v = $elem;
        // SAFETY: first element always fits.
        unsafe { new.push_unchecked(v) };
        new}
    );
    ($($x:expr),+ $(,)?) => (
            vec![$($x),+].into()
    );
}

mod tests {

    #[test]
    #[should_panic]
    fn test_unitvec_realloc_zero() {
        super::UnitVec::<usize>::new().realloc(0);
    }

    #[test]
    #[should_panic]
    fn test_unitvec_realloc_one() {
        super::UnitVec::<usize>::new().realloc(1);
    }

    #[test]
    #[should_panic]
    fn test_untivec_realloc_lt_len() {
        super::UnitVec::<usize>::from(&[1, 2][..]).realloc(1)
    }

    #[test]
    fn test_unitvec_clone() {
        {
            let v = unitvec![1usize];
            assert_eq!(v, v.clone());
        }

        for n in [
            26903816120209729usize,
            42566276440897687,
            44435161834424652,
            49390731489933083,
            51201454727649242,
            83861672190814841,
            92169290527847622,
            92476373900398436,
            95488551309275459,
            97499984126814549,
        ] {
            let v = unitvec![n];
            assert_eq!(v, v.clone());
        }
    }
}
