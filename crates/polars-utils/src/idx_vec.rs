use std::fmt::{Debug, Formatter};
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};

use crate::index::{IdxSize, NonZeroIdxSize};

pub type IdxVec = UnitVec<IdxSize>;

union PointerOrValue<T> {
    ptr: *mut T,
    value: ManuallyDrop<T>,
}

/// A type logically equivalent to `Vec<T>`, but which does not do a
/// memory allocation until at least two elements have been pushed, storing the
/// first element inside the UnitVec directly.
///
/// Uses IdxSize internally to store lengths, will panic if trying to reserve
/// for more elements.
pub struct UnitVec<T> {
    len: IdxSize,
    capacity: NonZeroIdxSize,
    data: PointerOrValue<T>,
}

unsafe impl<T: Send + Sync> Send for UnitVec<T> {}
unsafe impl<T: Send + Sync> Sync for UnitVec<T> {}

impl<T> UnitVec<T> {
    #[inline(always)]
    fn data_ptr_mut(&mut self) -> *mut T {
        if self.is_inline() {
            unsafe { &mut *self.data.value }
        } else {
            unsafe { self.data.ptr }
        }
    }

    #[inline(always)]
    fn data_ptr(&self) -> *const T {
        if self.is_inline() {
            unsafe { &*self.data.value }
        } else {
            unsafe { self.data.ptr }
        }
    }

    #[inline]
    pub fn new() -> Self {
        Self {
            len: 0,
            capacity: NonZeroIdxSize::new(1).unwrap(),
            data: PointerOrValue {
                ptr: std::ptr::null_mut(),
            },
        }
    }

    pub fn is_inline(&self) -> bool {
        self.capacity.get() == 1
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
        if std::mem::needs_drop::<T>() {
            while self.len > 0 {
                self.pop();
            }
        } else {
            self.len = 0;
        }
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

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            unsafe {
                self.len -= 1;
                Some(self.data_ptr().add(self.len as usize).read())
            }
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
    fn realloc(&mut self, mut new_cap: IdxSize) {
        assert!(new_cap > 1 && new_cap >= self.len);
        unsafe {
            let mut me = std::mem::ManuallyDrop::new(Vec::with_capacity(new_cap as usize));
            new_cap = me.capacity().try_into().unwrap();
            let buffer = me.as_mut_ptr();
            std::ptr::copy(self.data_ptr(), buffer, self.len as usize);
            self.dealloc();
            self.data = PointerOrValue { ptr: buffer };
            self.capacity = NonZeroIdxSize::new(new_cap).unwrap();
        }
    }

    unsafe fn dealloc(&mut self) {
        unsafe {
            if !self.is_inline() {
                drop(Vec::from_raw_parts(
                    self.data.ptr.cast::<ManuallyDrop<T>>(),
                    self.len as usize,
                    self.capacity(),
                ));
            }
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        if capacity <= 1 {
            Self::new()
        } else {
            let mut me = std::mem::ManuallyDrop::new(Vec::with_capacity(capacity));
            let cap = me.capacity().try_into().unwrap();
            let ptr = me.as_mut_ptr();
            Self {
                len: 0,
                capacity: NonZeroIdxSize::new(cap).unwrap(),
                data: PointerOrValue { ptr },
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
}

impl<T: Copy> UnitVec<T> {
    pub fn retain(&mut self, mut f: impl FnMut(T) -> bool) {
        let mut i = 0;
        for j in 0..self.len() {
            if f(self[j]) {
                self[i] = self[j];
                i += 1;
            }
        }

        if i == 0 {
            *self = Self::new();
        } else if i == 1 {
            *self = Self::from_slice(&[self[0]]);
        } else {
            self.len = i as IdxSize;
        }
    }
}

impl<T: Clone> UnitVec<T> {
    pub fn from_slice(sl: &[T]) -> Self {
        if sl.len() <= 1 {
            let mut new = UnitVec::new();
            if let Some(v) = sl.first() {
                new.push(v.clone())
            }
            new
        } else {
            sl.to_vec().into()
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
        self.clear();
        unsafe { self.dealloc() }
    }
}

impl<T: Clone> Clone for UnitVec<T> {
    fn clone(&self) -> Self {
        Self::from_iter(self.iter().cloned())
    }
}

impl<T: Debug> Debug for UnitVec<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "UnitVec: {:?}", self.as_slice())
    }
}

impl<T> Default for UnitVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Deref for UnitVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for UnitVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
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

impl<T: Eq> Eq for UnitVec<T> {}

impl<T> FromIterator<T> for UnitVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut iter = iter.into_iter();

        let Some(first) = iter.next() else {
            return Self::new();
        };

        let Some(second) = iter.next() else {
            let mut out = Self::new();
            out.push(first);
            return out;
        };

        let mut vec = Vec::with_capacity(iter.size_hint().0 + 2);
        vec.push(first);
        vec.push(second);
        vec.extend(iter);
        Self::from(vec)
    }
}

impl<T> IntoIterator for UnitVec<T> {
    type Item = T;

    type IntoIter = IntoIter<T>;

    fn into_iter(mut self) -> Self::IntoIter {
        if self.is_inline() {
            IntoIter::Inline(self.pop().into_iter())
        } else {
            IntoIter::External(Vec::from(self).into_iter())
        }
    }
}

pub enum IntoIter<T> {
    Inline(std::option::IntoIter<T>),
    External(std::vec::IntoIter<T>),
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            IntoIter::Inline(it) => it.next(),
            IntoIter::External(it) => it.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            IntoIter::Inline(it) => it.size_hint(),
            IntoIter::External(it) => it.size_hint(),
        }
    }
}

impl<T, const N: usize> From<[T; N]> for UnitVec<T> {
    fn from(value: [T; N]) -> Self {
        UnitVec::from_iter(value)
    }
}

impl<T> ExactSizeIterator for IntoIter<T> {}

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
                data: PointerOrValue {
                    ptr: me.as_mut_ptr(),
                },
                capacity: NonZeroIdxSize::new(me.capacity().try_into().unwrap()).unwrap(),
                len: me.len().try_into().unwrap(),
            }
        }
    }
}

impl<T> From<UnitVec<T>> for Vec<T> {
    fn from(mut value: UnitVec<T>) -> Self {
        if value.is_inline() {
            let mut out = Vec::with_capacity(value.len());
            if let Some(item) = value.pop() {
                out.push(item);
            }
            out
        } else {
            // SAFETY: when not inline, the data points to a buffer allocated by a Vec.
            let out = unsafe {
                Vec::from_raw_parts(value.data.ptr, value.len as usize, value.capacity())
            };
            // Prevent deallocating the buffer
            std::mem::forget(value);
            out
        }
    }
}

#[macro_export]
macro_rules! unitvec {
    () => {{
        $crate::idx_vec::UnitVec::new()
    }};
    ($elem:expr; $n:expr) => {{
        let mut new = $crate::idx_vec::UnitVec::new();
        for _ in 0..$n {
            new.push($elem)
        }
        new
    }};
    ($elem:expr) => {{
        let mut new = $crate::idx_vec::UnitVec::new();
        let v = $elem;
        // SAFETY: first element always fits.
        unsafe { new.push_unchecked(v) };
        new
    }};
    ($($x:expr),+ $(,)?) => {{
        vec![$($x),+].into()
    }};
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
        super::UnitVec::<usize>::from([1, 2]).realloc(1)
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

    #[test]
    fn test_unitvec_repeat_n() {
        assert_eq!(unitvec![5; 3].as_slice(), &[5, 5, 5])
    }
}
