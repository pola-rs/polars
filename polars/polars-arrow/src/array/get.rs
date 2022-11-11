use arrow::array::{Array, BinaryArray, BooleanArray, ListArray, PrimitiveArray, Utf8Array};
use arrow::types::NativeType;

use crate::is_valid::IsValid;

pub trait ArrowGetItem {
    type Item;

    fn get(&self, item: usize) -> Option<Self::Item>;

    /// # Safety
    /// Get item. It is the callers responsibility that the `item < self.len()`
    unsafe fn get_unchecked(&self, item: usize) -> Option<Self::Item>;
}

impl<T: NativeType> ArrowGetItem for PrimitiveArray<T> {
    type Item = T;

    #[inline]
    fn get(&self, item: usize) -> Option<Self::Item> {
        if item >= self.len() {
            None
        } else {
            unsafe { self.get_unchecked(item) }
        }
    }

    #[inline]
    unsafe fn get_unchecked(&self, item: usize) -> Option<Self::Item> {
        debug_assert!(item < self.len());
        if self.is_null_unchecked(item) {
            None
        } else {
            Some(self.value_unchecked(item))
        }
    }
}

impl ArrowGetItem for BooleanArray {
    type Item = bool;

    #[inline]
    fn get(&self, item: usize) -> Option<Self::Item> {
        if item >= self.len() {
            None
        } else {
            unsafe { self.get_unchecked(item) }
        }
    }

    #[inline]
    unsafe fn get_unchecked(&self, item: usize) -> Option<Self::Item> {
        debug_assert!(item < self.len());
        if self.is_null_unchecked(item) {
            None
        } else {
            Some(self.value_unchecked(item))
        }
    }
}

impl<'a> ArrowGetItem for &'a Utf8Array<i64> {
    type Item = &'a str;

    #[inline]
    fn get(&self, item: usize) -> Option<Self::Item> {
        if item >= self.len() {
            None
        } else {
            unsafe { self.get_unchecked(item) }
        }
    }

    #[inline]
    unsafe fn get_unchecked(&self, item: usize) -> Option<Self::Item> {
        debug_assert!(item < self.len());
        if self.is_null_unchecked(item) {
            None
        } else {
            Some(self.value_unchecked(item))
        }
    }
}

impl<'a> ArrowGetItem for &'a BinaryArray<i64> {
    type Item = &'a [u8];

    #[inline]
    fn get(&self, item: usize) -> Option<Self::Item> {
        if item >= self.len() {
            None
        } else {
            unsafe { self.get_unchecked(item) }
        }
    }

    #[inline]
    unsafe fn get_unchecked(&self, item: usize) -> Option<Self::Item> {
        debug_assert!(item < self.len());
        if self.is_null_unchecked(item) {
            None
        } else {
            Some(self.value_unchecked(item))
        }
    }
}

impl ArrowGetItem for ListArray<i64> {
    type Item = Box<dyn Array>;

    #[inline]
    fn get(&self, item: usize) -> Option<Self::Item> {
        debug_assert!(item < self.len());
        if item >= self.len() {
            None
        } else {
            unsafe { self.get_unchecked(item) }
        }
    }

    #[inline]
    unsafe fn get_unchecked(&self, item: usize) -> Option<Self::Item> {
        debug_assert!(item < self.len());
        if self.is_null_unchecked(item) {
            None
        } else {
            Some(self.value_unchecked(item))
        }
    }
}
