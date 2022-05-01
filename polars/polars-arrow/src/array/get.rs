use crate::is_valid::IsValid;
use arrow::{
    array::{Array, BooleanArray, ListArray, PrimitiveArray, Utf8Array},
    types::NativeType,
};

pub trait ArrowGetItem {
    type Item;

    fn get(&self, item: usize) -> Option<Self::Item>;

    /// # Safety
    /// Get item. It is the callers resposibility that the `item < self.len()`
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
        if item >= self.len() {
            None
        } else {
            unsafe { self.get_unchecked(item) }
        }
    }

    #[inline]
    unsafe fn get_unchecked(&self, item: usize) -> Option<Self::Item> {
        if self.is_null_unchecked(item) {
            None
        } else {
            Some(self.value_unchecked(item))
        }
    }
}
