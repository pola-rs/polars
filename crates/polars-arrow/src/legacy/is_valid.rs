use crate::array::{
    Array, BinaryArray, BooleanArray, FixedSizeListArray, ListArray, PrimitiveArray, Utf8Array,
};
use crate::types::NativeType;

pub trait IsValid {
    /// # Safety
    /// no bound checks
    unsafe fn is_valid_unchecked(&self, i: usize) -> bool;
}

pub trait ArrowArray: Array {}

impl ArrowArray for BinaryArray<i64> {}
impl ArrowArray for Utf8Array<i64> {}
impl<T: NativeType> ArrowArray for PrimitiveArray<T> {}
impl ArrowArray for BooleanArray {}
impl ArrowArray for ListArray<i64> {}
impl ArrowArray for FixedSizeListArray {}

impl<A: ArrowArray> IsValid for A {
    #[inline]
    unsafe fn is_valid_unchecked(&self, i: usize) -> bool {
        !self.is_null_unchecked(i)
    }
}
