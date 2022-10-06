use arrow::array::{Array, BinaryArray, BooleanArray, ListArray, PrimitiveArray, Utf8Array};
use arrow::types::NativeType;

pub trait IsValid {
    /// # Safety
    /// no bound checks
    unsafe fn is_valid_unchecked(&self, i: usize) -> bool;

    /// # Safety
    /// no bound checks
    unsafe fn is_null_unchecked(&self, i: usize) -> bool;
}

pub trait ArrowArray: Array {}

impl ArrowArray for BinaryArray<i64> {}
impl ArrowArray for Utf8Array<i64> {}
impl<T: NativeType> ArrowArray for PrimitiveArray<T> {}
impl ArrowArray for BooleanArray {}
impl ArrowArray for ListArray<i64> {}

impl<A: ArrowArray> IsValid for A {
    unsafe fn is_valid_unchecked(&self, i: usize) -> bool {
        if let Some(b) = self.validity() {
            b.get_bit_unchecked(i)
        } else {
            true
        }
    }

    unsafe fn is_null_unchecked(&self, i: usize) -> bool {
        !self.is_valid_unchecked(i)
    }
}
