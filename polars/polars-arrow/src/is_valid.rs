use arrow::array::{Array, BooleanArray, GenericListArray, GenericStringArray, PrimitiveArray};
use arrow::datatypes::ArrowNumericType;
use arrow::{buffer::Buffer, util::bit_util};

pub trait IsValid {
    /// # Safety
    /// no bound checks
    unsafe fn is_valid_unchecked(&self, i: usize) -> bool;

    /// # Safety
    /// no bound checks
    unsafe fn is_null_unchecked(&self, i: usize) -> bool;
}

impl IsValid for Buffer {
    #[inline]
    unsafe fn is_valid_unchecked(&self, i: usize) -> bool {
        bit_util::get_bit_raw(self.as_ptr(), i)
    }

    #[inline]
    unsafe fn is_null_unchecked(&self, i: usize) -> bool {
        !self.is_valid_unchecked(i)
    }
}

pub trait ArrowArray: Array {}

impl ArrowArray for GenericStringArray<i64> {}
impl<T: ArrowNumericType> ArrowArray for PrimitiveArray<T> {}
impl ArrowArray for BooleanArray {}
impl ArrowArray for GenericListArray<i64> {}

impl<A: ArrowArray> IsValid for A {
    unsafe fn is_valid_unchecked(&self, i: usize) -> bool {
        if let Some(b) = self.data_ref().null_buffer() {
            b.is_valid_unchecked(self.offset() + i)
        } else {
            true
        }
    }

    unsafe fn is_null_unchecked(&self, i: usize) -> bool {
        !self.is_valid_unchecked(i)
    }
}
