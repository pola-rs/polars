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
