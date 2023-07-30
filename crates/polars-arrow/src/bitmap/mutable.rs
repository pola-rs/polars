use arrow::bitmap::MutableBitmap;

pub trait MutableBitmapExtension {
    fn as_slice_mut(&mut self) -> &mut [u8];

    /// # Safety
    /// Caller must ensure `i` is in bounds.
    unsafe fn set_bit_unchecked(&mut self, i: usize, value: bool);
}

impl MutableBitmapExtension for MutableBitmap {
    fn as_slice_mut(&mut self) -> &mut [u8] {
        let slice = self.as_slice();
        unsafe { std::slice::from_raw_parts_mut(slice.as_ptr() as *mut u8, slice.len()) }
    }

    unsafe fn set_bit_unchecked(&mut self, i: usize, value: bool) {
        #[cfg(debug_assertions)]
        {
            arrow::bitmap::utils::set_bit(self.as_slice_mut(), i, value)
        }
        #[cfg(not(debug_assertions))]
        {
            arrow::bitmap::utils::set_bit_unchecked(self.as_slice_mut(), i, value)
        }
    }
}
