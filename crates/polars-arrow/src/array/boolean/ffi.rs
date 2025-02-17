use polars_error::PolarsResult;

use super::BooleanArray;
use crate::array::{FromFfi, ToFfi};
use crate::bitmap::align;
use crate::ffi;

unsafe impl ToFfi for BooleanArray {
    fn buffers(&self) -> Vec<Option<*const u8>> {
        vec![
            self.validity.as_ref().map(|x| x.as_ptr()),
            Some(self.values.as_ptr()),
        ]
    }

    fn offset(&self) -> Option<usize> {
        let offset = self.values.offset();
        if let Some(bitmap) = self.validity.as_ref() {
            if bitmap.offset() == offset {
                Some(offset)
            } else {
                None
            }
        } else {
            Some(offset)
        }
    }

    fn to_ffi_aligned(&self) -> Self {
        let offset = self.values.offset();

        let validity = self.validity.as_ref().map(|bitmap| {
            if bitmap.offset() == offset {
                bitmap.clone()
            } else {
                align(bitmap, offset)
            }
        });

        Self {
            dtype: self.dtype.clone(),
            validity,
            values: self.values.clone(),
        }
    }
}

impl<A: ffi::ArrowArrayRef> FromFfi<A> for BooleanArray {
    unsafe fn try_from_ffi(array: A) -> PolarsResult<Self> {
        let dtype = array.dtype().clone();
        let validity = unsafe { array.validity() }?;
        let values = unsafe { array.bitmap(1) }?;
        Self::try_new(dtype, values, validity)
    }
}
