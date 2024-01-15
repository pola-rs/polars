use polars_error::PolarsResult;

use super::FixedSizeBinaryArray;
use crate::array::{FromFfi, ToFfi};
use crate::bitmap::align;
use crate::ffi;

unsafe impl ToFfi for FixedSizeBinaryArray {
    fn buffers(&self) -> Vec<Option<*const u8>> {
        vec![
            self.validity.as_ref().map(|x| x.as_ptr()),
            Some(self.values.storage_ptr().cast::<u8>()),
        ]
    }

    fn offset(&self) -> Option<usize> {
        let offset = self.values.offset() / self.size;
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
        let offset = self.values.offset() / self.size;

        let validity = self.validity.as_ref().map(|bitmap| {
            if bitmap.offset() == offset {
                bitmap.clone()
            } else {
                align(bitmap, offset)
            }
        });

        Self {
            size: self.size,
            data_type: self.data_type.clone(),
            validity,
            values: self.values.clone(),
        }
    }
}

impl<A: ffi::ArrowArrayRef> FromFfi<A> for FixedSizeBinaryArray {
    unsafe fn try_from_ffi(array: A) -> PolarsResult<Self> {
        let data_type = array.data_type().clone();
        let validity = unsafe { array.validity() }?;
        let values = unsafe { array.buffer::<u8>(1) }?;

        Self::try_new(data_type, values, validity)
    }
}
