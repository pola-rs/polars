use crate::{
    array::{FromFfi, ToFfi},
    bitmap::align,
    ffi,
    offset::{Offset, OffsetsBuffer},
};

use crate::error::Result;

use super::BinaryArray;

unsafe impl<O: Offset> ToFfi for BinaryArray<O> {
    fn buffers(&self) -> Vec<Option<*const u8>> {
        vec![
            self.validity.as_ref().map(|x| x.as_ptr()),
            Some(self.offsets.buffer().as_ptr().cast::<u8>()),
            Some(self.values.as_ptr().cast::<u8>()),
        ]
    }

    fn offset(&self) -> Option<usize> {
        let offset = self.offsets.buffer().offset();
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
        let offset = self.offsets.buffer().offset();

        let validity = self.validity.as_ref().map(|bitmap| {
            if bitmap.offset() == offset {
                bitmap.clone()
            } else {
                align(bitmap, offset)
            }
        });

        Self {
            data_type: self.data_type.clone(),
            validity,
            offsets: self.offsets.clone(),
            values: self.values.clone(),
        }
    }
}

impl<O: Offset, A: ffi::ArrowArrayRef> FromFfi<A> for BinaryArray<O> {
    unsafe fn try_from_ffi(array: A) -> Result<Self> {
        let data_type = array.data_type().clone();

        let validity = unsafe { array.validity() }?;
        let offsets = unsafe { array.buffer::<O>(1) }?;
        let values = unsafe { array.buffer::<u8>(2) }?;

        // assumption that data from FFI is well constructed
        let offsets = unsafe { OffsetsBuffer::new_unchecked(offsets) };

        Ok(Self::new(data_type, offsets, values, validity))
    }
}
