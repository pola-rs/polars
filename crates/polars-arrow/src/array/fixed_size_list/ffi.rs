use polars_error::PolarsResult;

use super::FixedSizeListArray;
use crate::array::Array;
use crate::array::ffi::{FromFfi, ToFfi};
use crate::ffi;

unsafe impl ToFfi for FixedSizeListArray {
    fn buffers(&self) -> Vec<Option<*const u8>> {
        vec![self.validity.as_ref().map(|x| x.as_aligned_ptr().unwrap())]
    }

    fn children(&self) -> Vec<Box<dyn Array>> {
        vec![self.values.clone()]
    }

    fn offset(&self) -> Option<usize> {
        Some(0)
    }

    fn to_ffi_aligned(&self) -> Self {
        let mut ret = self.clone();

        if let Some(validity) = ret.validity()
            && validity.as_aligned_ptr().is_none()
        {
            ret.validity = Some(validity.to_aligned_bitmap());
        }

        ret
    }
}

impl<A: ffi::ArrowArrayRef> FromFfi<A> for FixedSizeListArray {
    unsafe fn try_from_ffi(array: A) -> PolarsResult<Self> {
        let dtype = array.dtype().clone();
        let (_, width) = FixedSizeListArray::try_child_and_size(&dtype)?;
        let validity = unsafe { array.validity() }?;
        let child = unsafe { array.child(0) }?;
        let mut values = ffi::try_from(child)?;
        values.slice(array.offset() * width, array.length() * width);

        Self::try_new(dtype, array.length(), values, validity)
    }
}
