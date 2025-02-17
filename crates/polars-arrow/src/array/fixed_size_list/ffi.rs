use polars_error::{polars_ensure, PolarsResult};

use super::FixedSizeListArray;
use crate::array::ffi::{FromFfi, ToFfi};
use crate::array::Array;
use crate::ffi;

unsafe impl ToFfi for FixedSizeListArray {
    fn buffers(&self) -> Vec<Option<*const u8>> {
        vec![self.validity.as_ref().map(|x| x.as_ptr())]
    }

    fn children(&self) -> Vec<Box<dyn Array>> {
        vec![self.values.clone()]
    }

    fn offset(&self) -> Option<usize> {
        Some(
            self.validity
                .as_ref()
                .map(|bitmap| bitmap.offset())
                .unwrap_or_default(),
        )
    }

    fn to_ffi_aligned(&self) -> Self {
        self.clone()
    }
}

impl<A: ffi::ArrowArrayRef> FromFfi<A> for FixedSizeListArray {
    unsafe fn try_from_ffi(array: A) -> PolarsResult<Self> {
        let dtype = array.dtype().clone();
        let (_, width) = FixedSizeListArray::try_child_and_size(&dtype)?;
        let validity = unsafe { array.validity() }?;
        let child = unsafe { array.child(0) }?;
        let values = ffi::try_from(child)?;

        let length = if values.len() == 0 {
            0
        } else {
            polars_ensure!(width > 0, InvalidOperation: "Zero-width array with values");
            values.len() / width
        };

        let mut fsl = Self::try_new(dtype, length, values, validity)?;
        fsl.slice(array.offset(), array.length());
        Ok(fsl)
    }
}
