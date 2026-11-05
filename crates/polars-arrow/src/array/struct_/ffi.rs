use polars_error::PolarsResult;

use super::super::ffi::ToFfi;
use super::super::{Array, FromFfi};
use super::StructArray;
use crate::ffi;

unsafe impl ToFfi for StructArray {
    fn buffers(&self) -> Vec<Option<*const u8>> {
        vec![self.validity.as_ref().map(|x| x.as_aligned_ptr().unwrap())]
    }

    fn children(&self) -> Vec<Box<dyn Array>> {
        self.values.clone()
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

impl<A: ffi::ArrowArrayRef> FromFfi<A> for StructArray {
    unsafe fn try_from_ffi(array: A) -> PolarsResult<Self> {
        let dtype = array.dtype().clone();
        let fields = Self::get_fields(&dtype);

        let arrow_array = array.array();
        let validity = unsafe { array.validity() }?;
        let len = arrow_array.len();
        let offset = arrow_array.offset();
        let values = (0..fields.len())
            .map(|index| {
                let child = array.child(index)?;
                ffi::try_from(child).map(|arr| {
                    // Old versions of polars_arrow exported sliced
                    // struct arrays differently.
                    // # Pyarrow
                    // ## struct array len 3
                    //  * slice 1 by with len 2
                    //      offset on struct array: 1
                    //      length on struct array: 2
                    //      offset on value array: 0
                    //      length on value array: 3
                    // # polars_arrow (old)
                    // ## struct array len 3
                    //  * slice 1 by with len 2
                    //      offset on struct array: 0
                    //      length on struct array: 3
                    //      offset on value array: 1
                    //      length on value array: 2
                    //
                    // this branch will ensure both can round trip
                    if arr.len() >= (len + offset) {
                        arr.sliced(offset, len)
                    } else {
                        arr
                    }
                })
            })
            .collect::<PolarsResult<Vec<Box<dyn Array>>>>()?;

        Self::try_new(dtype, len, values, validity)
    }
}
