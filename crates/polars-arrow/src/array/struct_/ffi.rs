use polars_error::PolarsResult;

use super::super::ffi::ToFfi;
use super::super::{Array, FromFfi};
use super::StructArray;
use crate::bitmap::align;
use crate::ffi;

unsafe impl ToFfi for StructArray {
    fn buffers(&self) -> Vec<Option<*const u8>> {
        vec![self.validity.as_ref().map(|x| x.as_ptr())]
    }

    fn children(&self) -> Vec<Box<dyn Array>> {
        self.values.clone()
    }

    fn offset(&self) -> Option<usize> {
        // The child arrays are sliced together with the struct (see `slice_unchecked`),
        // so they already start at logical index 0. The exported offset is applied to the
        // children by the consumer, hence it must be 0 to avoid shifting them a second time.
        //
        // The only buffer the struct owns is the validity bitmap. A non-zero bitmap offset
        // cannot be expressed through a raw pointer, so we signal that the array first needs
        // to be re-aligned via `to_ffi_aligned`.
        match self.validity.as_ref() {
            Some(bitmap) if bitmap.offset() != 0 => None,
            _ => Some(0),
        }
    }

    fn to_ffi_aligned(&self) -> Self {
        // Re-align the validity bitmap to offset 0 so it matches the already-sliced children.
        let validity = self.validity.as_ref().map(|bitmap| align(bitmap, 0));
        Self::new(
            self.dtype.clone(),
            self.length,
            self.values.clone(),
            validity,
        )
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
                    // there is a discrepancy with how polars_arrow exports sliced
                    // struct array and how pyarrow does it.
                    // # Pyarrow
                    // ## struct array len 3
                    //  * slice 1 by with len 2
                    //      offset on struct array: 1
                    //      length on struct array: 2
                    //      offset on value array: 0
                    //      length on value array: 3
                    // # Arrow2
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
