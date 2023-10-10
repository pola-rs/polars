use std::mem::ManuallyDrop;

use arrow::ffi;
use arrow::ffi::{ArrowArray, ArrowSchema};
use polars_core::error::PolarsResult;
use polars_core::prelude::{ArrayRef, ArrowField, Series};

// A utility that helps releasing/owning memory.
#[allow(dead_code)]
struct PrivateData {
    schema: Box<ArrowSchema>,
    arrays: Box<[*mut ArrowArray]>,
}

/// An FFI exported `Series`.
#[repr(C)]
pub struct SeriesExport {
    field: *mut ArrowSchema,
    // A double ptr, so we can easily release the buffer
    // without dropping the arrays.
    arrays: *mut *mut ArrowArray,
    len: usize,
    release: Option<unsafe extern "C" fn(arg1: *mut SeriesExport)>,
    private_data: *mut std::os::raw::c_void,
}

impl Drop for SeriesExport {
    fn drop(&mut self) {
        if let Some(release) = self.release {
            unsafe { release(self) }
        }
    }
}

// callback used to drop [SeriesExport] when it is exported.
unsafe extern "C" fn c_release_series_export(e: *mut SeriesExport) {
    if e.is_null() {
        return;
    }
    let e = &mut *e;
    let private = Box::from_raw(e.private_data as *mut PrivateData);
    for ptr in private.arrays.iter() {
        // drop the box, not the array
        let _ = Box::from_raw(*ptr as *mut ManuallyDrop<ArrowArray>);
    }

    e.release = None;
}

pub fn export_series(s: &Series) -> SeriesExport {
    let field = ArrowField::new(s.name(), s.dtype().to_arrow(), true);
    let schema = Box::new(ffi::export_field_to_c(&field));
    let mut arrays = s
        .chunks()
        .iter()
        .map(|arr| Box::into_raw(Box::new(ffi::export_array_to_c(arr.clone()))))
        .collect::<Box<_>>();
    let len = arrays.len();
    let ptr = arrays.as_mut_ptr();
    SeriesExport {
        field: schema.as_ref() as *const ArrowSchema as *mut ArrowSchema,
        arrays: ptr,
        len,
        release: Some(c_release_series_export),
        private_data: Box::into_raw(Box::new(PrivateData { arrays, schema }))
            as *mut std::os::raw::c_void,
    }
}

/// # Safety
/// `SeriesExport` must be valid
pub unsafe fn import_series(e: SeriesExport) -> PolarsResult<Series> {
    let field = ffi::import_field_from_c(&(*e.field))?;

    let pointers = std::slice::from_raw_parts_mut(e.arrays, e.len);
    let chunks = pointers
        .iter()
        .map(|ptr| {
            let arr = std::ptr::read(*ptr);
            import_array(arr, &(*e.field))
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    Ok(Series::from_chunks_and_dtype_unchecked(
        &field.name,
        chunks,
        &(&field.data_type).into(),
    ))
}

/// # Safety
/// `SeriesExport` must be valid
pub unsafe fn import_series_buffer(e: *mut SeriesExport, len: usize) -> PolarsResult<Vec<Series>> {
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let e = std::ptr::read(e.add(i));
        out.push(import_series(e)?)
    }
    Ok(out)
}

/// # Safety
/// `ArrowArray` and `ArrowSchema` must be valid
unsafe fn import_array(
    array: ffi::ArrowArray,
    schema: &ffi::ArrowSchema,
) -> PolarsResult<ArrayRef> {
    let field = ffi::import_field_from_c(schema)?;
    let out = ffi::import_array_from_c(array, field.data_type)?;
    Ok(out)
}

#[cfg(test)]
mod test {
    use polars_core::prelude::*;

    use super::*;

    #[test]
    fn test_ffi() {
        let s = Series::new("a", [1, 2]);
        let e = export_series(&s);

        unsafe {
            assert_eq!(import_series(e).unwrap(), s);
        };
    }
}
