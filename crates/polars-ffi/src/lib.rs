use arrow::ffi;
use arrow::ffi::{ArrowArray, ArrowSchema};
use polars_core::error::PolarsResult;
use polars_core::prelude::{ArrayRef, ArrowField, Series};

/// An FFI exported `Series`.
#[repr(C)]
pub struct SeriesExport {
    field: ArrowSchema,
    arrays: *mut ArrowArray,
    len: usize,
}

pub fn export_series(s: &Series) -> SeriesExport {
    let field = ArrowField::new(s.name(), s.dtype().to_arrow(), true);
    let schema = ffi::export_field_to_c(&field);
    let mut arrays = s
        .chunks()
        .iter()
        .map(|arr| ffi::export_array_to_c(arr.clone()))
        .collect::<Box<_>>();
    let len = arrays.len();
    let ptr = arrays.as_mut_ptr();
    Box::leak(arrays);
    SeriesExport {
        field: schema,
        arrays: ptr,
        len,
    }
}

/// # Safety
/// `SeriesExport` must be valid
pub unsafe fn import_series(e: SeriesExport) -> PolarsResult<Series> {
    let field = ffi::import_field_from_c(&e.field)?;

    let chunks = (0..e.len).map(|i| {
        let arr = std::ptr::read(e.arrays.add(i));
        import_array(arr, &e.field)
    }).collect::<PolarsResult<Vec<_>>>()?;

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

        let s = vec![s];
        let input = s.iter().map(export_series).collect::<Vec<_>>().into_boxed_slice();
        let slice_ptr = input.as_ptr();
        std::mem::forget(input);


        let len = 1;
        unsafe {
            let inputs = Vec::from_raw_parts(slice_ptr as *mut SeriesExport, len, len);
            drop(inputs);
        }
        drop(s);


        // unsafe {
        //     assert_eq!(import_series(e).unwrap(), s);
        // };
    }
}
