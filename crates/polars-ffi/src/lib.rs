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
/// `SeriesExport`
pub unsafe fn import_series(e: SeriesExport) -> PolarsResult<Series> {
    let field = ffi::import_field_from_c(&e.field)?;

    let arrays = Vec::from_raw_parts(e.arrays, e.len, e.len);
    let chunks = arrays
        .into_iter()
        .map(|arr| import_array(arr, &e.field))
        .collect::<PolarsResult<Vec<_>>>()?;
    Ok(Series::from_chunks_and_dtype_unchecked(
        &field.name,
        chunks,
        &(&field.data_type).into(),
    ))
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
