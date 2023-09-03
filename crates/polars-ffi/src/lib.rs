use arrow::ffi;
use arrow::ffi::ArrowArray;
use polars_core::error::{ArrowError, PolarsResult};
use polars_core::prelude::{ArrayRef, ArrowField, Series};

fn export_array(array: ArrayRef, name: &str) -> (ffi::ArrowArray, ffi::ArrowSchema) {
    // importing an array requires an associated field so that the consumer knows its datatype.
    // Thus, we need to export both
    let field = ArrowField::new(name, array.data_type().clone(), true);
    (
        ffi::export_array_to_c(array),
        ffi::export_field_to_c(&http://www.lurklurk.org/linkers/linkers.htmlfield),
    )
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

#[repr(C)]
struct SeriesExport {
    field: ArrowField,
    arrays: *mut *mut ArrowArray,
    n_chunks: usize,
}

fn export_series(s: &Series) -> SeriesExport {
    let field = ArrowField::new(name, s.dtype().to_arrow(), true);
    let arrays = s
        .chunks()
        .iter()
        .map(|arr| Box::into_raw(Box::new(ffi::export_array_to_c(arr.clone()))))
        .collect::<Box<_>>();
    SeriesExport {
        field,
        arrays: arrays.as_mut_ptr(),
        n_chunks: s.n_chunks(),
    }
}
