use polars_ffi::*;

use super::*;
use arrow::ffi::{
    ArrowSchema, import_field_from_c
};

pub(super) unsafe fn call_plugin(s: &[Series], lib: &str, symbol: &str) -> PolarsResult<Series> {
    let out = {
        let lib = libloading::Library::new(lib).map_err(|e| {
            PolarsError::ComputeError(format!("error loading dynamic library: {e}").into())
        })?;
        let symbol: libloading::Symbol<
            unsafe extern "C" fn(*const SeriesExport, usize) -> SeriesExport,
        > = lib.get(symbol.as_bytes()).unwrap();

        let n_args = s.len();

        let input = s.iter().map(export_series).collect::<Vec<_>>().into_boxed_slice();
        let slice_ptr = input.as_ptr();
        std::mem::forget(input);

        // let out = symbol(slice_ptr, n_args);
        // let out = import_series(out);
        // out
        slice_ptr
    };
        // dbg!(out);
        drop(out);
        Ok(Series::full_null("", s[0].len(), &DataType::Null))
}

pub(super) unsafe fn plugin_field(fields: &[Field], lib: &str, symbol: &str) -> PolarsResult<Field> {
    let lib = libloading::Library::new(lib).map_err(|e| {
        PolarsError::ComputeError(format!("error loading dynamic library: {e}").into())
    })?;

    let symbol: libloading::Symbol<
        unsafe extern "C" fn(*const ArrowSchema, usize) -> ArrowSchema,
    > = lib.get(symbol.as_bytes()).unwrap();

    let fields = fields.iter().map(|field| arrow::ffi::export_field_to_c(&field.to_arrow())).collect::<Vec<_>>().into_boxed_slice();
    let n_args = fields.len();
    let slice_ptr = fields.as_ptr();
    std::mem::forget(fields);
    let out = symbol(slice_ptr, n_args);

    let arrow_field = import_field_from_c(&out)?;
    Ok(Field::from(&arrow_field))
}
