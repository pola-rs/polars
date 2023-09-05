use polars_ffi::*;

use super::*;

pub(super) fn call_plugin(s: &[Series], lib: &str, symbol: &str) -> PolarsResult<Series> {
    unsafe {
        let lib = libloading::Library::new(lib).map_err(|e| {
            PolarsError::ComputeError(format!("error loading dynamic library: {e}").into())
        })?;
        let symbol: libloading::Symbol<
            unsafe extern "C" fn(*const SeriesExport, usize) -> SeriesExport,
        > = lib.get(symbol.as_bytes()).unwrap();

        let n_args = s.len();

        let input = s.iter().map(export_series).collect::<Vec<_>>();
        let slice_ptr = input.as_ptr();

        let out = symbol(slice_ptr, n_args);
        import_series(out)
    }
}
